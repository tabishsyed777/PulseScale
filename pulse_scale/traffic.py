"""Synthetic service telemetry and forecast features."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (0, 1, 5, 15, 30, 60)
WINDOWS = (5, 15, 60)


def make_traffic(minutes: int = 14 * 24 * 60, seed: int = 23) -> pd.DataFrame:
    """Create repeatable minute-level synthetic request-rate telemetry."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=minutes, freq="min", tz="UTC")
    t = np.arange(minutes, dtype=float)
    hour = (t % 1440) / 1440
    day = (t // 1440).astype(int)
    daily = 42 + 22 * np.sin(2 * np.pi * (hour - 0.30))
    workday = np.where(day % 7 < 5, 1.0, 0.68)
    shoulder = 15 * np.exp(-0.5 * ((hour - 0.57) / 0.11) ** 2)
    slow = 7 * np.sin(2 * np.pi * t / (1440 * 5))
    noise = rng.normal(0, 4.0, minutes)
    demand = (daily + shoulder + slow) * workday + noise
    # A handful of bursts create short-lived demand shifts.
    for start in rng.choice(np.arange(90, max(91, minutes - 45)), size=max(1, minutes // 7000), replace=False):
        width = int(rng.integers(8, 32))
        height = float(rng.uniform(25, 65))
        end = min(minutes, int(start + width))
        demand[start:end] += height * np.exp(-np.linspace(0, 2.5, end - start))
    return pd.DataFrame({"timestamp": idx, "requests_per_second": np.clip(demand, 0, None)})


def feature_row(history: np.ndarray, timestamp: pd.Timestamp) -> np.ndarray:
    """Build one model row from 60 or more recent observations."""
    values = np.asarray(history, dtype=float)
    if len(values) < 60:
        raise ValueError("At least 60 recent observations are required.")
    row = [float(values[-1 - lag]) for lag in LAGS]
    for window in WINDOWS:
        row.append(float(np.mean(values[-window:])))
    stamp = pd.Timestamp(timestamp)
    minute_of_day = stamp.hour * 60 + stamp.minute
    row.extend([
        float(np.sin(2 * np.pi * minute_of_day / 1440)),
        float(np.cos(2 * np.pi * minute_of_day / 1440)),
        float(np.sin(2 * np.pi * stamp.dayofweek / 7)),
        float(np.cos(2 * np.pi * stamp.dayofweek / 7)),
    ])
    return np.asarray(row, dtype=float)


def supervised(frame: pd.DataFrame, horizon: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Create a direct horizon-ahead supervised learning dataset."""
    values = frame["requests_per_second"].to_numpy(dtype=float)
    stamps = pd.DatetimeIndex(frame["timestamp"])
    start = 60
    stop = len(values) - horizon
    rows = [feature_row(values[: i + 1], stamps[i]) for i in range(start, stop)]
    targets = values[start + horizon : stop + horizon]
    return np.vstack(rows), targets
