"""Train and serve a synthetic-demand forecasting model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from pulse_scale.traffic import feature_row, make_traffic, supervised


@dataclass
class Forecast:
    point: float
    lower: float
    upper: float
    mae: float
    residual_radius: float


class DemandForecaster:
    """Gradient-boosted direct forecast with a chronological residual interval."""

    def __init__(self, horizon: int = 15, seed: int = 23):
        self.horizon = horizon
        self.seed = seed
        self.model: HistGradientBoostingRegressor | None = None
        self.residual_radius = 0.0
        self.validation_mae = 0.0

    def fit_demo(self) -> "DemandForecaster":
        frame = make_traffic(seed=self.seed)
        x, y = supervised(frame, horizon=self.horizon)
        split = int(len(y) * 0.82)
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=120, max_leaf_nodes=20,
            l2_regularization=1.5, random_state=self.seed
        )
        self.model.fit(x[:split], y[:split])
        validation_prediction = self.model.predict(x[split:])
        errors = np.abs(y[split:] - validation_prediction)
        self.validation_mae = float(mean_absolute_error(y[split:], validation_prediction))
        # Finite-sample 90% absolute residual quantile, for a demo interval.
        rank = min(len(errors), max(1, math.ceil((len(errors) + 1) * 0.90)))
        self.residual_radius = float(np.sort(errors)[rank - 1])
        return self

    def predict(self, history: list[float], timestamp: pd.Timestamp) -> Forecast:
        if self.model is None:
            raise RuntimeError("The forecaster has not been fitted.")
        row = feature_row(np.asarray(history, dtype=float), timestamp).reshape(1, -1)
        point = max(0.0, float(self.model.predict(row)[0]))
        return Forecast(
            point=point,
            lower=max(0.0, point - self.residual_radius),
            upper=point + self.residual_radius,
            mae=self.validation_mae,
            residual_radius=self.residual_radius,
        )
