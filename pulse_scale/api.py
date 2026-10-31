"""HTTP interface for forecasts, guarded recommendations, and replay metrics."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pulse_scale.model import DemandForecaster
from pulse_scale.policy import recommend, reactive_replicas
from pulse_scale.traffic import make_traffic

HORIZON = int(os.getenv("PULSE_FORECAST_HORIZON_MINUTES", "15"))
MAX_INTERVAL_RATIO = float(os.getenv("PULSE_MAX_INTERVAL_RATIO", "0.90"))
DEFAULT_CAPACITY = float(os.getenv("PULSE_DEFAULT_CAPACITY", "40"))
forecaster: DemandForecaster | None = None
counters = {"recommendations": 0, "fallbacks": 0, "request_seconds_sum": 0.0}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global forecaster
    forecaster = DemandForecaster(horizon=HORIZON).fit_demo()
    yield
    forecaster = None


app = FastAPI(
    title="PulseScale API",
    description="ML-assisted traffic forecasting and bounded replica recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Observation(BaseModel):
    timestamp: datetime | None = None
    requests_per_second: float = Field(ge=0, le=1_000_000)


class RecommendationRequest(BaseModel):
    history: list[Observation] = Field(min_length=60, max_length=5000)
    current_replicas: int = Field(ge=1, le=1000)
    min_replicas: int = Field(default=1, ge=1, le=1000)
    max_replicas: int = Field(default=20, ge=1, le=1000)
    capacity_per_replica: float = Field(default=DEFAULT_CAPACITY, gt=0, le=1_000_000)


def _ordered_history(observations: list[Observation]) -> tuple[list[float], pd.Timestamp]:
    if len(observations) < 60:
        raise HTTPException(status_code=422, detail="At least 60 observations are required.")
    stamps = [item.timestamp for item in observations if item.timestamp is not None]
    if stamps and len(stamps) != len(observations):
        raise HTTPException(status_code=422, detail="Either provide timestamps for every row or omit them.")
    if stamps and any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise HTTPException(status_code=422, detail="Timestamps must be strictly increasing.")
    timestamp = pd.Timestamp(stamps[-1]) if stamps else pd.Timestamp.now(tz="UTC")
    return [item.requests_per_second for item in observations], timestamp


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok" if forecaster is not None else "starting", "model_ready": forecaster is not None}


@app.get("/demo")
def demo():
    frame = make_traffic(minutes=360, seed=92).tail(120).reset_index(drop=True)
    history = [
        {"timestamp": stamp.isoformat(), "requests_per_second": round(float(value), 3)}
        for stamp, value in zip(frame["timestamp"], frame["requests_per_second"])
    ]
    return {"history": history, "current_replicas": 3, "capacity_per_replica": DEFAULT_CAPACITY}


@app.post("/recommend")
def get_recommendation(payload: RecommendationRequest, request: Request):
    started = time.perf_counter()
    if payload.min_replicas > payload.max_replicas:
        raise HTTPException(status_code=422, detail="min_replicas must not exceed max_replicas.")
    history, timestamp = _ordered_history(payload.history)
    try:
        result = recommend(
            forecaster, history, timestamp, payload.current_replicas,
            minimum=payload.min_replicas, maximum=payload.max_replicas,
            capacity_per_replica=payload.capacity_per_replica,
            max_interval_ratio=MAX_INTERVAL_RATIO,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed = time.perf_counter() - started
    counters["recommendations"] += 1
    counters["fallbacks"] += int(result["policy"] != "forecast")
    counters["request_seconds_sum"] += elapsed
    result["request_duration_seconds"] = round(elapsed, 6)
    return result


@app.get("/replay")
def replay():
    """Compare policies on one deterministic synthetic traffic segment."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model is starting.")
    frame = make_traffic(minutes=10 * 60, seed=123)
    values = frame["requests_per_second"].to_numpy()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    current_ml = current_reactive = 3
    capacity = DEFAULT_CAPACITY
    overloaded_ml = overloaded_reactive = replica_minutes_ml = replica_minutes_reactive = 0.0
    decisions = 0
    for i in range(60, len(values) - HORIZON, 5):
        observed = values[i]
        recent = values[i - 59 : i + 1].tolist()
        ml_decision = recommend(
            forecaster, recent, stamps[i], current_ml,
            capacity_per_replica=capacity, maximum=20, max_step=3,
            max_interval_ratio=MAX_INTERVAL_RATIO,
        )
        current_ml = ml_decision["recommended_replicas"]
        current_reactive = reactive_replicas(observed, capacity, 1, 20)
        future = values[i + 1 : i + 6]
        overloaded_ml += float(np.sum(future > current_ml * capacity))
        overloaded_reactive += float(np.sum(future > current_reactive * capacity))
        replica_minutes_ml += current_ml * len(future)
        replica_minutes_reactive += current_reactive * len(future)
        decisions += 1
    minutes = decisions * 5
    return {
        "synthetic": True,
        "minutes": minutes,
        "forecast_policy": {
            "overloaded_minutes": int(overloaded_ml),
            "replica_minutes": int(replica_minutes_ml),
            "average_replicas": round(replica_minutes_ml / minutes, 2),
        },
        "reactive_baseline": {
            "overloaded_minutes": int(overloaded_reactive),
            "replica_minutes": int(replica_minutes_reactive),
            "average_replicas": round(replica_minutes_reactive / minutes, 2),
        },
        "note": "A deterministic synthetic replay, not a production performance claim.",
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    average = counters["request_seconds_sum"] / max(counters["recommendations"], 1)
    body = (
        "# HELP pulsescale_recommendations_total Recommendation API calls\n"
        "# TYPE pulsescale_recommendations_total counter\n"
        f"pulsescale_recommendations_total {counters['recommendations']}\n"
        "# HELP pulsescale_fallbacks_total Recommendations using the reactive fallback\n"
        "# TYPE pulsescale_fallbacks_total counter\n"
        f"pulsescale_fallbacks_total {counters['fallbacks']}\n"
        "# HELP pulsescale_recommendation_duration_seconds Average recommendation duration\n"
        "# TYPE pulsescale_recommendation_duration_seconds gauge\n"
        f"pulsescale_recommendation_duration_seconds {average:.8f}\n"
    )
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


@app.get("/ready")
def ready():
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {"ready": True, "checked_at": datetime.now(timezone.utc).isoformat()}
