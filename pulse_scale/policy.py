"""Guarded conversion from demand estimates to replica recommendations."""
from __future__ import annotations

import math

from pulse_scale.model import DemandForecaster


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def reactive_replicas(current_rps: float, capacity_per_replica: float, minimum: int, maximum: int) -> int:
    """Simple reactive rule with a 70% target utilization."""
    desired = math.ceil(current_rps / max(capacity_per_replica * 0.70, 1e-9))
    return clamp(desired, minimum, maximum)


def recommend(
    forecaster: DemandForecaster | None,
    history: list[float],
    timestamp,
    current_replicas: int,
    minimum: int = 1,
    maximum: int = 20,
    capacity_per_replica: float = 40.0,
    max_step: int = 3,
    max_interval_ratio: float = 0.90,
) -> dict:
    """Choose forecast or reactive policy, then apply scale-step and replica bounds."""
    if minimum < 1 or maximum < minimum:
        raise ValueError("Replica bounds must satisfy 1 <= min_replicas <= max_replicas.")
    if capacity_per_replica <= 0:
        raise ValueError("capacity_per_replica must be positive.")
    if max_step < 1:
        raise ValueError("max_step must be at least one.")

    fallback_target = reactive_replicas(history[-1], capacity_per_replica, minimum, maximum)
    forecast = None
    policy = "reactive-fallback"
    reason = "Forecast unavailable."
    desired = fallback_target
    if forecaster is not None:
        try:
            forecast = forecaster.predict(history, timestamp)
            ratio = (forecast.upper - forecast.lower) / max(forecast.point, 1.0)
            if ratio <= max_interval_ratio:
                desired = math.ceil(forecast.upper / (capacity_per_replica * 0.70))
                policy = "forecast"
                reason = "Upper forecast bound is within the uncertainty guardrail."
            else:
                reason = "Forecast interval exceeded the uncertainty guardrail."
        except (ValueError, RuntimeError, FloatingPointError):
            reason = "Forecast failed; using current-load rule."

    desired = clamp(desired, minimum, maximum)
    stepped = max(current_replicas - max_step, min(current_replicas + max_step, desired))
    replicas = clamp(stepped, minimum, maximum)
    return {
        "policy": policy,
        "reason": reason,
        "recommended_replicas": replicas,
        "unconstrained_target": desired,
        "current_replicas": current_replicas,
        "forecast": None if forecast is None else {
            "horizon_minutes": forecaster.horizon,
            "point_rps": round(forecast.point, 3),
            "lower_rps": round(forecast.lower, 3),
            "upper_rps": round(forecast.upper, 3),
            "validation_mae_rps": round(forecast.mae, 3),
            "residual_radius_rps": round(forecast.residual_radius, 3),
        },
        "guardrails": {
            "min_replicas": minimum, "max_replicas": maximum,
            "max_step": max_step, "capacity_per_replica_rps": capacity_per_replica,
        },
    }
