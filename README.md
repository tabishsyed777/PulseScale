# PulseScale

**Created by Tabish Syed**

PulseScale is an ML-assisted autoscaling workbench for containerized services. It forecasts request demand, turns that forecast into a bounded replica recommendation, and falls back to a reactive rule when the prediction is unsafe or unavailable.

The project focuses on the operational path around a model: reproducible data, model artifact creation, an API, health and metrics endpoints, Docker, Kubernetes manifests, and a CI workflow.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn pulse_scale.api:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the interactive simulator. API docs are at [http://localhost:8000/docs](http://localhost:8000/docs), health is at `/health`, and Prometheus-format metrics are at `/metrics`.

Or run the container:

```bash
docker compose up --build
```

## Screenshots

![PulseScale workload forecast and scaling policy dashboard](assets/pulsescale-dashboard.png)

## What the demo does

- Generates a repeatable synthetic traffic history with daily cycles, weekday variation, bursts, and noise.
- Trains a gradient-boosting model on lag and rolling-window features to forecast demand 15 minutes ahead.
- Estimates a residual interval on a separate time segment.
- Converts the upper forecast bound to a replica recommendation using a configurable per-replica capacity.
- Applies minimum/maximum replica bounds and a maximum scale step.
- Uses a reactive current-load rule if the forecast interval is too wide or the model is unavailable.
- Replays the same traffic through a reactive baseline and the forecast policy, comparing overload minutes and replica-minutes.

The demo history and model are synthetic. This is a learning project, not a production autoscaler or a validated capacity plan.

## API

`GET /demo` returns a synthetic recent history and current replica count for the UI.

`POST /recommend` accepts a recent history:

```json
{
  "history": [
    {"timestamp": "2026-10-30T10:00:00Z", "requests_per_second": 84.2}
  ],
  "current_replicas": 3,
  "min_replicas": 1,
  "max_replicas": 20,
  "capacity_per_replica": 40
}
```

Provide at least 60 ordered observations with non-negative `requests_per_second`. The response includes the point forecast, residual interval, policy, and replica recommendation. See the interactive API docs for the complete schema.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PULSE_FORECAST_HORIZON_MINUTES` | `15` | Forecast horizon |
| `PULSE_MAX_INTERVAL_RATIO` | `0.90` | Wider intervals trigger the reactive fallback |
| `PULSE_DEFAULT_CAPACITY` | `40` | Requests per second per replica for the demo policy |

## Deploy to Kubernetes

The manifests in `k8s/` create a Deployment and ClusterIP Service. Build and publish the image to a registry you control, then change the image in `k8s/deployment.yaml` before applying:

```bash
kubectl apply -f k8s/
kubectl port-forward service/pulsescale 8000:80
```

The API returns recommendations; it does not change a cluster's replica count. A real controller needs workload-specific capacity measurements, authentication, authorization, rollout safeguards, and an explicit integration with the target platform.

## Model and operational limits

PulseScale uses a supervised gradient-boosting regressor over recent demand lags, rolling averages, and calendar features. A chronological tail is held aside to estimate an absolute residual interval. The policy uses the interval's upper bound to reduce under-provisioning risk, then applies scale and replica limits. If the model is uncertain, it uses a threshold-based current-load rule.

The interval is not a guarantee: traffic can shift, training data can be unrepresentative, and forecast errors can be correlated. Synthetic replay scores illustrate behavior only. Before any real deployment, retrain and evaluate on representative service telemetry, measure per-replica capacity, and add service-specific safeguards.

## License

MIT. See [LICENSE](LICENSE).
