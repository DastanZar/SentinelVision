# SentinelVision

![Python](https://img.shields.io/badge/python-3.11-blue) ![MLflow](https://img.shields.io/badge/MLflow-tracking-orange) ![PyOD](https://img.shields.io/badge/PyOD-anomaly--detection-green) ![CI](https://github.com/DastanZar/SentinelVision/actions/workflows/ci.yml/badge.svg)

Tabular anomaly detection MLOps platform — automated training, drift detection, versioned model registry, and production-ready inference serving on Azure Kubernetes.

## What This Is

SentinelVision is a production-style MLOps system built around tabular anomaly detection. It implements the full ML lifecycle: data ingestion, model training with experiment tracking, versioned artifact storage, drift-triggered automated retraining, and a FastAPI inference service with hot-reload support.

The anomaly detection core uses [PyOD](https://github.com/yzhao062/pyod) — specifically IsolationForest and ECOD — which are state-of-the-art unsupervised outlier detection algorithms. The system supports both supervised evaluation (precision/recall/F1/ROC-AUC via sklearn) when labels are available, and unsupervised evaluation (anomaly ratio, score statistics) when they are not.

## Architecture

```
sentinelvision_core/     # PyOD-based AnomalyDetector (IForest, ECOD)
training/                # Training pipeline with MLflow experiment tracking
inference_service/       # FastAPI serving with hot-reload and model versioning
monitoring/              # PSI-based drift detection, prediction logging, metrics aggregation
ml_pipeline/             # Automated retraining scheduler triggered by drift signals
model_registry/          # Versioned model artifacts (model.pkl, metrics.json, metadata.json)
infra/                   # Docker Compose + Kubernetes manifests
tests/                   # pytest suite covering detector, training, and API
```

## Core Components

### Anomaly Detector

The `AnomalyDetector` class wraps PyOD models with a consistent interface:

- `fit(X)` — trains the detector on a NumPy array
- `predict(X)` — returns predictions (0=normal, 1=anomaly), anomaly scores, and decision threshold
- `save(path)` / `load(path)` — joblib serialization with full state preservation
- Supports `iforest` (IsolationForest) and `ecod` (ECOD) via `model_type` config

### Training Pipeline

The training pipeline (`training/train_pipeline.py`) requires a real CSV dataset — there is no synthetic data fallback by design.

```bash
# Unsupervised (no labels)
python training/train_pipeline.py data/train.csv

# Supervised (with label column)
python training/train_pipeline.py data/train.csv label
```

Each run:
1. loads and validates the CSV
2. StandardScaler preprocessing
3. trains the PyOD detector
4. evaluates: precision/recall/F1/ROC-AUC (supervised) or anomaly ratio/score stats (unsupervised)
5. logs all params, metrics, and model artifact to MLflow
6. saves versioned artifact to `model_registry/model_vN/`
7. updates `model_registry/latest.txt` pointer

### Inference Service

```bash
python inference_service/api.py
```

Endpoints:
- `POST /predict` — accepts `{"data": [[f1, f2, ...]]}`, returns predictions and anomaly scores
- `GET /health` — service status and loaded model version
- `GET /model/info` — current model metadata
- `POST /retraining/trigger` — manually trigger retraining
- `GET /monitoring/metrics` — aggregated prediction metrics
- `GET /monitoring/drift` — current drift status

The service hot-reloads the model from registry every 60 seconds without restart.

### Monitoring and Drift Detection

Drift detection uses **Population Stability Index (PSI)** on prediction distributions. When drift exceeds threshold (default: 0.05) or anomaly rate spikes beyond 0.3, the retraining scheduler automatically triggers a new training run and promotes the new model version.

## Running Locally

```bash
# Install
pip install -e .
pip install -r requirements.txt

# Train a model (requires a CSV)
python training/train_pipeline.py path/to/data.csv

# Start inference service
python inference_service/api.py

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.5, 1.2, -0.3, 0.8, 2.1]]}'

# Run tests
pytest tests/ -v
```

## Docker Compose

```bash
cd infra/docker
docker-compose up -d
```

Starts inference service on port 8000 and monitoring service on port 8001.

## Kubernetes

```bash
kubectl apply -f infra/kubernetes/namespace.yaml
kubectl apply -f infra/kubernetes/pvc.yaml
kubectl apply -f infra/kubernetes/inference-deployment.yaml
kubectl apply -f infra/kubernetes/monitoring-deployment.yaml
```

3 replicas for inference, 2 for monitoring, with PersistentVolumeClaim for model storage.

## MLflow Tracking

All training runs are tracked under the `sentinelvision-anomaly-training` experiment. Each run logs:
- **Params:** model_type, contamination, data_path
- **Metrics:** precision, recall, f1_score, roc_auc (supervised) or anomaly_ratio, score_mean/std (unsupervised)
- **Tags:** project, model_type, dataset
- **Artifacts:** versioned model directory

```bash
mlflow ui
# Open http://localhost:5000
```

## Tests

```bash
pytest tests/ -v
```

Covers: fit/predict correctness, save/load round-trip, predict-before-fit error handling, ECOD model support, invalid model type validation, anomaly ratio vs contamination tolerance.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_REGISTRY_PATH` | `model_registry` | Path to versioned model artifacts |
| `DRIFT_THRESHOLD` | `0.05` | PSI threshold triggering retraining |
| `ANOMALY_RATE_THRESHOLD` | `0.3` | Anomaly spike threshold |
| `MIN_TRAINING_INTERVAL_HOURS` | `1` | Minimum hours between retraining runs |
