from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import threading
import time
from datetime import datetime

from sentinelvision_core import AnomalyDetector, DataProcessor
from monitoring.monitoring_service import get_monitoring_service
from ml_pipeline.retraining import get_retraining_manager, get_retraining_scheduler


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MODEL_REGISTRY_PATH = Path("model_registry")
MODEL_RELOAD_INTERVAL = 60

app = FastAPI(
    title="SentinelVision AI Platform",
    description="""
# SentinelVision - Production ML Platform

SentinelVision is an end-to-end ML platform that manages training pipelines, model registry, inference APIs, monitoring, drift detection, and automated retraining.

## System Overview

```
Training Pipeline → Model Registry → Inference API → Monitoring → Drift Detection → Retraining
```

## Features

- **Training Pipeline** - Automated model training with evaluation
- **Model Registry** - Versioned model storage with metadata
- **Inference API** - FastAPI-based prediction service
- **Monitoring** - Real-time metrics and prediction logging
- **Drift Detection** - Automatic detection of data/concept drift
- **Automated Retraining** - Self-healing model updates based on monitoring signals
""",
    version="1.0",
    contact={
        "name": "SentinelVision Platform",
        "url": "https://github.com/DastanZar/SentinelVision"
    },
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="SentinelVision AI Platform",
        swagger_ui_parameters={
            "syntaxHighlight": {"theme": "monokai"},
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "docExpansion": "list",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
        },
        swagger_css_url="/static/swagger.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc():
    from fastapi.openapi.docs import get_redoc_html
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="SentinelVision AI Platform - ReDoc"
    )

model: Optional[AnomalyDetector] = None
processor: Optional[DataProcessor] = None
monitoring_service = None
model_metadata: Dict[str, Any] = {}
current_model_version: Optional[str] = None
reload_lock = threading.Lock()
model_reload_thread: Optional[threading.Thread] = None
stop_reload_event = threading.Event()


def find_latest_model_version() -> Path:
    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Model registry not found at {MODEL_REGISTRY_PATH}")
    
    latest_file = MODEL_REGISTRY_PATH / "latest.txt"
    if latest_file.exists():
        with open(latest_file, "r") as f:
            latest_version_name = f.read().strip()
        latest_version = MODEL_REGISTRY_PATH / latest_version_name
        if latest_version.exists() and latest_version.is_dir():
            return latest_version
    
    versions = [d for d in MODEL_REGISTRY_PATH.iterdir() if d.is_dir() and d.name.startswith("model_v")]
    if not versions:
        raise FileNotFoundError("No model versions found in registry")
    
    latest = max(versions, key=lambda x: int(x.name.replace("model_v", "")))
    return latest


def load_model_from_registry():
    global model, processor, monitoring_service, model_metadata, current_model_version
    
    try:
        latest_version = find_latest_model_version()
        model_path = latest_version / "model.pkl"
        metadata_path = latest_version / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
            current_model_version = model_metadata.get("version")
            print(f"Loading model version: {current_model_version}")
        
        model = AnomalyDetector.load(str(model_path))
        processor = DataProcessor()
        monitoring_service = get_monitoring_service()
        
    except Exception as e:
        print(f"Warning: Could not load model from registry: {e}")
        model = AnomalyDetector()
        processor = DataProcessor()
        monitoring_service = get_monitoring_service()


def reload_model_if_needed():
    global model, processor, model_metadata, current_model_version
    
    with reload_lock:
        try:
            latest_version = find_latest_model_version()
            metadata_path = latest_version / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    new_metadata = json.load(f)
                
                new_version = new_metadata.get("version")
                
                if new_version != current_model_version:
                    print(f"[ModelReload] New model version detected: {new_version} (current: {current_model_version})")
                    
                    model_path = latest_version / "model.pkl"
                    model = AnomalyDetector.load(str(model_path))
                    model_metadata = new_metadata
                    current_model_version = new_version
                    
                    print(f"[ModelReload] Model reloaded successfully to version {new_version}")
                    return True
                    
        except Exception as e:
            print(f"[ModelReload] Error checking for new model: {e}")
    
    return False


def model_reload_worker():
    while not stop_reload_event.is_set():
        time.sleep(MODEL_RELOAD_INTERVAL)
        if not stop_reload_event.is_set():
            reload_model_if_needed()


class PredictionRequest(BaseModel):
    data: List[List[float]]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [[0.21, 0.45, 0.12, 0.67, 0.33, 0.89, 0.15, 0.42, 0.78, 0.56]]
                },
                {
                    "data": [
                        [1.2, 3.4, 5.6, 2.1, 4.3],
                        [0.5, 0.8, 0.3, 0.9, 0.7]
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predictions: List[bool]
    scores: List[float]
    model_version: Optional[str] = None
    model_reloaded: Optional[bool] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predictions": [True],
                    "scores": [0.87],
                    "model_version": "v1"
                },
                {
                    "predictions": [False, True, False],
                    "scores": [0.23, 0.91, 0.15],
                    "model_version": "v2",
                    "model_reloaded": True
                }
            ]
        }
    }


@app.on_event("startup")
async def startup_event():
    load_model_from_registry()
    
    global model_reload_thread
    model_reload_thread = threading.Thread(target=model_reload_worker, daemon=True)
    model_reload_thread.start()
    print(f"[Startup] Model reload worker started (interval: {MODEL_RELOAD_INTERVAL}s)")
    
    try:
        scheduler = get_retraining_scheduler()
        scheduler.start()
        print("[Startup] Retraining scheduler started")
    except Exception as e:
        print(f"[Startup] Could not start retraining scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global stop_reload_event
    stop_reload_event.set()
    print("[Shutdown] Model reload worker stopped")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Run anomaly detection inference on input data",
    description="""
## Prediction API

Run anomaly detection inference on input data. This endpoint processes input features 
through the deployed ML model and returns predictions with confidence scores.

### Features:
- Real-time anomaly detection
- Confidence scores for each prediction
- Automatic prediction logging for monitoring
- Automatic model version tracking
"""
)
async def predict(request: PredictionRequest):
    global model, processor, monitoring_service, model_metadata
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    import numpy as np
    X = np.array(request.data)
    X_processed = processor.transform(X)
    
    result = model.predict(X_processed)
    
    predictions = result.get("predictions", [])
    scores = result.get("scores", [])
    
    if monitoring_service:
        for i, (input_data, pred, score) in enumerate(zip(request.data, predictions, scores)):
            monitoring_service.record_prediction_event(
                input_data=input_data,
                prediction=pred,
                confidence_score=score,
                model_version=model_metadata.get("version", "unknown")
            )
    
    return PredictionResponse(
        predictions=predictions,
        scores=scores,
        model_version=model_metadata.get("version")
    )


@app.get(
    "/model/info",
    tags=["Model Management"],
    summary="Get information about the currently deployed model",
    description="""
## Model Management

Retrieve detailed information about the currently deployed model version.

### Returns:
- Model name and version
- Creation timestamp
- Training metrics (accuracy, precision, recall, F1 score)
"""
)
async def model_info():
    return {
        "model_name": model_metadata.get("model_name", "unknown"),
        "version": model_metadata.get("version", "unknown"),
        "created_at": model_metadata.get("created_at", "unknown"),
        "metrics": model_metadata.get("metrics", {})
    }


@app.get(
    "/model/reload",
    tags=["Model Management"],
    summary="Force reload of model from registry",
    description="""
## Force Model Reload

Manually trigger a reload of the model from the registry. This checks for new model 
versions and hot-swaps the current model if a newer version is available.
"""
)
async def trigger_model_reload():
    reloaded = reload_model_if_needed()
    return {
        "success": True,
        "model_reloaded": reloaded,
        "current_version": current_model_version
    }


@app.get(
    "/retraining/status",
    tags=["Retraining"],
    summary="Get retraining scheduler and manager status",
    description="""
## Retraining Status

Retrieve the current status of the automated retraining system including:
- Scheduler configuration and state
- Last retraining timestamp
- Current model version
- Retraining thresholds
"""
)
async def get_retraining_status():
    try:
        from ml_pipeline.retraining import get_retraining_scheduler
        scheduler = get_retraining_scheduler()
        manager = get_retraining_manager()
        return {
            "scheduler": scheduler.get_status(),
            "manager": manager.get_retraining_history()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/retraining/trigger",
    tags=["Retraining"],
    summary="Manually trigger the retraining pipeline",
    description="""
## Trigger Retraining

Manually trigger the automated retraining pipeline. This will:
1. Run the training pipeline with current data
2. Save the new model version to the registry
3. Reload the inference service with the new model
"""
)
async def trigger_retraining():
    try:
        from ml_pipeline.retraining import get_retraining_scheduler
        scheduler = get_retraining_scheduler()
        result = scheduler.force_retrain()
        
        reload_model_if_needed()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/monitoring/metrics",
    tags=["Monitoring"],
    summary="Retrieve monitoring metrics for recent predictions",
    description="""
## Monitoring Metrics

Retrieve aggregated monitoring metrics for recent predictions.

### Query Parameters:
- `window_minutes` (optional): Time window for metrics aggregation

### Returns:
- Prediction distribution (normal vs anomaly)
- Anomaly rate
- Confidence score statistics (mean, std, percentiles)
"""
)
async def get_metrics(window_minutes: Optional[int] = None):
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_metrics_summary(window_minutes)


@app.get(
    "/monitoring/drift",
    tags=["Monitoring"],
    summary="Get drift detection status",
    description="""
## Drift Detection

Retrieve the current drift detection status including:
- Prediction drift score
- Anomaly rate drift from baseline
- Confidence drift
- Whether drift threshold has been exceeded
"""
)
async def get_drift_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_drift_status()


@app.get(
    "/monitoring/status",
    tags=["Monitoring"],
    summary="Get full monitoring system status",
    description="""
## Full System Status

Get a comprehensive view of the monitoring system including:
- All metrics summaries
- Drift detection status
- Time-series metrics
"""
)
async def get_full_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_full_status()


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="""
## System Health

Check the health status of the inference service including:
- Service status
- Model loading status
- Current model version
- Auto-reload configuration
"""
)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_metadata.get("version"),
        "auto_reload_enabled": True,
        "reload_interval_seconds": MODEL_RELOAD_INTERVAL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
