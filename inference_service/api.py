from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import threading
import time
from datetime import datetime

from sentinelvision_core import AnomalyDetector, DataProcessor
from monitoring.monitoring_service import get_monitoring_service
from ml_pipeline.retraining import get_retraining_manager, get_retraining_scheduler


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
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.mount("/static", StaticFiles(directory="inference_service/static"), name="static")

model: Optional[AnomalyDetector] = None
processor: Optional[DataProcessor] = None
monitoring_service = None
model_metadata: Dict[str, Any] = {}
current_model_version: Optional[str] = None
reload_lock = threading.Lock()
model_reload_thread: Optional[threading.Thread] = None
stop_reload_event = threading.Event()


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SentinelVision AI Platform</title>
        <link rel="stylesheet" type="text/css" href="/static/swagger.css">
        <link rel="icon" type="image/x-icon" href="/favicon.ico">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js" charset="UTF-8"></script>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js" charset="UTF-8"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: "/openapi.json",
                    dom_id: "#swagger-ui",
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    layout: "StandaloneLayout",
                    docExpansion: "list",
                    filter: true,
                    showExtensions: true,
                    showCommonExtensions: true,
                    syntaxHighlight: {
                        activate: true,
                        theme: "monokai"
                    },
                    tryItOutEnabled: true,
                    persistAuthorization: true,
                    oauth2RedirectUrl: "/docs/oauth2-redirect"
                });
                window.ui = ui;
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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


class PredictionResponse(BaseModel):
    predictions: List[bool]
    scores: List[float]
    model_version: Optional[str] = None
    model_reloaded: Optional[bool] = None


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


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
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


@app.get("/model/info", tags=["Model Management"])
async def model_info():
    return {
        "model_name": model_metadata.get("model_name", "unknown"),
        "version": model_metadata.get("version", "unknown"),
        "created_at": model_metadata.get("created_at", "unknown"),
        "metrics": model_metadata.get("metrics", {})
    }


@app.get("/model/reload", tags=["Model Management"])
async def trigger_model_reload():
    reloaded = reload_model_if_needed()
    return {
        "success": True,
        "model_reloaded": reloaded,
        "current_version": current_model_version
    }


@app.get("/retraining/status", tags=["Retraining"])
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


@app.post("/retraining/trigger", tags=["Retraining"])
async def trigger_retraining():
    try:
        from ml_pipeline.retraining import get_retraining_scheduler
        scheduler = get_retraining_scheduler()
        result = scheduler.force_retrain()
        
        reload_model_if_needed()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/metrics", tags=["Monitoring"])
async def get_metrics(window_minutes: Optional[int] = None):
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_metrics_summary(window_minutes)


@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_drift_status()


@app.get("/monitoring/status", tags=["Monitoring"])
async def get_full_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_full_status()


@app.get("/health", tags=["System"])
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
