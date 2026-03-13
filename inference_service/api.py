from pathlib import Path
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="SentinelVision Inference API")

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


@app.post("/predict", response_model=PredictionResponse)
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


@app.get("/model/info")
async def model_info():
    return {
        "model_name": model_metadata.get("model_name", "unknown"),
        "version": model_metadata.get("version", "unknown"),
        "created_at": model_metadata.get("created_at", "unknown"),
        "metrics": model_metadata.get("metrics", {})
    }


@app.get("/model/reload")
async def trigger_model_reload():
    reloaded = reload_model_if_needed()
    return {
        "success": True,
        "model_reloaded": reloaded,
        "current_version": current_model_version
    }


@app.get("/retraining/status")
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


@app.post("/retraining/trigger")
async def trigger_retraining():
    try:
        from ml_pipeline.retraining import get_retraining_scheduler
        scheduler = get_retraining_scheduler()
        result = scheduler.force_retrain()
        
        reload_model_if_needed()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/metrics")
async def get_metrics(window_minutes: Optional[int] = None):
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_metrics_summary(window_minutes)


@app.get("/monitoring/drift")
async def get_drift_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_drift_status()


@app.get("/monitoring/status")
async def get_full_status():
    if monitoring_service is None:
        raise HTTPException(status_code=500, detail="Monitoring service not initialized")
    return monitoring_service.get_full_status()


@app.get("/health")
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
