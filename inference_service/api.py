from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from sentinelvision_core import AnomalyDetector, DataProcessor


app = FastAPI(title="SentinelVision Inference API")

model: Optional[AnomalyDetector] = None
processor: Optional[DataProcessor] = None


class PredictionRequest(BaseModel):
    data: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[bool]
    scores: List[float]


@app.on_event("startup")
async def load_model():
    global model, processor
    model = AnomalyDetector.load("models/anomaly_detector.pkl")
    processor = DataProcessor()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    import numpy as np
    X = np.array(request.data)
    X_processed = processor.transform(X)
    
    result = model.predict(X_processed)
    
    return PredictionResponse(
        predictions=result.get("predictions", []),
        scores=result.get("scores", [])
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
