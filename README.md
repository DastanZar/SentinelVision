# SentinelVision

An ML-powered anomaly detection system with full MLOps lifecycle management.

## ML Lifecycle Pipeline

SentinelVision implements a complete ML lifecycle pipeline with the following components:

### Training Pipeline

The training pipeline (`training/train_pipeline.py`) orchestrates the full model training workflow:

1. **Load Dataset** - Loads training data from CSV or generates synthetic data
2. **Data Preprocessing** - Normalizes features using standard scaling
3. **Model Training** - Trains the AnomalyDetector model
4. **Model Evaluation** - Computes accuracy, precision, recall, and F1 score
5. **Model Artifact Saving** - Saves trained model with metrics and metadata

Run the training pipeline:
```bash
python training/train_pipeline.py
```

### Model Registry

The model registry (`model_registry/`) stores versioned model artifacts:

```
model_registry/
├── model_v1/
│   ├── model.pkl       # Trained model file
│   ├── metrics.json    # Training metrics
│   └── metadata.json   # Model metadata
├── model_v2/
│   └── ...
└── latest -> model_v1/ # Symlink to latest version
```

Each model version contains:
- **model.pkl** - Serialized model artifact
- **metrics.json** - Training/evaluation metrics (accuracy, precision, recall, F1)
- **metadata.json** - Model version, creation timestamp, configuration

### Inference Service

The inference service (`inference_service/api.py`) provides a FastAPI endpoint for predictions:

- Automatically loads the latest model version from the registry
- Serves predictions via REST API
- Returns confidence scores with each prediction

Start the inference service:
```bash
python inference_service/api.py
```

API Endpoints:
- `POST /predict` - Make predictions
- `GET /model/info` - Get current model information
- `GET /health` - Health check

### Monitoring

The monitoring module (`monitoring/`) tracks model performance:

- **prediction_logger.py** - Logs predictions with timestamp, confidence score, and model version
- **metrics.py** - Calculates model performance metrics
- **drift_detection.py** - Detects data drift using statistical methods

Prediction logs include:
- Prediction result
- Confidence score
- Timestamp
- Model version

## Project Structure

```
sentinelvision_core/     # Core ML models and preprocessing
training/                # Training pipeline scripts
inference_service/       # FastAPI inference service
ml_pipeline/            # Pipeline orchestration
monitoring/              # Monitoring and logging
infra/                   # Deployment configurations
configs/                 # Configuration files
model_registry/          # Versioned model artifacts
```

## Getting Started

1. Train a model:
   ```bash
   python training/train_pipeline.py
   ```

2. Start inference service:
   ```bash
   python inference_service/api.py
   ```

3. Make predictions:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}'
   ```
