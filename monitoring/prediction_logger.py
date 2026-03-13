from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path


class PredictionLogger:
    def __init__(self, log_path: str = "logs/predictions.log"):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        input_data: List[float],
        prediction: bool,
        score: float,
        model_version: str = "1.0.0",
        input_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_metadata": input_metadata or {},
            "input_data": input_data,
            "prediction": prediction,
            "confidence_score": score,
            "model_version": model_version
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_recent_predictions(self, n: int = 100) -> List[Dict[str, Any]]:
        predictions = []
        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    predictions.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass
        return predictions

    def get_all_predictions(self) -> List[Dict[str, Any]]:
        predictions = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    predictions.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass
        return predictions

    def clear_logs(self) -> None:
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def get_prediction_count(self) -> int:
        try:
            with open(self.log_path, "r") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
