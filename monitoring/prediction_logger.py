from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class PredictionLogger:
    def __init__(self, log_path: str = "logs/predictions.log"):
        self.log_path = log_path

    def log_prediction(
        self,
        input_data: List[float],
        prediction: bool,
        score: float,
        model_version: str = "1.0.0"
    ) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": input_data,
            "prediction": prediction,
            "score": score,
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
