import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class RetrainingManager:
    def __init__(
        self,
        model_registry_path: str = "model_registry",
        training_data_path: str = "data/train.csv",
        retrain_threshold: float = 0.05,
        min_training_interval_hours: int = 1
    ):
        self.model_registry_path = Path(model_registry_path)
        self.training_data_path = training_data_path
        self.retrain_threshold = retrain_threshold
        self.min_training_interval_hours = min_training_interval_hours
        self.last_retrain_time: Optional[datetime] = None
        self.current_model_version: Optional[str] = None
        self._load_current_version()

    def _load_current_version(self) -> None:
        if not self.model_registry_path.exists():
            return
        
        versions = [d for d in self.model_registry_path.iterdir() 
                    if d.is_dir() and d.name.startswith("model_v")]
        if versions:
            latest = max(versions, key=lambda x: int(x.name.replace("model_v", "")))
            metadata_path = latest / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.current_model_version = metadata.get("version")

    def check_retraining_needed(self, drift_status: Dict[str, Any]) -> bool:
        if drift_status.get("drift_detected", False):
            if self._check_min_interval():
                return True
        return False

    def _check_min_interval(self) -> bool:
        if self.last_retrain_time is None:
            return True
        
        hours_since_last = (datetime.utcnow() - self.last_retrain_time).total_seconds() / 3600
        return hours_since_last >= self.min_training_interval_hours

    def trigger_retraining(self) -> Dict[str, Any]:
        print(f"[RetrainingManager] Triggering model retraining...")
        
        try:
            from training.train_pipeline import TrainingPipeline
            
            pipeline = TrainingPipeline(
                data_path=self.training_data_path,
                model_registry_path=str(self.model_registry_path),
                model_name="anomaly_detector"
            )
            
            result = pipeline.run()
            
            self.last_retrain_time = datetime.utcnow()
            self.current_model_version = result.get("version")
            
            print(f"[RetrainingManager] Retraining completed. New version: {self.current_model_version}")
            
            return {
                "status": "success",
                "new_version": self.current_model_version,
                "model_path": result.get("model_path"),
                "metrics": result.get("metrics")
            }
            
        except Exception as e:
            print(f"[RetrainingManager] Retraining failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_current_version(self) -> Optional[str]:
        return self.current_model_version

    def get_retraining_history(self) -> Dict[str, Any]:
        return {
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "current_model_version": self.current_model_version,
            "retrain_threshold": self.retrain_threshold,
            "min_interval_hours": self.min_training_interval_hours
        }


_retraining_manager_instance: Optional[RetrainingManager] = None


def get_retraining_manager() -> RetrainingManager:
    global _retraining_manager_instance
    
    if _retraining_manager_instance is None:
        _retraining_manager_instance = RetrainingManager()
    
    return _retraining_manager_instance
