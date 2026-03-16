import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sentinelvision_core import AnomalyDetector

EXPERIMENT_NAME = "sentinelvision-anomaly-training"


class TrainingPipeline:
    def __init__(self, data_path: str, model_registry_path: str = "model_registry", model_name: str = "anomaly_detector", model_type: str = "iforest", contamination: float = 0.05, label_column: Optional[str] = None):
        self.data_path = data_path
        self.model_registry_path = Path(model_registry_path)
        self.model_name = model_name
        self.model_type = model_type
        self.contamination = contamination
        self.label_column = label_column
        self.model: Optional[AnomalyDetector] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def load_dataset(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print("[1/5] Loading dataset...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at '{self.data_path}'. Provide a real CSV file - synthetic fallback is disabled.")
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")
        y = None
        if self.label_column and self.label_column in df.columns:
            y = df[self.label_column].values.astype(int)
            X = df.drop(columns=[self.label_column]).values
            print(f"  Label column '{self.label_column}' found. Supervised evaluation enabled.")
        else:
            X = df.values
            print("  No label column. Unsupervised evaluation only.")
        return X.astype(float), y

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        print("[2/5] Preprocessing data...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print(f"  Scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
        return X_scaled

    def train_model(self, X_train: np.ndarray) -> AnomalyDetector:
        print("[3/5] Training model...")
        self.model = AnomalyDetector(model_type=self.model_type, contamination=self.contamination)
        self.model.fit(X_train)
        print(f"  Trained {self.model_type} on {len(X_train)} samples.")
        return self.model

    def evaluate_model(self, X_val: np.ndarray, y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        print("[4/5] Evaluating model...")
        result = self.model.predict(X_val)
        preds = np.array(result["predictions"])
        scores = np.array(result["scores"])
        if y_val is not None:
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            self.metrics = {
                "precision": float(precision_score(y_val, preds, zero_division=0)),
                "recall": float(recall_score(y_val, preds, zero_division=0)),
                "f1_score": float(f1_score(y_val, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_val, scores)),
                "true_positives": int(tp), "false_positives": int(fp),
                "true_negatives": int(tn), "false_negatives": int(fn),
                "samples_evaluated": len(X_val), "evaluation_mode": "supervised",
            }
        else:
            self.metrics = {
                "anomaly_ratio": float(preds.sum() / len(preds)),
                "score_mean": float(scores.mean()), "score_std": float(scores.std()),
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "samples_evaluated": len(X_val), "evaluation_mode": "unsupervised",
            }
        self.metrics["timestamp"] = datetime.utcnow().isoformat()
        return self.metrics

    def save_model_artifact(self) -> str:
        print("[5/5] Saving model artifact...")
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        versions = [int(v.replace("model_v", "")) for v in os.listdir(self.model_registry_path) if v.startswith("model_v")]
        new_version = max(versions) + 1 if versions else 1
        version_dir = self.model_registry_path / f"model_v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(version_dir / "model.pkl"))
        with open(version_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.metadata = {"model_name": self.model_name, "model_type": self.model_type, "version": f"v{new_version}", "version_number": new_version, "contamination": self.contamination, "created_at": datetime.utcnow().isoformat(), "metrics": self.metrics}
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.model_registry_path / "latest.txt", "w") as f:
            f.write(f"model_v{new_version}")
        print(f"  Saved to: {version_dir}")
        return str(version_dir)

    def run(self) -> Dict[str, Any]:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{self.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tags({"project": "SentinelVision", "model_type": self.model_type, "dataset": self.data_path})
            mlflow.log_params({"model_type": self.model_type, "contamination": self.contamination, "data_path": self.data_path})
            X, y = self.load_dataset()
            X_scaled = self.preprocess(X)
            if y is not None:
                X_train, X_val, _, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
                y_val = None
            self.train_model(X_train)
            self.evaluate_model(X_val, y_val)
            mlflow.log_metrics({k: v for k, v in self.metrics.items() if isinstance(v, (int, float))})
            model_path = self.save_model_artifact()
            mlflow.log_artifact(model_path)
            return {"status": "success", "model_path": model_path, "version": self.metadata["version"], "metrics": self.metrics}


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    label_col = sys.argv[2] if len(sys.argv) > 2 else None
    pipeline = TrainingPipeline(data_path=data_path, model_registry_path="model_registry", model_name="anomaly_detector", model_type="iforest", contamination=0.05, label_column=label_col)
    result = pipeline.run()
    print(f"\nStatus: {result['status']}\nVersion: {result['version']}\nMetrics: {json.dumps(result['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
