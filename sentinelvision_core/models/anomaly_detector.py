import joblib
import numpy as np
from typing import Dict, Any, Optional
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD


class AnomalyDetector:
    SUPPORTED_MODELS = {"iforest": IForest, "ecod": ECOD}

    def __init__(self, model_type: str = "iforest", contamination: float = 0.05, random_state: int = 42, **kwargs):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {list(self.SUPPORTED_MODELS.keys())}, got '{model_type}'")
        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs
        self._detector = None
        self.is_trained = False

    def _build_detector(self):
        cls = self.SUPPORTED_MODELS[self.model_type]
        if self.model_type == "iforest":
            return cls(contamination=self.contamination, random_state=self.random_state, **self.kwargs)
        return cls(contamination=self.contamination, **self.kwargs)

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        self._detector = self._build_detector()
        self._detector.fit(X)
        self.is_trained = True
        return self

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.fit(X)

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained or self._detector is None:
            raise RuntimeError("Model is not trained. Call fit() before predict().")
        raw_labels = self._detector.predict(X)
        scores = self._detector.decision_function(X)
        return {
            "predictions": raw_labels.tolist(),
            "scores": scores.tolist(),
            "threshold": float(self._detector.threshold_),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self._detector is None:
            raise RuntimeError("Model is not trained. Call fit() before predict_proba().")
        return self._detector.predict_proba(X)

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        payload = {
            "model_type": self.model_type,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "kwargs": self.kwargs,
            "detector": self._detector,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        payload = joblib.load(path)
        instance = cls(model_type=payload["model_type"], contamination=payload["contamination"], random_state=payload["random_state"], **payload["kwargs"])
        instance._detector = payload["detector"]
        instance.is_trained = True
        return instance
