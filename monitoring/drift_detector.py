from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datetime import datetime


class DriftDetector:
    def __init__(
        self,
        threshold: float = 0.05,
        psi_threshold: float = 0.2,
        window_size: int = 100
    ):
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.window_size = window_size
        self.reference_mean: Optional[np.ndarray] = None
        self.reference_std: Optional[np.ndarray] = None
        self.reference_distribution: Optional[np.ndarray] = None
        self.baseline_set = False

    def set_baseline(self, baseline_data: np.ndarray) -> None:
        self.reference_mean = np.mean(baseline_data, axis=0)
        self.reference_std = np.std(baseline_data, axis=0)
        self.reference_distribution = self._compute_distribution(baseline_data)
        self.baseline_set = True

    def detect_prediction_drift(
        self,
        recent_predictions: List[Dict[str, Any]],
        baseline_anomaly_rate: float = 0.1
    ) -> Dict[str, Any]:
        if not recent_predictions:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "anomaly_rate_drift": 0.0,
                "confidence_drift": 0.0
            }
        
        recent_anomaly_rate = sum(
            1 for p in recent_predictions if p.get("prediction", False)
        ) / len(recent_predictions)
        
        anomaly_rate_drift = abs(recent_anomaly_rate - baseline_anomaly_rate)
        
        scores = [p.get("confidence_score", 0.0) for p in recent_predictions]
        if scores:
            recent_mean_confidence = np.mean(scores)
            confidence_drift = abs(recent_mean_confidence - 0.5)
        else:
            confidence_drift = 0.0
        
        drift_score = (anomaly_rate_drift + confidence_drift) / 2
        drift_detected = drift_score > self.threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "anomaly_rate_drift": float(anomaly_rate_drift),
            "confidence_drift": float(confidence_drift),
            "recent_anomaly_rate": float(recent_anomaly_rate),
            "baseline_anomaly_rate": baseline_anomaly_rate,
            "timestamp": datetime.utcnow().isoformat()
        }

    def detect_feature_drift(
        self,
        current_data: np.ndarray
    ) -> Dict[str, Any]:
        if not self.baseline_set:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "message": "No baseline set"
            }
        
        current_mean = np.mean(current_data, axis=0)
        drift_scores = np.abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)
        avg_drift = float(np.mean(drift_scores))
        
        max_feature_drift = float(np.max(drift_scores))
        drift_detected = avg_drift > self.threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": avg_drift,
            "max_feature_drift": max_feature_drift,
            "timestamp": datetime.utcnow().isoformat()
        }

    def calculate_psi(
        self,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        bins: int = 10
    ) -> float:
        if len(reference_dist) == 0 or len(current_dist) == 0:
            return 0.0
        
        reference_hist, _ = np.histogram(reference_dist, bins=bins)
        current_hist, _ = np.histogram(current_dist, bins=bins)
        
        reference_prob = reference_hist / (reference_hist.sum() + 1e-8)
        current_prob = current_hist / (current_hist.sum() + 1e-8)
        
        psi_values = (current_prob - reference_prob) * np.log(
            (current_prob + 1e-8) / (reference_prob + 1e-8)
        )
        
        return float(np.sum(psi_values))

    def get_drift_status(self) -> Dict[str, Any]:
        return {
            "baseline_set": self.baseline_set,
            "threshold": self.threshold,
            "psi_threshold": self.psi_threshold,
            "window_size": self.window_size
        }

    def _compute_distribution(self, data: np.ndarray) -> np.ndarray:
        return np.histogram(data, bins=10)[0]


def population_stability_index(
    reference_dist: np.ndarray,
    current_dist: np.ndarray,
    bins: int = 10
) -> float:
    if len(reference_dist) == 0 or len(current_dist) == 0:
        return 0.0
    
    reference_hist, _ = np.histogram(reference_dist, bins=bins)
    current_hist, _ = np.histogram(current_dist, bins=bins)
    
    reference_prob = reference_hist / (reference_hist.sum() + 1e-8)
    current_prob = current_hist / (current_hist.sum() + 1e-8)
    
    psi_values = (current_prob - reference_prob) * np.log(
        (current_prob + 1e-8) / (reference_prob + 1e-8)
    )
    
    return float(np.sum(psi_values))
