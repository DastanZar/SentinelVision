from typing import List, Dict, Any, Tuple
import numpy as np


class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_mean: np.ndarray = None
        self.reference_std: np.ndarray = None

    def fit(self, reference_data: np.ndarray) -> None:
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)

    def detect_drift(self, current_data: np.ndarray) -> Tuple[bool, float]:
        if self.reference_mean is None or self.reference_std is None:
            raise ValueError("DriftDetector must be fitted before detecting drift")

        current_mean = np.mean(current_data, axis=0)
        drift_score = np.abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)
        avg_drift = np.mean(drift_score)
        
        is_drift = avg_drift > self.threshold
        return is_drift, float(avg_drift)


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
