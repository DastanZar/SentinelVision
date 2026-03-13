from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class MetricsAggregator:
    def __init__(self):
        self.predictions: List[Dict[str, Any]] = []

    def add_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        self.predictions.extend(predictions)

    def calculate_prediction_distribution(self, window_minutes: Optional[int] = None) -> Dict[str, float]:
        predictions = self._filter_by_window(window_minutes)
        
        if not predictions:
            return {"normal": 0.0, "anomaly": 0.0}
        
        normal_count = sum(1 for p in predictions if not p.get("prediction", False))
        anomaly_count = sum(1 for p in predictions if p.get("prediction", False))
        total = len(predictions)
        
        return {
            "normal": normal_count / total if total > 0 else 0.0,
            "anomaly": anomaly_count / total if total > 0 else 0.0,
            "total_predictions": total
        }

    def calculate_anomaly_rate(self, window_minutes: Optional[int] = None) -> float:
        distribution = self.calculate_prediction_distribution(window_minutes)
        return distribution.get("anomaly", 0.0)

    def calculate_confidence_statistics(self, window_minutes: Optional[int] = None) -> Dict[str, float]:
        predictions = self._filter_by_window(window_minutes)
        
        if not predictions:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "p95": 0.0
            }
        
        scores = [p.get("confidence_score", 0.0) for p in predictions]
        scores_array = np.array(scores)
        
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "p25": float(np.percentile(scores_array, 25)),
            "p75": float(np.percentile(scores_array, 75)),
            "p95": float(np.percentile(scores_array, 95))
        }

    def calculate_metrics_over_time(
        self,
        interval_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.predictions:
            return []
        
        time_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for pred in self.predictions:
            timestamp = pred.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                bucket_key = dt.replace(
                    minute=(dt.minute // interval_minutes) * interval_minutes,
                    second=0,
                    microsecond=0
                ).isoformat()
                time_buckets[bucket_key].append(pred)
            except (ValueError, AttributeError):
                continue
        
        results = []
        for bucket_key in sorted(time_buckets.keys()):
            bucket_preds = time_buckets[bucket_key]
            scores = [p.get("confidence_score", 0.0) for p in bucket_preds]
            anomalies = sum(1 for p in bucket_preds if p.get("prediction", False))
            
            results.append({
                "timestamp": bucket_key,
                "prediction_count": len(bucket_preds),
                "anomaly_count": anomalies,
                "anomaly_rate": anomalies / len(bucket_preds) if bucket_preds else 0.0,
                "mean_confidence": float(np.mean(scores)) if scores else 0.0
            })
        
        return results

    def get_summary_metrics(self) -> Dict[str, Any]:
        if not self.predictions:
            return {
                "total_predictions": 0,
                "anomaly_rate": 0.0,
                "distribution": {"normal": 0.0, "anomaly": 0.0},
                "confidence_stats": {}
            }
        
        distribution = self.calculate_prediction_distribution()
        confidence_stats = self.calculate_confidence_statistics()
        
        return {
            "total_predictions": len(self.predictions),
            "anomaly_rate": distribution.get("anomaly", 0.0),
            "distribution": distribution,
            "confidence_stats": confidence_stats,
            "time_range": {
                "earliest": self.predictions[0].get("timestamp") if self.predictions else None,
                "latest": self.predictions[-1].get("timestamp") if self.predictions else None
            }
        }

    def _filter_by_window(self, window_minutes: Optional[int]) -> List[Dict[str, Any]]:
        if window_minutes is None:
            return self.predictions
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        filtered = []
        
        for pred in self.predictions:
            timestamp = pred.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if dt.replace(tzinfo=None) >= cutoff_time:
                    filtered.append(pred)
            except (ValueError, AttributeError):
                continue
        
        return filtered

    def clear(self) -> None:
        self.predictions = []
