from typing import Dict, Any, List, Optional
from datetime import datetime

from monitoring.prediction_logger import PredictionLogger
from monitoring.metrics_aggregator import MetricsAggregator
from monitoring.drift_detector import DriftDetector


class MonitoringService:
    def __init__(
        self,
        log_path: str = "logs/predictions.log",
        baseline_anomaly_rate: float = 0.1,
        drift_threshold: float = 0.05
    ):
        self.prediction_logger = PredictionLogger(log_path=log_path)
        self.metrics_aggregator = MetricsAggregator()
        self.drift_detector = DriftDetector(
            threshold=drift_threshold,
            window_size=100
        )
        self.baseline_anomaly_rate = baseline_anomaly_rate
        
        self._load_predictions()

    def _load_predictions(self) -> None:
        predictions = self.prediction_logger.get_all_predictions()
        self.metrics_aggregator.add_predictions(predictions)

    def record_prediction_event(
        self,
        input_data: List[float],
        prediction: bool,
        confidence_score: float,
        model_version: str = "1.0.0",
        input_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.prediction_logger.log_prediction(
            input_data=input_data,
            prediction=prediction,
            score=confidence_score,
            model_version=model_version,
            input_metadata=input_metadata
        )
        
        self.metrics_aggregator.add_predictions([{
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": input_data,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "model_version": model_version
        }])

    def get_metrics_summary(
        self,
        window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        distribution = self.metrics_aggregator.calculate_prediction_distribution(window_minutes)
        confidence_stats = self.metrics_aggregator.calculate_confidence_statistics(window_minutes)
        
        return {
            "distribution": distribution,
            "confidence_stats": confidence_stats,
            "anomaly_rate": distribution.get("anomaly", 0.0),
            "total_predictions": distribution.get("total_predictions", 0)
        }

    def get_drift_status(self) -> Dict[str, Any]:
        recent_predictions = self.prediction_logger.get_recent_predictions(
            n=self.drift_detector.window_size
        )
        
        prediction_drift = self.drift_detector.detect_prediction_drift(
            recent_predictions=recent_predictions,
            baseline_anomaly_rate=self.baseline_anomaly_rate
        )
        
        return {
            "prediction_drift": prediction_drift,
            "drift_detector_status": self.drift_detector.get_drift_status()
        }

    def get_time_series_metrics(
        self,
        interval_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        return self.metrics_aggregator.calculate_metrics_over_time(interval_minutes)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "metrics": self.get_metrics_summary(),
            "drift": self.get_drift_status(),
            "time_series": self.get_time_series_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }

    def clear_metrics(self) -> None:
        self.prediction_logger.clear_logs()
        self.metrics_aggregator.clear()


_monitoring_service_instance: Optional[MonitoringService] = None


def get_monitoring_service(
    log_path: str = "logs/predictions.log",
    baseline_anomaly_rate: float = 0.1,
    drift_threshold: float = 0.05
) -> MonitoringService:
    global _monitoring_service_instance
    
    if _monitoring_service_instance is None:
        _monitoring_service_instance = MonitoringService(
            log_path=log_path,
            baseline_anomaly_rate=baseline_anomaly_rate,
            drift_threshold=drift_threshold
        )
    
    return _monitoring_service_instance


def record_prediction_event(
    input_data: List[float],
    prediction: bool,
    confidence_score: float,
    model_version: str = "1.0.0",
    input_metadata: Optional[Dict[str, Any]] = None
) -> None:
    service = get_monitoring_service()
    service.record_prediction_event(
        input_data=input_data,
        prediction=prediction,
        confidence_score=confidence_score,
        model_version=model_version,
        input_metadata=input_metadata
    )
