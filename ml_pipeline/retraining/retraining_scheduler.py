import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from ml_pipeline.retraining.retraining_manager import RetrainingManager


class RetrainingScheduler:
    def __init__(
        self,
        retraining_manager: RetrainingManager,
        check_interval_seconds: int = 300,
        drift_threshold: float = 0.05,
        anomaly_rate_threshold: float = 0.3,
        enabled: bool = True
    ):
        self.retraining_manager = retraining_manager
        self.check_interval_seconds = check_interval_seconds
        self.drift_threshold = drift_threshold
        self.anomaly_rate_threshold = anomaly_rate_threshold
        self.enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list = []
        self.last_check_time: Optional[datetime] = None
        self.last_retrain_triggered: Optional[datetime] = None

    def start(self) -> None:
        if not self.enabled:
            print("[RetrainingScheduler] Scheduler is disabled")
            return
        
        if self._thread is not None and self._thread.is_alive():
            print("[RetrainingScheduler] Scheduler already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        print(f"[RetrainingScheduler] Started with check interval: {self.check_interval_seconds}s")

    def stop(self) -> None:
        if self._thread is None:
            return
        
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        print("[RetrainingScheduler] Stopped")

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def _run_scheduler(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_and_retrain()
            except Exception as e:
                print(f"[RetrainingScheduler] Error in scheduler loop: {str(e)}")
            
            self._stop_event.wait(self.check_interval_seconds)

    def _check_and_retrain(self) -> None:
        from monitoring.monitoring_service import get_monitoring_service
        
        self.last_check_time = datetime.utcnow()
        print("[RetrainingScheduler] Checking monitoring metrics...")
        
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        drift_status = monitoring_service.get_drift_status()
        
        drift_score = drift_status.get("prediction_drift", {}).get("drift_score", 0.0)
        anomaly_rate = metrics.get("anomaly_rate", 0.0)
        
        print(f"[RetrainingScheduler] Drift score: {drift_score:.4f}, Anomaly rate: {anomaly_rate:.4f}")
        
        should_retrain = False
        trigger_reason = None
        
        if drift_score > self.drift_threshold:
            should_retrain = True
            trigger_reason = f"drift_exceeded ({drift_score:.4f} > {self.drift_threshold})"
        
        if anomaly_rate > self.anomaly_rate_threshold:
            should_retrain = True
            trigger_reason = f"anomaly_rate_spike ({anomaly_rate:.4f} > {self.anomaly_rate_threshold})"
        
        if should_retrain:
            print(f"[RetrainingScheduler] Triggering retraining: {trigger_reason}")
            result = self.retraining_manager.trigger_retraining()
            self.last_retrain_triggered = datetime.utcnow()
            
            for callback in self._callbacks:
                try:
                    callback({
                        "trigger_reason": trigger_reason,
                        "result": result,
                        "timestamp": self.last_retrain_triggered.isoformat()
                    })
                except Exception as e:
                    print(f"[RetrainingScheduler] Callback error: {str(e)}")
        else:
            print("[RetrainingScheduler] No retraining needed")

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self._thread is not None and self._thread.is_alive(),
            "check_interval_seconds": self.check_interval_seconds,
            "drift_threshold": self.drift_threshold,
            "anomaly_rate_threshold": self.anomaly_rate_threshold,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_retrain_triggered": self.last_retrain_triggered.isoformat() if self.last_retrain_triggered else None
        }

    def force_retrain(self) -> Dict[str, Any]:
        return self.retraining_manager.trigger_retraining()


_retraining_scheduler_instance: Optional[RetrainingScheduler] = None


def get_retraining_scheduler() -> RetrainingScheduler:
    global _retraining_scheduler_instance
    
    if _retraining_scheduler_instance is None:
        retraining_manager = get_retraining_manager()
        _retraining_scheduler_instance = RetrainingScheduler(
            retraining_manager=retraining_manager,
            check_interval_seconds=300,
            drift_threshold=0.05,
            anomaly_rate_threshold=0.3,
            enabled=True
        )
    
    return _retraining_scheduler_instance
