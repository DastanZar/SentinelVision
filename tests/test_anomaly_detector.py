import numpy as np
import pytest
import tempfile
import os
from sentinelvision_core.models.anomaly_detector import AnomalyDetector


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(200, 5)


def test_fit_predict(sample_data):
    detector = AnomalyDetector(model_type="iforest", contamination=0.05)
    detector.fit(sample_data)
    result = detector.predict(sample_data)
    assert "predictions" in result
    assert "scores" in result
    assert "threshold" in result
    assert len(result["predictions"]) == len(sample_data)
    assert len(result["scores"]) == len(sample_data)
    assert all(p in [0, 1] for p in result["predictions"])


def test_predict_before_fit_raises():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(np.random.randn(10, 5))


def test_save_and_load(sample_data):
    detector = AnomalyDetector(model_type="iforest", contamination=0.05)
    detector.fit(sample_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        detector.save(path)
        assert os.path.exists(path)
        loaded = AnomalyDetector.load(path)
        assert loaded.is_trained
        result = loaded.predict(sample_data)
        assert len(result["predictions"]) == len(sample_data)


def test_save_before_fit_raises():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.save("/tmp/should_fail.pkl")


def test_ecod_model(sample_data):
    detector = AnomalyDetector(model_type="ecod", contamination=0.05)
    detector.fit(sample_data)
    result = detector.predict(sample_data)
    assert len(result["predictions"]) == len(sample_data)


def test_invalid_model_type_raises():
    with pytest.raises(ValueError):
        AnomalyDetector(model_type="nonexistent")


def test_anomaly_ratio_is_close_to_contamination(sample_data):
    contamination = 0.05
    detector = AnomalyDetector(model_type="iforest", contamination=contamination)
    detector.fit(sample_data)
    result = detector.predict(sample_data)
    anomaly_ratio = sum(result["predictions"]) / len(result["predictions"])
    assert abs(anomaly_ratio - contamination) < 0.03
