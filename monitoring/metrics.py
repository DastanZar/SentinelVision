from typing import List, Dict, Any
import numpy as np


def calculate_accuracy(predictions: List[bool], ground_truth: List[bool]) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, ground_truth))
    return correct / len(predictions)


def calculate_precision(predictions: List[bool], ground_truth: List[bool]) -> float:
    true_positives = sum(p and t for p, t in zip(predictions, ground_truth))
    predicted_positives = sum(predictions)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def calculate_recall(predictions: List[bool], ground_truth: List[bool]) -> float:
    true_positives = sum(p and t for p, t in zip(predictions, ground_truth))
    actual_positives = sum(ground_truth)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def calculate_f1_score(predictions: List[bool], ground_truth: List[bool]) -> float:
    precision = calculate_precision(predictions, ground_truth)
    recall = calculate_recall(predictions, ground_truth)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mean(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return np.mean(scores)


def calculate_std(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return np.std(scores)


def calculate_percentile(scores: List[float], percentile: float) -> float:
    if not scores:
        return 0.0
    return np.percentile(scores, percentile)
