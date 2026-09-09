import numpy as np

from src.evaluate import compute_metrics, find_best_threshold, find_cost_optimal_threshold


def test_find_best_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    threshold, score = find_best_threshold(y_true, probabilities)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= score <= 1.0


def test_cost_threshold_returns_valid_threshold_and_cost():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.40, 0.45, 0.70, 0.95])
    threshold, cost = find_cost_optimal_threshold(
        y_true, probabilities, false_positive_cost=1, false_negative_cost=10
    )
    assert 0.0 <= threshold <= 1.0
    assert cost >= 0.0


def test_compute_metrics_contains_core_fraud_metrics():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    metrics = compute_metrics(y_true, probabilities, threshold=0.5)
    for key in ["auc_roc", "auc_pr", "precision", "recall", "f1"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
