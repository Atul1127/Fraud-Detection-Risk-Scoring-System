from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Return threshold that maximises F1 on the provided tuning data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
) -> tuple[float, float]:
    """Minimise expected classification cost on tuning data.

    Costs are configurable because real fraud operations can assign very
    different business costs to missed fraud versus unnecessary declines/reviews.
    The returned threshold must be frozen before evaluating an untouched test set.
    """
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("classification costs must be non-negative")
    if false_positive_cost == 0 and false_negative_cost == 0:
        raise ValueError("at least one classification cost must be positive")

    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    candidates = np.unique(np.concatenate(([0.0, 1.0], thresholds)))
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in candidates:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = false_positive_cost * fp + false_negative_cost * fn
        if total_cost < best_cost:
            best_cost = float(total_cost)
            best_threshold = float(threshold)

    return best_threshold, best_cost


def get_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    return roc_curve(y_true, y_proba)


def get_pr_curve(y_true: np.ndarray, y_proba: np.ndarray):
    return precision_recall_curve(y_true, y_proba)


def get_confusion_matrix(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> np.ndarray:
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)
