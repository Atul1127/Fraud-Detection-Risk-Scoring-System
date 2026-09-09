from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluate import compute_metrics, find_cost_optimal_threshold, find_best_threshold
from src.models.ensemble import FraudEnsemble


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model = FraudEnsemble(cfg)
        self.category_mappings: dict[str, list] = {}

    def set_category_mappings(self, mappings: dict[str, list]) -> None:
        self.category_mappings = mappings
        self.model.category_mappings = mappings

    def _select_threshold(self, y_true, proba):
        evaluation = self.cfg.get("evaluation", {})
        strategy = evaluation.get("threshold_strategy", "f1").lower()
        if strategy == "cost":
            costs = evaluation.get("costs", {})
            return find_cost_optimal_threshold(
                y_true,
                proba,
                false_positive_cost=float(costs.get("false_positive", 1.0)),
                false_negative_cost=float(costs.get("false_negative", 10.0)),
            )
        return find_best_threshold(y_true, proba)

    def run(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None) -> dict:
        self.model.fit(X_train, y_train, X_val, y_val)

        print("\nEvaluating on validation set (threshold tuning only)...")
        val_predictions = self.model.predict_individual(X_val)
        comparison = {}

        for name, model_proba in val_predictions.items():
            threshold, objective = self._select_threshold(y_val.values, model_proba)
            comparison[name] = compute_metrics(y_val.values, model_proba, threshold=threshold)
            comparison[name]["threshold_objective"] = float(objective)
            print(
                f"{name:10s} | AUC-ROC: {comparison[name]['auc_roc']:.4f} | "
                f"AUC-PR: {comparison[name]['auc_pr']:.4f} | F1: {comparison[name]['f1']:.4f}"
            )

        proba = val_predictions["Ensemble"]
        best_threshold, threshold_objective = self._select_threshold(y_val.values, proba)
        val_metrics = compute_metrics(y_val.values, proba, threshold=best_threshold)

        print("\nValidation / threshold tuning:")
        print(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {val_metrics['auc_pr']:.4f}")
        print(f"  Selected threshold: {best_threshold:.3f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")

        self.cfg["ensemble"]["default_threshold"] = float(best_threshold)

        report = {
            **val_metrics,
            "best_threshold": float(best_threshold),
            "threshold_strategy": self.cfg.get("evaluation", {}).get("threshold_strategy", "f1"),
            "threshold_objective": float(threshold_objective),
            "top_features": sorted(
                dict(zip(self.model.feature_names, self.model.xgb_model.feature_importances_.tolist())).items(),
                key=lambda x: x[1], reverse=True,
            )[:25],
            "model_comparison": comparison,
        }

        if X_test is not None and y_test is not None:
            print("\nFinal evaluation on untouched chronological test set...")
            test_predictions = self.model.predict_individual(X_test)
            test_proba = test_predictions["Ensemble"]
            test_metrics = compute_metrics(y_test.values, test_proba, threshold=best_threshold)
            report["test_metrics"] = test_metrics
            print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"  AUC-PR: {test_metrics['auc_pr']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  F1: {test_metrics['f1']:.4f}")

        return report

    def save(self, checkpoint_dir: str | Path, report: dict) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.model.save(checkpoint_dir)
        with open(checkpoint_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Training report saved to {checkpoint_dir / 'training_report.json'}")
