from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.features import apply_smote
from src.data.loader import load_processed
from src.evaluate import compute_metrics, find_cost_optimal_threshold
from src.models.ensemble import FraudEnsemble


STRATEGIES = ("both", "weights", "smote", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fraud imbalance strategies on the fixed temporal validation set")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/imbalance_ablation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    proc = Path(base_cfg["data"]["processed_dir"])
    X_train, y_train = load_processed(proc / "features_train.pkl")
    X_val, y_val = load_processed(proc / "features_val.pkl")
    metadata = load_processed(proc / "feature_metadata.pkl")

    rows = []
    for strategy in STRATEGIES:
        cfg = copy.deepcopy(base_cfg)
        cfg["imbalance"] = {"strategy": strategy}
        cfg["mlflow"] = {**cfg.get("mlflow", {}), "enabled": False}

        X_fit, y_fit = X_train.copy(), y_train.copy()
        if strategy in {"smote", "both"}:
            X_fit, y_fit = apply_smote(X_fit, y_fit, cfg)

        model = FraudEnsemble(cfg)
        model.category_mappings = metadata["category_mappings"]
        model.selected_v_columns = metadata["selected_v_columns"]
        model.fit(X_fit, y_fit, X_val, y_val)

        val_proba = model.predict_proba(X_val)
        threshold, cost = find_cost_optimal_threshold(
            y_val.values,
            val_proba,
            false_positive_cost=float(cfg["evaluation"]["costs"]["false_positive"]),
            false_negative_cost=float(cfg["evaluation"]["costs"]["false_negative"]),
        )
        metrics = compute_metrics(y_val.values, val_proba, threshold=threshold)
        metrics.update({"strategy": strategy, "threshold_cost": float(cost), "train_rows": int(len(X_fit))})
        rows.append(metrics)
        print(
            f"{strategy:8s} | PR-AUC {metrics['auc_pr']:.4f} | "
            f"ROC-AUC {metrics['auc_roc']:.4f} | P {metrics['precision']:.4f} | "
            f"R {metrics['recall']:.4f} | F1 {metrics['f1']:.4f} | "
            f"threshold {threshold:.4f} | cost {cost:.0f}"
        )

    result = {
        "metric_selection": "temporal validation only; final test is intentionally not used",
        "strategies": rows,
        "recommended_strategy": max(rows, key=lambda row: (row["auc_pr"], -row["threshold_cost"]))["strategy"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {output}")
    print("Do not promote a strategy to production based on this validation result alone; retrain the selected strategy and run the frozen final test evaluation once.")


if __name__ == "__main__":
    main()
