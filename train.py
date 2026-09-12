from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-preprocess", action="store_true", help="Recompute features even if cache exists")
    parser.add_argument("--skip-smote", action="store_true", help="Skip SMOTE oversampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.data.loader import load_raw, train_val_test_split, save_processed, load_processed, processed_exists
    from src.data.features import (
        build_features,
        apply_smote,
        fit_category_mappings,
        encode_categoricals,
        select_v_features,
    )
    from src.train import Trainer

    mlflow_enabled = cfg.get("mlflow", {}).get("enabled", False)
    if mlflow_enabled:
        from src.mlflow_tracker import log_training_run, start_run
        mlflow_context = start_run(cfg)
    else:
        mlflow_context = nullcontext()
        log_training_run = None

    with mlflow_context:
        proc = Path(cfg["data"]["processed_dir"])
        target = cfg["features"]["target_col"]
        category_mappings = {}
        selected_v_columns = []

        if not args.force_preprocess and processed_exists(cfg):
            print("Loading cached train/validation/test features...")
            X_train, y_train = load_processed(proc / "features_train.pkl")
            X_val, y_val = load_processed(proc / "features_val.pkl")
            X_test, y_test = load_processed(proc / "features_test.pkl")
            metadata = load_processed(proc / "feature_metadata.pkl")
            category_mappings = metadata["category_mappings"]
            selected_v_columns = metadata["selected_v_columns"]
            print(f"  Loaded train-fitted metadata for {len(category_mappings)} categorical columns.")
        else:
            print("Loading raw data...")
            df = load_raw(cfg)
            print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
            print(f"  Fraud rate: {df[target].mean():.4f}")

            print("Engineering causal features...")
            # History-dependent features are built on the full chronological stream so
            # validation/test rows can use legitimate history from earlier rows.
            # Categorical encoding is deliberately deferred until after the temporal
            # split so category vocabularies are fitted on training data only.
            df_feat = build_features(df, cfg, select_v=False, encode_categories=False)

            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_feat, cfg)
            print(
                f"  Train: {len(X_train):,} | Val: {len(X_val):,} | "
                f"Test: {len(X_test):,}"
            )

            v_cols = [c for c in X_train.columns if c.startswith("V")]
            selected_v_columns = X_train[v_cols].isnull().mean().nsmallest(50).index.tolist() if v_cols else []
            X_train = select_v_features(X_train, selected_columns=selected_v_columns)
            X_val = select_v_features(X_val, selected_columns=selected_v_columns)
            X_test = select_v_features(X_test, selected_columns=selected_v_columns)

            print("Fitting categorical mappings on training data only...")
            category_mappings = fit_category_mappings(X_train)
            X_train = encode_categoricals(X_train, category_mappings)
            X_val = encode_categoricals(X_val, category_mappings)
            X_test = encode_categoricals(X_test, category_mappings)
            print(f"  Saved mappings for {len(category_mappings)} categorical columns.")

            save_processed((X_train, y_train), proc / "features_train.pkl")
            save_processed((X_val, y_val), proc / "features_val.pkl")
            save_processed((X_test, y_test), proc / "features_test.pkl")
            save_processed(
                {"category_mappings": category_mappings, "selected_v_columns": selected_v_columns},
                proc / "feature_metadata.pkl",
            )
            print("  Cached processed features and train-fitted metadata.")

        strategy = cfg.get("imbalance", {}).get("strategy")
        if strategy is None:
            strategy = "both" if cfg.get("smote", {}).get("enabled", False) else "weights"
        if args.skip_smote:
            strategy = "weights" if strategy in {"both", "smote"} else strategy
        cfg["imbalance"] = {"strategy": strategy}
        print(f"\nImbalance strategy: {strategy}")

        if strategy in {"smote", "both"}:
            print("Applying SMOTE to training data only...")
            X_train, y_train = apply_smote(X_train, y_train, cfg)
        else:
            print("Skipping SMOTE; training uses original class distribution.")

        print("\nTraining ensemble...")
        trainer = Trainer(cfg)
        trainer.set_category_mappings(category_mappings)
        trainer.model.selected_v_columns = selected_v_columns
        report = trainer.run(X_train, y_train, X_val, y_val, X_test, y_test)

        ckpt_dir = Path(cfg["data"].get("model_dir", "models")) / "checkpoints"
        trainer.save(ckpt_dir, report)

        cfg["ensemble"]["default_threshold"] = report["best_threshold"]
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"\nValidation-selected threshold ({report['best_threshold']:.3f}) written to {args.config}")

        if mlflow_enabled and log_training_run is not None:
            log_training_run(cfg, report, ckpt_dir)
            print("MLflow run logged successfully.")

        print("Done.")


if __name__ == "__main__":
    main()
