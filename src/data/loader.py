from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def _require_files(raw_dir: Path, filenames: list[str]) -> None:
    missing = [name for name in filenames if not (raw_dir / name).exists()]
    if missing:
        expected = "\n".join(f"  - {raw_dir / name}" for name in missing)
        raise FileNotFoundError(
            "FraudX raw dataset files are missing.\n"
            "Download the IEEE-CIS Fraud Detection dataset and place these files "
            f"under '{raw_dir}':\n{expected}\n\n"
            "The dataset is intentionally excluded from Git because of its size "
            "and Kaggle distribution restrictions."
        )


def load_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn_name = cfg["data"].get("train_file", "train_transaction.csv")
    identity_name = cfg["data"].get("train_identity_file", "train_identity.csv")
    _require_files(raw_dir, [txn_name, identity_name])
    return pd.read_csv(raw_dir / txn_name).merge(
        pd.read_csv(raw_dir / identity_name), on="TransactionID", how="left"
    )


def load_test_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn_name = cfg["data"].get("test_file", "test_transaction.csv")
    identity_name = cfg["data"].get("test_identity_file", "test_identity.csv")
    _require_files(raw_dir, [txn_name, identity_name])
    return pd.read_csv(raw_dir / txn_name).merge(
        pd.read_csv(raw_dir / identity_name), on="TransactionID", how="left"
    )


def train_val_split(df: pd.DataFrame, cfg: dict):
    """Backward-compatible chronological train/validation split."""
    target = cfg["features"]["target_col"]
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * (1 - cfg["data"]["test_size"]))
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]
    return (
        train_df.drop(columns=[target]), val_df.drop(columns=[target]),
        train_df[target], val_df[target]
    )


def train_val_test_split(df: pd.DataFrame, cfg: dict):
    """Chronologically split into train, validation, and untouched final test."""
    target = cfg["features"]["target_col"]
    data_cfg = cfg["data"]
    test_size = float(data_cfg.get("test_size", 0.15))
    validation_size = float(data_cfg.get("validation_size", 0.15))
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to < 1")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - validation_size - test_size))
    val_end = int(n * (1 - test_size))
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    return (
        train_df.drop(columns=[target]), val_df.drop(columns=[target]), test_df.drop(columns=[target]),
        train_df[target], val_df[target], test_df[target]
    )


def save_processed(obj: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_processed(path: str | Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def processed_exists(cfg: dict) -> bool:
    proc = Path(cfg["data"]["processed_dir"])
    return all((proc / name).exists() for name in (
        "features_train.pkl", "features_val.pkl", "features_test.pkl"
    ))
