from types import SimpleNamespace

import numpy as np
import pandas as pd

from api.feature_store import OnlineFeatureStore
from src.data.features import build_features, fit_category_mappings


CFG = {
    "features": {
        "velocity": {"enabled": True, "windows": [3600, 86400, 604800]},
        "frequency": {
            "enabled": True,
            "columns": [
                "card1",
                "card2",
                "card3",
                "card5",
                "addr1",
                "addr2",
                "P_emaildomain",
                "R_emaildomain",
            ],
        },
    }
}


class FakeMongo:
    def __init__(self, history):
        self.history = history

    def get_history(self, transaction, frequency_columns, max_window):
        assert "card1" in frequency_columns
        return [row.copy() for row in self.history if row["TransactionDT"] < transaction["TransactionDT"]]


def _rows():
    return [
        {
            "TransactionID": 1,
            "TransactionDT": 100,
            "TransactionAmt": 20.0,
            "card1": 111,
            "card2": 10,
            "card3": 20,
            "card5": 30,
            "addr1": 5,
            "addr2": 6,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "M1": "T",
        },
        {
            "TransactionID": 2,
            "TransactionDT": 500,
            "TransactionAmt": 35.0,
            "card1": 111,
            "card2": 10,
            "card3": 20,
            "card5": 30,
            "addr1": 5,
            "addr2": 6,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "yahoo.com",
            "M1": "F",
        },
        {
            "TransactionID": 3,
            "TransactionDT": 2000,
            "TransactionAmt": 80.0,
            "card1": 111,
            "card2": 10,
            "card3": 20,
            "card5": 30,
            "addr1": 5,
            "addr2": 6,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "M1": "T",
        },
    ]


def _assert_online_matches_offline(rows, current):
    frame = pd.DataFrame(rows + [current])
    mappings = fit_category_mappings(frame)
    offline = build_features(frame, CFG, category_mappings=mappings)
    current_offline = offline.loc[offline["TransactionID"].eq(current["TransactionID"])].drop(
        columns=["TransactionID"]
    )

    model = SimpleNamespace(
        category_mappings=mappings,
        selected_v_columns=None,
        feature_names=current_offline.columns.tolist(),
    )
    online = OnlineFeatureStore(FakeMongo(rows), CFG).build(current, model)

    pd.testing.assert_frame_equal(
        current_offline.reset_index(drop=True),
        online.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_offline_and_online_features_match_for_prior_history():
    rows = _rows()
    current = {
        "TransactionID": 4,
        "TransactionDT": 3000,
        "TransactionAmt": 120.0,
        "card1": 111,
        "card2": 10,
        "card3": 20,
        "card5": 30,
        "addr1": 5,
        "addr2": 6,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "M1": "T",
    }
    _assert_online_matches_offline(rows, current)


def test_offline_and_online_features_match_at_same_transaction_timestamp():
    rows = _rows()
    current = {
        "TransactionID": 4,
        "TransactionDT": 2000,
        "TransactionAmt": 120.0,
        "card1": 111,
        "card2": 10,
        "card3": 20,
        "card5": 30,
        "addr1": 5,
        "addr2": 6,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "M1": "T",
    }
    _assert_online_matches_offline(rows, current)
