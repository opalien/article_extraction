from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd
from sqlalchemy.engine import Engine

from .columns import COLUMN_CSV_TO_ATTR, COLUMN_ORDER


def normalize_epoch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_CSV_TO_ATTR)
    for column in COLUMN_ORDER:
        if column not in df.columns:
            df[column] = pd.NA
    return df[COLUMN_ORDER]


def shuffled_split(
    df: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    dev_ratio: float,
    *,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    if any(r < 0 for r in (train_ratio, test_ratio, dev_ratio)):
        raise ValueError("Ratios must be non-negative")
    total = train_ratio + test_ratio + dev_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Ratios must sum to 1.0")

    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    splits = {
        "train": shuffled.iloc[:train_end],
        "test": shuffled.iloc[train_end:test_end],
        "dev": shuffled.iloc[test_end:],
    }
    remainder = n - sum(len(x) for x in splits.values())
    if remainder > 0:
        extra = shuffled.iloc[-remainder:]
        splits["dev"] = pd.concat([splits["dev"], extra], ignore_index=True)
    return splits


