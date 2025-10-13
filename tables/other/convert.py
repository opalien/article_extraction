from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd


def to_date(value: Any) -> date | None:
    if pd.isna(value) or value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def to_datetime(value: Any) -> datetime | None:
    if pd.isna(value) or value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def to_boolean(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y", "t"}:
        return True
    if value_str in {"false", "0", "no", "n", "f"}:
        return False
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    coerced = pd.to_numeric(value, errors="coerce")
    if pd.isna(coerced):
        return None
    return float(coerced)


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    coerced = pd.to_numeric(value, errors="coerce")
    if pd.isna(coerced):
        return None
    return int(coerced)


