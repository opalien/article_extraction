from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, Iterable, Optional

import pandas as pd


DistanceFn = Callable[[object, object], Optional[float]]


@dataclass(frozen=True)
class ColumnDistance:
    name: str
    distance: DistanceFn
    description: str


def format_m_p(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions) / matches)
    ) / 3

    prefix = 0
    for char_left, char_right in zip(s1, s2):
        if char_left == char_right:
            prefix += 1
        else:
            break
        if prefix == 4:
            break

    return jaro + prefix * prefix_scale * (1 - jaro)


def jaro_winkler_distance(left: object, right: object) -> Optional[float]:
    if left is None or right is None:
        return None

    left_fmt = format_m_p(str(left))
    right_fmt = format_m_p(str(right))
    if left_fmt is None or right_fmt is None:
        return None

    similarity = _jaro_winkler_similarity(left_fmt, right_fmt)
    return 1.0 - similarity


def _coerce_number(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def relative_distance(left: object, right: object) -> Optional[float]:
    left_val = _coerce_number(left)
    right_val = _coerce_number(right)
    if left_val is None or right_val is None:
        return None
    denominator = abs(left_val)
    if denominator == 0:
        denominator = abs(right_val)
    if denominator == 0:
        return 0.0
    return abs(left_val - right_val) / denominator


def absolute_distance(left: object, right: object) -> Optional[float]:
    left_val = _coerce_number(left)
    right_val = _coerce_number(right)
    if left_val is None or right_val is None:
        return None
    return abs(left_val - right_val)


def indicator_distance(left: object, right: object) -> Optional[float]:
    if left is None or right is None:
        return None
    return 0.0 if left == right else 1.0


COLUMN_DISTANCES: Iterable[ColumnDistance] = (
    ColumnDistance("model", jaro_winkler_distance, "d_{JW, rel}(F_{m+p}(.))"),
    ColumnDistance("abstract", jaro_winkler_distance, "d_{JW, rel}(F_{m+p}(.))"),
    ColumnDistance("architecture", jaro_winkler_distance, "d_{JW, rel}(F_{m+p}(.))"),
    ColumnDistance("parameters", relative_distance, "d_{rel}"),
    ColumnDistance("id_country", indicator_distance, "I_{id_1 = id_2}"),
    ColumnDistance("id_hardware", indicator_distance, "I_{id_1 = id_2}"),
    ColumnDistance("h_compute", relative_distance, "d_{rel}"),
    ColumnDistance("h_power", relative_distance, "d_{rel}"),
    ColumnDistance("h_number", relative_distance, "d_{rel}"),
    ColumnDistance("training_time_id_hardware", indicator_distance, "I_{id_1 = id_2}"),
    ColumnDistance("year", absolute_distance, "d_1"),
    ColumnDistance("training_compute", relative_distance, "d_{rel}"),
    ColumnDistance("power_draw", relative_distance, "d_{rel}"),
    ColumnDistance("co2eq", relative_distance, "d_{rel}"),
)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _materialise_series(series: pd.Series) -> list[object]:
    materialised: list[object] = []
    for value in series.tolist():
        if value is None or _is_nan(value):
            materialised.append(None)
        else:
            materialised.append(value)
    return materialised


def _summarise_distances(distances: list[float]) -> dict[str, Optional[float]]:
    if not distances:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    if len(distances) == 1:
        single = distances[0]
        return {
            "mean": single,
            "median": single,
            "std": 0.0,
            "min": single,
            "max": single,
        }
    return {
        "mean": mean(distances),
        "median": median(distances),
        "std": pstdev(distances),
        "min": min(distances),
        "max": max(distances),
    }


def evaluate_column(
    merged: pd.DataFrame,
    column_distance: ColumnDistance,
) -> Optional[dict[str, object]]:
    true_column = f"{column_distance.name}_true"
    pred_column = f"{column_distance.name}_pred"
    if true_column not in merged.columns or pred_column not in merged.columns:
        return None

    true_values = _materialise_series(merged[true_column])
    pred_values = _materialise_series(merged[pred_column])

    total_pairs = len(true_values)
    missing_true = 0
    missing_pred = 0
    evaluated = 0
    distances: list[float] = []

    for left, right in zip(true_values, pred_values):
        if left is None:
            missing_true += 1
        if right is None:
            missing_pred += 1
        if left is None or right is None:
            continue
        distance = column_distance.distance(left, right)
        if distance is None:
            continue
        evaluated += 1
        distances.append(float(distance))

    summary = _summarise_distances(distances)
    summary.update(
        {
            "column": column_distance.name,
            "distance": column_distance.description,
            "pairs": total_pairs,
            "evaluated": evaluated,
            "missing_true": missing_true,
            "missing_pred": missing_pred,
        }
    )
    return summary


def evaluate_table(
    connection: sqlite3.Connection,
    true_table: str,
    pred_table: str,
) -> dict[str, object]:
    query_true = f"SELECT * FROM {true_table}"
    query_pred = f"SELECT * FROM {pred_table}"

    true_df = pd.read_sql_query(query_true, connection)
    pred_df = pd.read_sql_query(query_pred, connection)

    merged = true_df.merge(
        pred_df,
        on="id_paper",
        how="outer",
        suffixes=("_true", "_pred"),
    )

    column_reports = []
    for column_distance in COLUMN_DISTANCES:
        report = evaluate_column(merged, column_distance)
        if report is not None:
            column_reports.append(report)

    true_ids = {
        int(value)
        for value in true_df["id_paper"].dropna().astype(int).tolist()
    }
    pred_ids = {
        int(value)
        for value in pred_df["id_paper"].dropna().astype(int).tolist()
    }
    missing_in_pred = sorted(true_ids - pred_ids)
    missing_in_true = sorted(pred_ids - true_ids)

    return {
        "true_table": true_table,
        "pred_table": pred_table,
        "rows_true": len(true_df),
        "rows_pred": len(pred_df),
        "missing_in_pred": {
            "count": len(missing_in_pred),
            "examples": missing_in_pred[:10],
        },
        "missing_in_true": {
            "count": len(missing_in_true),
            "examples": missing_in_true[:10],
        },
        "columns": column_reports,
    }


def list_tables(connection: sqlite3.Connection) -> list[str]:
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cursor.fetchall()
    return sorted(row[0] for row in rows)


def find_prediction_tables(true_table: str, table_names: Iterable[str]) -> list[str]:
    token = "_true_"
    if token not in true_table:
        return []
    head, tail = true_table.split(token, 1)
    prefix = f"{head}_"
    suffix = f"_{tail}" if tail else ""
    predictions = []
    for candidate in table_names:
        if candidate == true_table:
            continue
        if not candidate.startswith(prefix):
            continue
        if suffix and not candidate.endswith(suffix):
            continue
        predictions.append(candidate)
    return sorted(predictions)


def resolve_database_path(raw: str) -> str:
    prefix = "sqlite:///"
    if raw.startswith(prefix):
        raw = raw[len(prefix):]
    return str(Path(raw).expanduser())


def load_connection(database_path: str) -> sqlite3.Connection:
    resolved = resolve_database_path(database_path)
    return sqlite3.connect(resolved)


def benchmark_all(
    connection: sqlite3.Connection,
    true_tables: Iterable[str],
    explicit_predictions: Optional[Iterable[str]] = None,
) -> list[dict[str, object]]:
    table_names = list_tables(connection)
    results = []
    explicit_predictions = set(explicit_predictions or [])
    for true_table in true_tables:
        candidates = find_prediction_tables(true_table, table_names)
        if explicit_predictions:
            candidates = sorted(set(candidates).intersection(explicit_predictions))
        if not candidates:
            continue
        for candidate in candidates:
            results.append(evaluate_table(connection, true_table, candidate))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark predicted paper_information tables against ground truth.",
    )
    parser.add_argument(
        "--database",
        default="exemple.db",
        help="Chemin vers la base SQLite (par défaut: exemple.db).",
    )
    parser.add_argument(
        "--true-table",
        action="append",
        help="Table de référence (peut être fournie plusieurs fois). Par défaut toutes les tables contenant '_true_'.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        help="Table de prédiction à inclure explicitement (optionnelle).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Affiche le rapport complet au format JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with load_connection(args.database) as connection:
        if args.true_table:
            true_tables = args.true_table
        else:
            true_tables = [name for name in list_tables(connection) if "_true_" in name]
        reports = benchmark_all(connection, true_tables, args.prediction)
    if args.as_json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    if not reports:
        print("Aucun benchmark à afficher.")
        return

    for report in reports:
        print(f"# Benchmark {report['pred_table']} vs {report['true_table']}")
        print(f"- lignes true: {report['rows_true']} | lignes préd: {report['rows_pred']}")
        missing_pred = report["missing_in_pred"]
        if missing_pred["count"]:
            print(
                f"- id absents côté prédiction: {missing_pred['count']} "
                f"(exemples: {missing_pred['examples']})"
            )
        missing_true = report["missing_in_true"]
        if missing_true["count"]:
            print(
                f"- id absents côté ground truth: {missing_true['count']} "
                f"(exemples: {missing_true['examples']})"
            )
        for column in report["columns"]:
            mean_value = column["mean"]
            mean_display = f"{mean_value:.4f}" if isinstance(mean_value, float) else "n/a"
            print(
                f"  · {column['column']} ({column['distance']}): "
                f"mean={mean_display}, pairs={column['evaluated']}/{column['pairs']}, "
                f"missing_true={column['missing_true']}, missing_pred={column['missing_pred']}"
            )
        print()


if __name__ == "__main__":
    main()
