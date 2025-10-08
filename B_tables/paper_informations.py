r"""Structure et définition de la table ``paper_information``.

Le bloc LaTeX ci-dessous rappelle le cahier des charges original, conservé ici
comme documentation de référence.

\begin{table*}[htbp]
\centering
\caption{Tableau récapitulatif des modèles du dataset}
\label{tab:dataset_summary}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l|ccccc@{}}
\hline
\textbf{labels} & \textbf{\textcolor{blue}{Model}} & \textbf{\textcolor{blue}{Abstract}} & \textbf{\textcolor{blue}{Architecture}} & \textbf{Parameters} & \textbf{Country} \\
\hline
\textbf{Format} & Text & Text & ? & Num & Cat \\
\hline
\textbf{Unit} & -- & -- & -- & -- & -- \\
\hline
\textbf{Estimation} & -- & -- & -- & ? & -- \\
\hline
\textbf{Distance} & $d_{JW, rel}(F_{m+p}(.))$ & $d_{JW, rel}(F_{m+p}(.))$ & ? & $d_{rel}$ & $I_{\{id_1 = id_2\}}$ \\
\hline
\end{tabular*}

\vspace{0.6em}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l|ccccc@{}}
\hline
\textbf{labels} & \textbf{Hardware} & \textbf{H Compute} & \textbf{H Power} & \textbf{H Number} & \textbf{Training Time} \\
\hline
\textbf{Format} & Cat & Cat | Num & Cat | Num & Num & Cat \\
\hline
\textbf{Unit} & -- & FLOP & W & -- & h \\
\hline
\textbf{Estimation} & -- & Compute(TH) & Power(TH) & -- & ? \\
\hline
\textbf{Distance} & $I_{\{id_1 = id_2\}}$ & $d_{rel}$ & $d_{rel}$ & $d_{rel}$ & $d_{rel}$ \\
\hline
\end{tabular*}

\vspace{0.6em}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l|cccc@{}}
\hline
\textbf{labels} & \textbf{Year} & \textbf{\textcolor{red}{Training Compute}} & \textbf{\textcolor{orange}{Power Draw}} & \textbf{\textcolor{green}{CO$_2$eq}} \\
\hline
\textbf{Format} & Num & Num & Num & Num \\
\hline
\textbf{Unit} & year & FLOP & Wh & kg CO$_2$eq \\
\hline
\textbf{Estimation} & -- & TT*HN*HC*$\alpha$ & TT*HN*HP*PUE & co2/w(Country)*PD \\
\hline
\textbf{Distance} & $d_1$ & $d_{rel}$ & $d_{rel}$ & $d_{rel}$ \\
\hline
\end{tabular*}
\end{table*}
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy import ForeignKey, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Float, Integer, String, Text

from .base import Base, get_engine


class PaperInformation(Base):
    """ORM représentant une ligne de la table ``paper_information``."""

    __tablename__ = "paper_information"

    id_paper: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    architecture: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    parameters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    id_country: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("country.id_country"), nullable=True
    )

    id_hardware: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("hardware.id_hardware"), nullable=True
    )
    h_compute: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    h_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    h_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    training_time_id_hardware: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("hardware.id_hardware"), nullable=True
    )

    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    training_compute: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    co2eq: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


def _get_variant_table(name: str):
    """Return the SQLAlchemy Table matching ``name`` (clone if needed)."""

    metadata = Base.metadata
    if name in metadata.tables:
        return metadata.tables[name]
    return PaperInformation.__table__.tometadata(metadata, name=name)


INSERTABLE_COLUMNS = [
    column.name
    for column in PaperInformation.__table__.columns
    if column.name != "id_paper"
]

CSV_COLUMN_MAPPING: Mapping[str, str] = {
    "model": "Model",
    "abstract": "Abstract",
    "parameters": "Parameters",
    "training_compute": "Training compute (FLOP)",
    "power_draw": "Training power draw (W)",
    "h_number": "Hardware quantity",
}

ARCHITECTURE_CANDIDATES = (
    "Architecture",
    "Approach",
    "Base model",
)

PUBLICATION_DATE_COLUMN = "Publication date"
COUNTRY_COLUMN = "Country (of organization)"
HARDWARE_COLUMN = "Training hardware"
IGNORED_COUNTRY_TOKENS = {
    "multinational",
    "multiple countries",
    "various",
    "global",
    "unspecified",
    "unknown",
    "n a",
}
COUNTRY_ALIASES = {
    "korea republic of": "korea republic of",
    "republic of korea": "korea republic of",
    "south korea": "korea republic of",
    "korea": "korea republic of",
}

IGNORED_HARDWARE_TOKENS = {
    "unspecified",
    "unknown",
    "n a",
    "none",
    "na",
}


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if pd.isna(value):
        return None
    return value


def _to_float(value):
    value = _clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    value = _clean_value(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_year(value):
    value = _clean_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year)


def _pick_first(row: Mapping[str, object], candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in row:
            value = _clean_value(row[column])
            if value is not None:
                return str(value)
    return None


def _normalize_country(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", without_accents)
    normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
    return normalized


def _split_country_tokens(raw: object) -> list[str]:
    value = _clean_value(raw)
    if value is None:
        return []
    fragments = re.split(r"[,/;]+", value)
    tokens: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        key = fragment.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(fragment)
    return tokens


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
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix += 1
        else:
            break
        if prefix == 4:
            break

    return jaro + prefix * prefix_scale * (1 - jaro)


def _jaro_winkler_distance(s1: str, s2: str) -> float:
    return 1.0 - _jaro_winkler_similarity(s1, s2)


def _select_country_id(raw_value: object, countries: Sequence[tuple[int, str]]) -> Optional[int]:
    tokens = _split_country_tokens(raw_value)
    if not tokens:
        return None

    best_id: Optional[int] = None
    best_distance = float("inf")

    for token in tokens:
        normalized_token = _normalize_country(token)
        if not normalized_token:
            continue
        if normalized_token in COUNTRY_ALIASES:
            normalized_token = COUNTRY_ALIASES[normalized_token]
        if normalized_token in IGNORED_COUNTRY_TOKENS:
            continue
        for country_id, normalized_country in countries:
            if (
                normalized_token in normalized_country
                or normalized_country in normalized_token
            ):
                distance = 0.0
            else:
                distance = _jaro_winkler_distance(normalized_token, normalized_country)
            if distance < best_distance:
                best_distance = distance
                best_id = country_id

    return best_id


def _normalize_hardware(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", without_accents)
    normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
    return normalized


def _split_hardware_tokens(raw: object) -> list[str]:
    value = _clean_value(raw)
    if value is None:
        return []
    fragments = re.split(r"[,/;+]| and |&", value, flags=re.IGNORECASE)
    tokens: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        key = fragment.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(fragment)
    return tokens


def _select_hardware_info(
    raw_value: object,
    hardware_catalog: Sequence[tuple[int, str, Optional[float], Optional[float]]],
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    tokens = _split_hardware_tokens(raw_value)
    if not tokens or not hardware_catalog:
        return None, None, None

    best_id: Optional[int] = None
    best_compute: Optional[float] = None
    best_power: Optional[float] = None
    best_distance = float("inf")

    for token in tokens:
        normalized_token = _normalize_hardware(token)
        if not normalized_token or normalized_token in IGNORED_HARDWARE_TOKENS:
            continue
        for hardware_id, normalized_name, compute, power in hardware_catalog:
            if not normalized_name:
                continue
            if (
                normalized_token in normalized_name
                or normalized_name in normalized_token
            ):
                distance = 0.0
            else:
                distance = _jaro_winkler_distance(normalized_token, normalized_name)
            if distance < best_distance:
                best_distance = distance
                best_id = hardware_id
                best_compute = compute
                best_power = power

    return best_id, best_compute, best_power


def _build_variant_records(
    df: pd.DataFrame,
    *,
    countries: Sequence[tuple[int, str]],
    hardware_catalog: Sequence[tuple[int, str, Optional[float], Optional[float]]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, raw_row in enumerate(df.to_dict(orient="records"), start=1):
        row: dict[str, object] = {"id_paper": index}
        for column in INSERTABLE_COLUMNS:
            row[column] = None

        row["model"] = _clean_value(raw_row.get(CSV_COLUMN_MAPPING["model"]))
        row["abstract"] = _clean_value(raw_row.get(CSV_COLUMN_MAPPING["abstract"]))
        row["architecture"] = _pick_first(raw_row, ARCHITECTURE_CANDIDATES)
        row["parameters"] = _to_int(raw_row.get(CSV_COLUMN_MAPPING["parameters"]))
        row["training_compute"] = _to_float(
            raw_row.get(CSV_COLUMN_MAPPING["training_compute"])
        )
        row["power_draw"] = _to_float(raw_row.get(CSV_COLUMN_MAPPING["power_draw"]))
        row["h_number"] = _to_int(raw_row.get(CSV_COLUMN_MAPPING["h_number"]))
        row["year"] = _to_year(raw_row.get(PUBLICATION_DATE_COLUMN))
        row["id_country"] = _select_country_id(raw_row.get(COUNTRY_COLUMN), countries)
        hardware_id, compute, power = _select_hardware_info(
            raw_row.get(HARDWARE_COLUMN), hardware_catalog
        )
        row["id_hardware"] = hardware_id
        row["h_compute"] = compute
        row["h_power"] = power

        records.append(row)

    return records


def create_paper_information_tables(
    engine: Optional[Engine] = None,
    *,
    variants: Sequence[str] | None = None,
    drop: bool = True,
    variant_sources: Mapping[str, str | Path] | None = None,
) -> None:
    """Create the canonical ``paper_information`` table and optional variants.

    Parameters
    ----------
    engine:
        Database engine to use. When omitted, the project engine is created.
    variants:
        Iterable of additional table names to materialise. For example
        ``["paper_information_true_train", "paper_information_pred_llm_train"]``.
    drop:
        When ``True`` (default) each table is dropped before being (re)created,
        ensuring an empty dataset.
    """

    engine = engine or get_engine()
    sources: dict[str, Path] = {}
    if variant_sources:
        sources = {name: Path(path).resolve() for name, path in variant_sources.items()}

    table_names = [PaperInformation.__tablename__]
    if variants:
        table_names.extend(variants)

    with engine.begin() as connection:
        countries: list[tuple[int, str]] = []
        hardware_catalog: list[tuple[int, str, Optional[float], Optional[float]]] = []
        inspector = inspect(connection)
        if inspector.has_table("country"):
            country_df = pd.read_sql_query(
                "SELECT id_country, country FROM country",
                connection,
            )
            for row in country_df.itertuples(index=False):
                normalized = _normalize_country(str(row.country))
                if normalized:
                    countries.append((int(row.id_country), normalized))
        if inspector.has_table("hardware"):
            hardware_df = pd.read_sql_query(
                "SELECT id_hardware, hardware, compute, power FROM hardware",
                connection,
            )
            for row in hardware_df.itertuples(index=False):
                normalized = _normalize_hardware(str(row.hardware))
                hardware_catalog.append(
                    (
                        int(row.id_hardware),
                        normalized,
                        row.compute if pd.notna(row.compute) else None,
                        row.power if pd.notna(row.power) else None,
                    )
                )

        for name in table_names:
            table = _get_variant_table(name)
            if drop:
                table.drop(connection, checkfirst=True)
            table.create(connection, checkfirst=True)

            source_path = sources.get(name)
            if not source_path:
                continue
            if not source_path.exists():
                raise FileNotFoundError(f"CSV source not found for {name}: {source_path}")

            df = pd.read_csv(source_path)
            records = _build_variant_records(
                df,
                countries=countries,
                hardware_catalog=hardware_catalog,
            )
            if records:
                connection.execute(table.insert(), records)


__all__ = [
    "PaperInformation",
    "create_paper_information_tables",
]
