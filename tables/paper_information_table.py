from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Type, cast

import pandas as pd  # type: ignore[reportMissingTypeStubs]
from sqlalchemy import ForeignKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.sql.schema import Table
from sqlalchemy.types import Float, Integer, String, Text

from .base import Base


class PaperInformation(Base):
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


    # --- infrastructure helpers ---
    def __init__(self, engine: Engine, table_name: Optional[str] = None):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.table_name = table_name or self.__tablename__
        self._table: Table = _get_variant_table(self.table_name)

    def get_type_map(self) -> dict[str, type]:
        return {col.name: col.type.python_type for col in self._table.columns}

    @classmethod
    def connect(cls: Type["PaperInformation"], engine: Engine, name: str = "paper_information") -> "PaperInformation":
        return cls(engine, table_name=name)

    @classmethod
    def create_empty_table(cls: Type["PaperInformation"], engine: Engine, name: str = "paper_information") -> "PaperInformation":
        tbl = _get_variant_table(name)
        with engine.begin() as connection:
            try:
                connection.exec_driver_sql("PRAGMA busy_timeout=5000")
            except Exception:
                pass
            tbl.drop(connection, checkfirst=True)
            tbl.create(connection, checkfirst=True)
        return cls(engine, table_name=name)


    # --- data loaders ---
    def load_from_epoch(self, epoch_table: Any, country_table: Any, hardware_table: Any) -> None:
        # ensure destination exists
        with self.engine.begin() as connection:
            self._table.create(connection, checkfirst=True)

        # lookups from country and hardware tables (assumes they exist)
        countries: list[tuple[int, str]] = []
        hardware_catalog: list[tuple[int, str, Optional[float], Optional[float]]] = []
        country_tbl_name = cast(str, country_table.__tablename__)
        hardware_tbl_name = cast(str, hardware_table.__tablename__)
        country_df = pd.read_sql_query(
            f"SELECT id_country, country FROM {country_tbl_name}", self.engine
        )  # type: ignore[reportUnknownMemberType]
        for row in country_df.itertuples(index=False):
            normalized = _normalize_country(str(row.country))
            if normalized:
                countries.append((int(row.id_country), normalized))  # type: ignore[arg-type]
        hardware_df = pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
            f"SELECT id_hardware, hardware, compute, power FROM {hardware_tbl_name}",
            self.engine,
        )
        for row in hardware_df.itertuples(index=False):
            normalized = _normalize_hardware(str(row.hardware))
            compute = None if pd.isna(row.compute) else float(row.compute)  # type: ignore[arg-type]
            power = None if pd.isna(row.power) else float(row.power)  # type: ignore[arg-type]
            hardware_catalog.append((int(row.id_hardware), normalized, compute, power))  # type: ignore[arg-type]

        # source epoch columns
        epoch_table_name = cast(str, epoch_table.__tablename__)
        cols = [
            "id_paper",
            "model",
            "abstract",
            "approach",
            "parameters",
            "training_compute",
            "training_power_draw_w",
            "hardware_quantity",
            "publication_date",
            "country_of_organization",
            "training_hardware",
        ]
        df = pd.read_sql_table(epoch_table_name, self.engine, columns=cols)  # type: ignore[reportUnknownMemberType]

        records: list[dict[str, Any]] = []
        for row in df.to_dict(orient="records"):  # type: ignore[reportUnknownMemberType]
            rec: dict[str, Any] = {k: None for k in INSERTABLE_COLUMNS}
            rec["id_paper"] = int(row["id_paper"]) if row.get("id_paper") is not None else None
            rec["model"] = _clean_value(row.get("model"))
            rec["abstract"] = _clean_value(row.get("abstract"))
            rec["architecture"] = _clean_value(row.get("approach"))
            rec["parameters"] = _to_int(row.get("parameters"))
            rec["training_compute"] = _to_float(row.get("training_compute"))
            rec["power_draw"] = _to_float(row.get("training_power_draw_w"))
            rec["h_number"] = _to_int(row.get("hardware_quantity"))
            rec["year"] = _to_year(row.get("publication_date"))

            rec["id_country"] = _select_country_id(row.get("country_of_organization"), countries)
            hw_id, hw_compute, hw_power = _select_hardware_info(row.get("training_hardware"), hardware_catalog)
            rec["id_hardware"] = hw_id
            rec["h_compute"] = hw_compute
            rec["h_power"] = hw_power

            records.append(rec)

        session: Session = self.SessionLocal()
        try:
            if records:
                session.execute(self._table.insert(), records)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


    # --- inference from text ---
    def extract_informations_from_text(self, table: Any, extract_fn: Callable[[str, str], Any]) -> None:
        source_table_name = table.__tablename__
        source_engine = table.engine

        # ensure destination exists
        with self.engine.begin() as connection:
            self._table.create(connection, checkfirst=True)

        # reflect types of destination columns
        column_names: list[str] = [col.name for col in self._table.columns]
        type_map: dict[str, type] = {col.name: col.type.python_type for col in self._table.columns}

        # load all texts (simple approach suitable for testing)
        df = pd.read_sql_table(source_table_name, source_engine, columns=["id_paper", "text"])  # type: ignore[reportUnknownMemberType]
        if df.empty:
            return

        session: Session = self.SessionLocal()
        try:
            processed = 0
            for id_paper, text_value in df[["id_paper", "text"]].itertuples(index=False, name=None):
                id_paper = int(id_paper)
                text_value = str(text_value)

                values: dict[str, Any] = {}
                for name in column_names[1:]:  # skip primary key
                    raw = extract_fn(text_value, name)
                    coerced = _coerce_value(raw, type_map[name])
                    values[name] = coerced

                pk_col = self._table.c.id_paper
                update_stmt = self._table.update().where(pk_col == id_paper).values(**values)
                result = session.execute(update_stmt)
                if result.rowcount == 0:
                    session.execute(self._table.insert().values({"id_paper": id_paper, **values}))
                processed += 1
                # Commit frequently to persist partial progress (e.g., if interrupted)
                if processed % 1 == 0:
                    session.commit()
            # Final commit in case last batch had remainder
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


    def complete_informations(self) -> None:
        pass


# ---------- helpers (adapted from B_tables.paper_informations) ----------

def _get_variant_table(name: str) -> Table:
    metadata = Base.metadata
    if name in metadata.tables:
        return metadata.tables[name]
    return cast(Table, PaperInformation.__table__.tometadata(metadata, name=name))  # type: ignore[attr-defined]


INSERTABLE_COLUMNS: list[str] = [
    column.name for column in cast(Table, PaperInformation.__table__).columns
]


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if pd.isna(value):  # type: ignore[reportUnknownMemberType]
        return None
    return str(value)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        f = float(value)
        # Treat NaN as missing
        try:
            import math
            if math.isnan(f):
                return None
        except Exception:
            pass
        return f
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    return int(f) if f is not None else None


def _to_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")  # type: ignore[reportUnknownMemberType]
    if pd.isna(parsed):  # type: ignore[reportUnknownMemberType]
        return None
    return int(parsed.year)  # type: ignore[reportUnknownMemberType]


# country matching
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


def _normalize_country(value: str) -> str:
    value = value.strip().lower()
    result_chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch.isspace():
            result_chars.append(ch)
        else:
            result_chars.append(" ")
    normalized = " ".join("".join(result_chars).split())
    return normalized


def _split_country_tokens(raw: object) -> list[str]:
    value = _clean_value(raw)
    if value is None:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for fragment in value.replace("/", ",").replace(";", ",").split(","):
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

    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions) / matches)) / 3

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
            if (normalized_token in normalized_country) or (normalized_country in normalized_token):
                distance = 0.0
            else:
                distance = _jaro_winkler_distance(normalized_token, normalized_country)
            if distance < best_distance:
                best_distance = distance
                best_id = country_id

    return best_id


# hardware matching
def _normalize_hardware(value: str) -> str:
    value = value.strip().lower()
    result_chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch.isspace():
            result_chars.append(ch)
        else:
            result_chars.append(" ")
    normalized = " ".join("".join(result_chars).split())
    return normalized


def _split_hardware_tokens(raw: object) -> list[str]:
    value = _clean_value(raw)
    if value is None:
        return []
    tmp = value.replace("/", ",").replace(";", ",").replace("+", ",").replace("&", ",")
    tmp = tmp.replace(" and ", ",")
    tokens: list[str] = []
    seen: set[str] = set()
    for fragment in tmp.split(","):
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
        if not normalized_token:
            continue
        for hardware_id, normalized_name, compute, power in hardware_catalog:
            if not normalized_name:
                continue
            if (normalized_token in normalized_name) or (normalized_name in normalized_token):
                distance = 0.0
            else:
                distance = _jaro_winkler_distance(normalized_token, normalized_name)
            if distance < best_distance:
                best_distance = distance
                best_id = hardware_id
                best_compute = compute
                best_power = power

    return best_id, best_compute, best_power


# coercion for inference values
NUMERAL_MULTIPLIERS: dict[str, float] = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

SHORT_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None

    # remove thousands separators and spaces
    cleaned = text.replace(",", "").replace(" ", "")

    # simple suffix handling without regex
    if cleaned[-1:] in SHORT_SUFFIX_MULTIPLIERS:
        try:
            base = float(cleaned[:-1])
            return base * SHORT_SUFFIX_MULTIPLIERS[cleaned[-1]]
        except ValueError:
            return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_value(value: Any, target_type: type) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped

    if target_type is str:
        return str(value)

    if target_type is float:
        parsed = _parse_numeric(value)
        return float(parsed) if parsed is not None else None

    if target_type is int:
        parsed = _parse_numeric(value)
        return int(parsed) if parsed is not None else None

    try:
        return target_type(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PaperInformation",
]