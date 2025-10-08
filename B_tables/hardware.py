from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from sqlalchemy import CheckConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import Float, Integer, String

from .base import Base, get_engine


SRC = "https://epoch.ai/data/generated/ml_hardware.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_DIR = PROJECT_ROOT / "data" / "brut"
ARCHIVE_NAME = "ml_hardware.zip"


class Hardware(Base):
    __tablename__ = "hardware"

    id_hardware: Mapped[int] = mapped_column(Integer, primary_key=True)
    hardware: Mapped[str] = mapped_column(String, nullable=False)
    compute: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    __table_args__ = (
        CheckConstraint("compute >= 0", name="ck_compute_nonneg"),
        CheckConstraint("power >= 0", name="ck_power_nonneg"),
    )


def _download_archive(
    source_url: str = SRC, *, cache_dir: Path = CACHE_DIR
) -> Path:
    """Download the compressed hardware dataset into the cache directory."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / ARCHIVE_NAME
    if archive_path.exists():
        return archive_path

    try:
        with urlopen(source_url) as response, archive_path.open("wb") as target:
            shutil.copyfileobj(response, target)
    except URLError as exc:  # pragma: no cover - network restrictions handled at runtime
        raise RuntimeError(
            "Unable to download hardware dataset. Provide the archive manually at"
            f" {archive_path}"
        ) from exc

    return archive_path


def _extract_hardware_csv(
    archive_path: Path, *, data_dir: Path = DATA_DIR
) -> Path:
    """Extract the hardware CSV from the archive, ignoring ancillary files."""

    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / "ml_hardware.csv"

    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            filename = Path(info.filename).name
            if not filename:
                continue
            if filename.lower().endswith(".md"):
                continue
            if filename.lower().endswith(".csv"):
                with zf.open(info) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                return target_path

    raise RuntimeError("Hardware CSV not found inside archive")


def _resolve_column(
    df: pd.DataFrame,
    aliases: tuple[str, ...],
    keywords: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> str:
    excluded = set(exclude)

    for alias in aliases:
        if alias in df.columns and alias not in excluded:
            return alias

    for column in df.columns:
        if column in excluded:
            continue
        lower = column.lower()
        if any(keyword in lower for keyword in keywords):
            return column

    raise KeyError(f"Unable to locate a column matching {aliases!r}")


def _convert_compute(series: pd.Series, column_name: str) -> pd.Series:
    lower = column_name.lower()
    factor = 1.0
    if "pflop" in lower:
        factor = 1e15
    elif "tflop" in lower:
        factor = 1e12
    elif "gflop" in lower:
        factor = 1e9
    elif "mflop" in lower:
        factor = 1e6
    elif "kflop" in lower:
        factor = 1e3

    values = pd.to_numeric(series, errors="coerce") * factor
    return values


def _convert_power(series: pd.Series, column_name: str) -> pd.Series:
    lower = column_name.lower()
    factor = 1.0
    if "mw" in lower:
        factor = 1e3
    elif "kw" in lower:
        factor = 1.0
    elif "w" in lower:
        factor = 1e-3

    values = pd.to_numeric(series, errors="coerce") * factor
    return values


def _prepare_hardware_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    hardware_col = _resolve_column(
        df,
        aliases=("hardware", "Hardware", "model", "Model", "chip", "Chip", "name", "Name"),
        keywords=("hardware", "chip", "model", "name"),
    )
    compute_col = _resolve_column(
        df,
        aliases=("compute", "Compute", "Compute (FLOP/s)", "Performance (FLOP/s)"),
        keywords=("flop", "compute"),
        exclude=(hardware_col,),
    )
    power_col = _resolve_column(
        df,
        aliases=(
            "power",
            "Power",
            "Power (kW)",
            "Power (W)",
            "power_kw",
            "TDP (W)",
            "TDP",
        ),
        keywords=("power", " kw", "w)", " watt", "tdp"),
        exclude=(hardware_col, compute_col),
    )

    result = df[[hardware_col, compute_col, power_col]].copy()
    # Normalize column names so downstream processing is deterministic.
    result.columns = ["hardware", "compute", "power"]

    result["hardware"] = result["hardware"].astype(str).str.strip()
    result["compute"] = _convert_compute(result["compute"], compute_col)
    result["power"] = _convert_power(result["power"], power_col)

    result = result.dropna(subset=["hardware"])
    result = result[result["hardware"] != ""]
    result = result.drop_duplicates(subset=["hardware"])

    return result


def refresh_hardware_table(
    engine: Optional[Engine] = None,
    *,
    source_url: str = SRC,
) -> None:
    """Drop, recreate, and populate the hardware table."""

    engine = engine or get_engine()

    archive_path = _download_archive(source_url)
    csv_path = _extract_hardware_csv(archive_path)
    hardware_df = _prepare_hardware_frame(csv_path)

    with engine.begin() as connection:
        Hardware.__table__.drop(connection, checkfirst=True)
        Hardware.__table__.create(connection)

    records = []
    for row in hardware_df.to_dict("records"):
        compute = row["compute"]
        power = row["power"]
        records.append(
            Hardware(
                hardware=row["hardware"],
                compute=None if compute is None or pd.isna(compute) else float(compute),
                power=None if power is None or pd.isna(power) else float(power),
            )
        )

    with Session(engine) as session:
        session.add_all(records)
        session.commit()

    print("Hardware table refreshed successfully")


if __name__ == "__main__":
    refresh_hardware_table()
