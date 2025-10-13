from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[reportMissingTypeStubs]
from sqlalchemy import CheckConstraint, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import Float, Integer, String

from .base import Base
from .utils import download_zip, extract_zip


HARDWARE_URL = "https://epoch.ai/data/generated/ml_hardware.zip"
CACHE_DIR = Path(".cache/hardware_table")


class HardwareTable(Base):
    __tablename__ = "hardware"

    id_hardware: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hardware: Mapped[str] = mapped_column(String, nullable=False)
    compute: Mapped[float | None] = mapped_column(Float, nullable=True)
    power: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (
        CheckConstraint("compute >= 0", name="ck_compute_nonneg"),
        CheckConstraint("power >= 0", name="ck_power_nonneg"),
    )

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def connect(engine: Engine) -> HardwareTable:
        if not inspect(engine).has_table("hardware"):
            raise RuntimeError("table hardware not found")
        return HardwareTable(engine)

    @staticmethod
    def create_table(engine: Engine) -> HardwareTable:
        cast(Table, HardwareTable.__table__).create(engine, checkfirst=True)
        return HardwareTable(engine)

    # No alias resolution needed; we rely on the stable Epoch CSV schema

    # No per-value conversion helpers needed; conversions are fixed per column

    def load_df(self) -> pd.DataFrame:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = download_zip(HARDWARE_URL, CACHE_DIR / "hardware.zip")
        extract_dir = CACHE_DIR / "hardware"
        extract_zip(zip_path, extract_dir)

        # Expect fixed filename, fail fast if missing
        csv_path = extract_dir / "ml_hardware.csv"
        return pd.read_csv(csv_path)  # type: ignore[reportUnknownMemberType]

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect the standard Epoch CSV columns
        required_cols = ["Hardware name", "Max performance", "TDP (W)"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected hardware columns: {missing}")

        result = df[["Hardware name", "Max performance", "TDP (W)"]].copy()
        result.columns = ["hardware", "compute", "power"]

        result["hardware"] = result["hardware"].astype(str).str.strip()  # type: ignore[reportUnknownMemberType]
        result["compute"] = pd.to_numeric(result["compute"], errors="coerce")  # type: ignore[reportUnknownMemberType]
        result["power"] = pd.to_numeric(result["power"], errors="coerce") * 1e-3  # type: ignore[reportUnknownMemberType]

        result = result.dropna(subset=["hardware"])  # drop rows without hardware name  # type: ignore[reportUnknownMemberType]
        result = result[result["hardware"] != ""]
        result = result.drop_duplicates(subset=["hardware"])  # type: ignore[reportUnknownMemberType]

        # Normalize dtypes and bounds
        result["compute"] = pd.to_numeric(result["compute"], errors="coerce").clip(lower=0)  # type: ignore[reportUnknownMemberType]
        result["power"] = pd.to_numeric(result["power"], errors="coerce").clip(lower=0)  # type: ignore[reportUnknownMemberType]

        return result

    def load_table(self) -> None:
        df = self.load_df()
        prepared = self._prepare_frame(df)

        # Ensure table exists with constraints, then bulk insert
        with self.engine.begin() as connection:
            tbl: Table = cast(Table, HardwareTable.__table__)
            tbl.drop(connection, checkfirst=True)
            tbl.create(connection)

        records: list[dict[str, Any]] = []
        for hardware, compute, power in prepared[["hardware", "compute", "power"]].itertuples(index=False, name=None):
            records.append(
                {
                    "hardware": str(hardware),
                    "compute": None if compute is None or pd.isna(compute) else float(compute),
                    "power": None if power is None or pd.isna(power) else float(power),
                }
            )

        session: Session = self.SessionLocal()
        try:
            if records:
                session.execute(cast(Table, HardwareTable.__table__).insert(), records)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///data/epoch.db")
    try:
        hardware_table = HardwareTable.connect(engine)
    except Exception as e:
        print(e)
        hardware_table = HardwareTable.create_table(engine)
    hardware_table.load_table()