from __future__ import annotations

from typing import cast
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore[reportMissingTypeStubs]
from sqlalchemy import CheckConstraint, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import Float, Integer, String

from .base import Base
from .other.utils import USER_AGENT


COUNTRY_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"


class CountryTable(Base):
    __tablename__ = "country"

    id_country: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country: Mapped[str] = mapped_column(String, nullable=False)
    gco2_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (
        CheckConstraint("gco2_kwh >= 0", name="ck_gco2_kwh_nonneg"),
    )

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def connect(engine: Engine) -> CountryTable:
        if not inspect(engine).has_table("country"):
            raise RuntimeError("table country not found")
        return CountryTable(engine)

    @staticmethod
    def create_table(engine: Engine) -> CountryTable:
        cast(Table, CountryTable.__table__).create(engine, checkfirst=True)
        return CountryTable(engine)

    @staticmethod
    def load_df() -> pd.DataFrame:
        # Some environments require a User-Agent to fetch this CSV
        request = Request(COUNTRY_URL, headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response:
            df = pd.read_csv(response)  # type: ignore[reportUnknownMemberType]
        df = df.rename(
            columns={
                "Entity": "country",
                "Year": "year",
                "Carbon intensity of electricity - gCO2/kWh": "gco2_kwh",
            }
        )
        return df

    @staticmethod
    def latest_per_country(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["gco2_kwh"])  # drop rows without value  # type: ignore[reportUnknownMemberType]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")  # type: ignore[reportUnknownMemberType]
        df = df.dropna(subset=["year"]).copy()  # type: ignore[reportUnknownMemberType]
        df["year"] = df["year"].astype(int)

        df = df.sort_values(["country", "year"])  # ensure deterministic
        latest_idx = df.groupby("country")["year"].idxmax()  # type: ignore[reportUnknownMemberType]
        latest = (
            df.loc[latest_idx, ["country", "gco2_kwh"]]
            .sort_values("country")
            .reset_index(drop=True)
        )
        return latest

    def load_table(self) -> None:
        df = self.load_df()
        latest = self.latest_per_country(df)

        with self.engine.begin() as connection:
            tbl: Table = cast(Table, CountryTable.__table__)
            tbl.drop(connection, checkfirst=True)
            tbl.create(connection)

        records: list[dict[str, float | str]] = []
        for country, gco2 in latest[["country", "gco2_kwh"]].itertuples(index=False, name=None):
            records.append({"country": str(country), "gco2_kwh": float(gco2)})

        session: Session = self.SessionLocal()
        try:
            if records:
                session.execute(cast(Table, CountryTable.__table__).insert(), records)
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
        country_table = CountryTable.connect(engine)
    except Exception as e:
        print(e)
        country_table = CountryTable.create_table(engine)
    country_table.load_table()

