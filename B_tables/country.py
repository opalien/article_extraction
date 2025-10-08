from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import CheckConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import Float, Integer, String

from .base import Base, get_engine


URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"


class Country(Base):
    __tablename__ = "country"

    id_country: Mapped[int] = mapped_column(Integer, primary_key=True)
    country: Mapped[str] = mapped_column(String, nullable=False)
    gco2_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (CheckConstraint("gco2_kwh >= 0", name="ck_gco2_kwh_nonneg"),)


def _prepare_latest_country_frame(url: str) -> pd.DataFrame:
    """Return the latest carbon intensity value per country."""

    df = pd.read_csv(url)
    df.rename(
        columns={
            "Entity": "country",
            "Year": "year",
            "Carbon intensity of electricity - gCO2/kWh": "gco2_kwh",
        },
        inplace=True,
    )

    df = df.dropna(subset=["gco2_kwh"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df = df.sort_values(["country", "year"])
    latest_idx = df.groupby("country")["year"].idxmax()
    latest_df = (
        df.loc[latest_idx, ["country", "gco2_kwh"]]
        .sort_values("country")
        .reset_index(drop=True)
    )
    return latest_df


def refresh_country_table(
    engine: Optional[Engine] = None,
    *,
    url: str = URL,
) -> None:
    """Drop, recreate, and repopulate the country table."""

    engine = engine or get_engine()

    latest_df = _prepare_latest_country_frame(url)

    with engine.begin() as connection:
        Country.__table__.drop(connection, checkfirst=True)
        Country.__table__.create(connection)

    with Session(engine) as session:
        session.add_all(
            Country(country=row["country"], gco2_kwh=row["gco2_kwh"])
            for row in latest_df.to_dict("records")
        )
        session.commit()

    print("Country table refreshed successfully")


if __name__ == "__main__":
    refresh_country_table()
