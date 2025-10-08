from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

from .base import *


DEFAULT_PAPER_VARIANTS = (
    "paper_information_true_train",
    "paper_information_true_test",
)

DEFAULT_PAPER_SOURCES: Mapping[str, Path] = {
    "paper_information_true_train": Path("data/tables/train.csv"),
    "paper_information_true_test": Path("data/tables/test.csv"),
}





def _ensure_models_loaded() -> None:
    """Import ORM models so that metadata knows about every table."""

    from . import country, hardware, paper_informations  # noqa: F401


def create_tables(
    engine: Optional[Engine] = None,
    *,
    populate: bool = True,
    paper_variants: Sequence[str] | None = None,
    paper_variant_sources: Mapping[str, str | Path] | None = None,
    drop_paper_tables: bool = True,
) -> Engine:
    """Create (and optionally populate) every project table.

    Parameters
    ----------
    engine
        Optional pre-configured SQLAlchemy engine. When omitted a new engine
        targeting ``DATABASE_URL`` is created.
    populate
        When ``True`` (default) the domain-specific modules are responsible
        for dropping, recreating, and loading their respective tables. Passing
        ``False`` only creates an empty schema without triggering data loads.
    paper_variants
        Extra ``paper_information``-like table names to materialise (e.g.
        ``["paper_information_true_train", "paper_information_pred_llm_train"]``).
        When omitted, ``paper_information_true_train`` et
        ``paper_information_true_test`` sont créées par défaut.
    paper_variant_sources
        Optional mapping ``table_name -> csv_path`` used to seed the table. Par
        défaut, les variantes ``paper_information_true_train`` et
        ``paper_information_true_test`` sont alimentées depuis
        ``data/tables/train.csv`` et ``data/tables/test.csv``.
    drop_paper_tables
        When ``True`` the paper tables are dropped prior to creation so that
        each run starts with empty datasets.
    """

    engine = engine or get_engine()
    if paper_variants is None:
        paper_variants = DEFAULT_PAPER_VARIANTS
    if paper_variant_sources is None:
        paper_variant_sources = DEFAULT_PAPER_SOURCES

    _ensure_models_loaded()

    if populate:
        from .country import refresh_country_table
        from .hardware import refresh_hardware_table
        from .paper_informations import create_paper_information_tables

        refresh_country_table(engine)
        refresh_hardware_table(engine)
        create_paper_information_tables(
            engine,
            variants=paper_variants,
            drop=drop_paper_tables,
            variant_sources=paper_variant_sources,
        )
    else:
        Base.metadata.create_all(engine)
        from .paper_informations import create_paper_information_tables

        create_paper_information_tables(
            engine,
            variants=paper_variants,
            drop=drop_paper_tables,
            variant_sources=paper_variant_sources,
        )

    return engine


def delete_tables(engine: Optional[Engine] = None) -> None:
    """Drop every project table from the database."""

    engine = engine or get_engine()
    with engine.begin() as connection:
        metadata = MetaData()
        metadata.reflect(bind=connection)
        metadata.drop_all(bind=connection)
