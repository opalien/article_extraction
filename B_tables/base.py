from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///exemple.db"


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base used across all table modules."""


def get_engine(database_url: str = DATABASE_URL, *, echo: bool = True) -> Engine:
    """Return a new SQLAlchemy Engine configured for the project database."""

    return create_engine(database_url, echo=echo)