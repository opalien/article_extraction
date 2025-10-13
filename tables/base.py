from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///exemple.db"



class Base(DeclarativeBase):
    pass

    @staticmethod
    def get_engine(database_url: str = DATABASE_URL, *, echo: bool = True) -> Engine:
        return create_engine(database_url, echo=echo)