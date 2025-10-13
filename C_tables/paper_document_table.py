from __future__ import annotations

from typing import Any, Type, cast

import pandas as pd  # type: ignore[reportMissingTypeStubs]
import requests
from sqlalchemy import CheckConstraint, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import Integer, String, LargeBinary

from .base import Base
from .url_solver import solve_url


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}
TIMEOUT = 30


class PaperDocumentColumns:
    __abstract__ = True

    id_paper: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_url: Mapped[str] = mapped_column(String, nullable=False)
    document_type: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    __table_args__ = (
        CheckConstraint("document_url is not null", name="ck_document_url_not_null"),
        CheckConstraint("document_type is not null", name="ck_document_type_not_null"),
        CheckConstraint("document is not null", name="ck_document_not_null"),
    )


class AbstractPaperDocumentTable(PaperDocumentColumns, Base):
    __abstract__ = True

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @classmethod
    def connect(cls: Type["AbstractPaperDocumentTable"], engine: Engine):
        if not inspect(engine).has_table(cls.__tablename__):
            raise RuntimeError(f"table {cls.__tablename__} not found")
        return cls(engine)

    @classmethod
    def create_table(cls: Type["AbstractPaperDocumentTable"], engine: Engine):
        cast(Table, cls.__table__).create(engine, checkfirst=True)
        return cls(engine)


class PaperDocumentTable(AbstractPaperDocumentTable):
    __tablename__ = "paper_document"


def _doc_type_from_ct(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    if "html" in ct or "xml" in ct:
        return "html"
    if ct.startswith("text/") or "charset" in ct:
        return "txt"
    if "json" in ct:
        return "json"
    return "bin"


def _fetch_document(url: str) -> tuple[str, bytes]:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    doc_type = _doc_type_from_ct(response.headers.get("Content-Type", ""))
    return doc_type, response.content


def _existing_ids(session: Session, table_cls: Type[AbstractPaperDocumentTable]) -> set[int]:
    rows = session.query(table_cls.id_paper).all()
    return {row[0] for row in rows}


def _iter_epoch_links(engine: Engine, *, split: str) -> pd.DataFrame:
    # Join epoch with epoch_split to get id and link for the requested split
    epoch = pd.read_sql_table("epoch", engine, columns=["id_paper", "link"])  # type: ignore[reportUnknownMemberType]
    mapping = pd.read_sql_table("epoch_split", engine, columns=["id_paper", "split"])  # type: ignore[reportUnknownMemberType]
    df: pd.DataFrame = mapping[mapping["split"] == split].merge(epoch, on="id_paper", how="left")
    # drop rows without links; mypy/pyright sometimes complain about subset types
    result: pd.DataFrame = df.dropna(subset=["link"])  # type: ignore[reportUnknownMemberType]
    return result


def load_documents(
    engine: Engine,
    *,
    build_splits: list[str],  # e.g., ["train", "dev"] (exclude test)
) -> None:
    # Ensure the table schema does not include deprecated 'split' column
    with engine.begin() as connection:
        try:
            cols = [c["name"] for c in inspect(connection).get_columns("paper_document")]
        except Exception:
            cols = []
        tbl: Table = cast(Table, PaperDocumentTable.__table__)
        if "split" in cols:
            tbl.drop(connection, checkfirst=True)
        tbl.create(connection, checkfirst=True)

    session: Session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        already = _existing_ids(session, PaperDocumentTable)
        to_insert: list[dict[str, Any]] = []
        inserted_total = 0
        BATCH = 20

        for split in build_splits:
            df = _iter_epoch_links(engine, split=split)
            filtered: pd.DataFrame = df[~df["id_paper"].isin(list(already))]  # type: ignore[reportUnknownMemberType]
            for id_paper, raw_url in filtered[["id_paper", "link"]].itertuples(index=False, name=None):
                id_paper = int(id_paper)
                raw_url = str(raw_url).strip()
                print(f"[{split}] id_paper={id_paper} raw_url={raw_url}")
                try:
                    final_url = solve_url(raw_url)
                    print(f"[{split}] id_paper={id_paper} solved_url={final_url}")
                except Exception as exc:
                    print(f"[{split}] id_paper={id_paper} solve_url failed: {exc}")
                    continue
                try:
                    doc_type, content = _fetch_document(final_url)
                    print(f"[{split}] id_paper={id_paper} fetched type={doc_type} size={len(content)}B")
                except Exception as exc:
                    print(f"[{split}] id_paper={id_paper} fetch failed: {exc}")
                    continue
                to_insert.append(
                    {
                        "id_paper": id_paper,
                        "document_url": final_url,
                        "document_type": doc_type,
                        "document": content,
                    }
                )

                if len(to_insert) >= BATCH:
                    session.execute(cast(Table, PaperDocumentTable.__table__).insert(), to_insert)
                    session.commit()
                    inserted_total += len(to_insert)
                    print(f"[commit] inserted {inserted_total} rows (batch {len(to_insert)})")
                    already.update(doc["id_paper"] for doc in to_insert)
                    to_insert.clear()

        if to_insert:
            session.execute(cast(Table, PaperDocumentTable.__table__).insert(), to_insert)
            session.commit()
            inserted_total += len(to_insert)
            print(f"[commit] inserted {inserted_total} rows (final batch {len(to_insert)})")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///data/epoch.db")

    # Ensure the document table exists and build for train+dev only; test remains empty
    try:
        PaperDocumentTable.connect(engine)
    except Exception:
        PaperDocumentTable.create_table(engine)
    load_documents(engine, build_splits=["train", "dev"])
