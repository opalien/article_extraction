from __future__ import annotations

from typing import Any

from html.parser import HTMLParser

import pandas as pd
from sqlalchemy import CheckConstraint, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import Integer, Text

from .base import Base
from .paper_document_table import PaperDocumentTable


class PaperTextTable(Base):
    __tablename__ = "paper_text"

    id_paper: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("text is not null", name="ck_text_not_null"),
    )

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def connect(engine: Engine) -> "PaperTextTable":
        if not inspect(engine).has_table("paper_text"):
            raise RuntimeError("table paper_text not found")
        return PaperTextTable(engine)

    @staticmethod
    def create_table(engine: Engine) -> "PaperTextTable":
        PaperTextTable.metadata.create_all(engine, tables=[PaperTextTable.__table__])
        return PaperTextTable(engine)


class ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_p = False
        self._current_chunks: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "p" and not self._inside_p:
            self._inside_p = True
            self._current_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p" and self._inside_p:
            paragraph = "".join(self._current_chunks).strip()
            if paragraph:
                self.paragraphs.append(paragraph)
            self._inside_p = False
            self._current_chunks = []

    def handle_data(self, data: str) -> None:
        if self._inside_p:
            self._current_chunks.append(data)


def _pdf_bytes_to_text(data: bytes) -> str:
    try:
        import fitz  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required to extract PDF text") from exc

    text_chunks: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text())
    return "\n".join(text_chunks)


def _html_bytes_to_text(data: bytes) -> str:
    html = data.decode("utf-8", errors="ignore")
    parser = ParagraphExtractor()
    parser.feed(html)
    if not parser.paragraphs:
        return ""
    return "\n".join(parser.paragraphs)


def _txt_bytes_to_text(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _to_text(document_type: str, document: bytes) -> str:
    kind = (document_type or "").lower()
    if kind == "pdf":
        return _pdf_bytes_to_text(document)
    if kind == "html":
        return _html_bytes_to_text(document)
    if kind == "txt":
        return _txt_bytes_to_text(document)
    if kind == "json":
        return document.decode("utf-8", errors="ignore")
    return ""


def _existing_ids(session: Session) -> set[int]:
    rows = session.query(PaperTextTable.id_paper).all()
    return {row[0] for row in rows}


def load_texts(engine: Engine) -> None:
    with engine.begin() as connection:
        PaperTextTable.__table__.create(connection, checkfirst=True)

    session: Session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        existing = _existing_ids(session)
        # Fetch candidate documents not yet converted
        df = pd.read_sql_table(
            "paper_document",
            engine,
            columns=["id_paper", "document_type", "document"],
        )
        if df.empty:
            print("No documents to convert.")
            return
        df = df[~df["id_paper"].isin(existing)]
        if df.empty:
            print("All documents already converted.")
            return

        to_insert: list[dict[str, Any]] = []
        BATCH = 20
        processed = 0
        for row in df.to_dict(orient="records"):
            id_paper = int(row["id_paper"])  # type: ignore[arg-type]
            doc_type = str(row["document_type"])
            data: bytes = row["document"]  # type: ignore[assignment]
            print(f"[text] id_paper={id_paper} type={doc_type} bytes={len(data)}")
            try:
                text_value = _to_text(doc_type, data)
            except Exception as exc:
                print(f"[text] id_paper={id_paper} conversion failed: {exc}")
                continue
            if not text_value:
                print(f"[text] id_paper={id_paper} produced empty text; skipped")
                continue
            to_insert.append({"id_paper": id_paper, "text": text_value})
            processed += 1

            if len(to_insert) >= BATCH:
                session.bulk_insert_mappings(PaperTextTable, to_insert)
                session.commit()
                print(f"[text][commit] inserted {processed} rows (batch {len(to_insert)})")
                to_insert.clear()

        if to_insert:
            session.bulk_insert_mappings(PaperTextTable, to_insert)
            session.commit()
            print(f"[text][commit] inserted {processed} rows (final batch {len(to_insert)})")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///data/epoch.db")
    try:
        PaperTextTable.connect(engine)
    except Exception:
        PaperTextTable.create_table(engine)
    load_texts(engine)





