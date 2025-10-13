from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine


if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent
    parent_dir = package_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from B_tables.infer import InfertInformation, PaperDataset
from B_tables.models.llm import extract_fn as llm_extract_fn
from B_tables.models.qa_squad import extract_fn as qa_extract_fn
from B_tables.tables import create_tables
from C_tables.paper_information_table import PaperInformation
from C_tables.paper_text import PaperTextTable
from sqlalchemy.orm import Session

INFERENCE_RUNS = [
    ("paper_information_llm_train", llm_extract_fn, "LLM extractor"),
    ("paper_information_qa_train", qa_extract_fn, "QA Squad extractor"),
]


if __name__ == "__main__":
    engine = create_engine("sqlite:////home/cloud/CDD_2025/code/exemple.db")
    create_tables(
        engine,
        populate=False,
        paper_variants=[table_name for table_name, *_ in INFERENCE_RUNS],
        drop_paper_tables=True,
    )

    dataset = PaperDataset("data/files/train")
    for table_name, extractor, _ in INFERENCE_RUNS:
        infer = InfertInformation(engine, table_name, dataset)
        infer.infer(extractor)

    # ---- Minimal test for C_tables.PaperInformation.extract_informations_from_text ----
    # 1) Build/refresh paper_text from dataset
    with engine.begin() as connection:
        PaperTextTable.__table__.drop(connection, checkfirst=True)
        PaperTextTable.__table__.create(connection, checkfirst=True)

    session: Session = Session(bind=engine)
    try:
        to_insert: list[dict[str, object]] = []
        for file_content, id_paper, i, dir_name in dataset:
            to_insert.append({"id_paper": int(id_paper), "text": str(file_content)})
        if to_insert:
            session.execute(PaperTextTable.__table__.insert(), to_insert)
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # 2) Create an empty variant table for results
    variant_name = "paper_information_stub_train"
    pi = PaperInformation.create_empty_table(engine, variant_name)

    # 3) Build a stub extractor that returns "0" for string columns and 0 otherwise
    type_map = {col.name: col.type.python_type for col in pi._table.columns}

    def stub_extract_fn(text: str, column_name: str):
        target_type = type_map.get(column_name, str)
        return "0" if target_type is str else 0

    # 4) Run extraction from paper_text into our variant table
    pi.extract_informations_from_text(PaperTextTable, stub_extract_fn)
