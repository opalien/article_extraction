from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine


if __package__ is None or __package__ == "":  # pragma: no cover - script execution
    package_root = Path(__file__).resolve().parent
    parent_dir = package_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from B_tables.infer import InfertInformation, PaperDataset
from B_tables.models.llm import extract_fn as llm_extract_fn
from B_tables.models.qa_squad import extract_fn as qa_extract_fn
from B_tables.tables import create_tables

INFERENCE_RUNS = [
#    ("paper_information_llm_train", llm_extract_fn, "LLM extractor"),
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
    for table_name, extractor, description in INFERENCE_RUNS:
        print(f"Running {description} on table '{table_name}'")
        infer = InfertInformation(engine, table_name, dataset)
        infer.infer(extractor)