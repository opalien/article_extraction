import os
from functools import partial

from sqlalchemy import create_engine

from config import GENERATION_KWARGS, MAX_CONTEXT_TOKENS, MODEL_ID, WINDOW_STRIDE_TOKENS
from models.llm import extract_fn as _extract_fn
from tables.country_table import CountryTable
from tables.hardware_table import HardwareTable
from tables.paper_information_table import PaperInformation
from tables.paper_text_table import PaperTextTable


LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", MODEL_ID)

extract_fn = partial(
    _extract_fn,
    model_id=LLM_MODEL_ID,
    window_tokens=MAX_CONTEXT_TOKENS,
    stride_tokens=WINDOW_STRIDE_TOKENS,
    max_new_tokens=GENERATION_KWARGS["max_new_tokens"],
    temperature=GENERATION_KWARGS["temperature"],
    top_p=GENERATION_KWARGS["top_p"],
)


if __name__ == "__main__":
    db_path = "data/epoch.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 10})

    HardwareTable.connect(engine)
    CountryTable.connect(engine)
    text_table = PaperTextTable.connect(engine)

    variant_name = "llm"
    paper_info = PaperInformation.create_empty_table(engine, variant_name)
    paper_info.extract_informations_from_text_per_cell(text_table, extract_fn)
    paper_info.complete_informations()
