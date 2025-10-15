import os
from sqlalchemy import create_engine
from tables.hardware_table import HardwareTable
from tables.epoch_table import EpochTable
from tables.country_table import CountryTable
# from tables.paper_document_table import PaperDocumentTable, load_documents
from tables.paper_text_table import PaperTextTable, load_texts
from tables.paper_information_table import PaperInformation

from models.llm import extract_fn as extract_fn_llm
from models.qa_squad import extract_fn as extract_fn_qa_squad
from functools import partial


question_map = {
    "model": "What is the name of the proposed model in this paper ?",
    #"abstract": "Provide the abstract of the paper in one sentence.",
}

extract_fn_qa_curried = partial(
    extract_fn_qa_squad,
    model_id="FredNajjar/bigbird-QA-squad_v2.3",
    question_map=question_map,
)

LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "openai/gpt-oss-20b")
extract_fn_llm_curried = partial(
    extract_fn_llm,
    model_id=LLM_MODEL_ID,
    question_map=question_map,
)



if __name__ == "__main__":
    DB_PATH = "data/epoch.db"
    db_path = DB_PATH

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 10})

    hardware_table = HardwareTable.connect(engine)
    country_table = CountryTable.connect(engine)
    text_table = PaperTextTable.connect(engine)


    variant_name = "llm"
    pi = PaperInformation.create_empty_table(engine, variant_name)
    pi.extract_informations_from_text(text_table, extract_fn_llm_curried)

    variant_name = "qa_squad"
    pi = PaperInformation.create_empty_table(engine, variant_name)
    pi.extract_informations_from_text(text_table, extract_fn_qa_curried)