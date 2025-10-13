import os
from sqlalchemy import create_engine
from .hardware_table import HardwareTable
from .epoch_table import EpochTable
from .country_table import CountryTable
from .paper_document_table import PaperDocumentTable, load_documents
from .paper_text import PaperTextTable, load_texts
from .paper_information_table import PaperInformation

DB_PATH = "data/epoch.db"

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Do not delete the DB; keep incremental state between runs
    engine = create_engine(f"sqlite:///{DB_PATH}")

    hardware_table = HardwareTable.create_table(engine)
    hardware_table.load_table()

    country_table = CountryTable.create_table(engine)
    country_table.load_table()

    epoch_table = EpochTable.create_table(engine)
    epoch_table.load_table()

    epoch_table.create_splits(test_ratio=0.99, dev_ratio=0.005)



    PaperDocumentTable.create_table(engine)
    load_documents(engine, build_splits=["train", "dev"])  # test non construit

    PaperTextTable.create_table(engine)
    load_texts(engine)

    # --- Build PaperInformation from epoch and from text (stub extractor) ---
    # 1) Create an empty variant table and populate from epoch
    variant_name = "paper_information_from_epoch"
    pi = PaperInformation.create_empty_table(engine, variant_name)
    # Reuse the already-created canonical epoch table instance
    pi.load_from_epoch(epoch_table)

    # 2) Extract from text with a stub extractor that writes 0/"0"
    type_map = pi.get_type_map()
    def stub_extract_fn(text: str, column_name: str):
        target_type = type_map.get(column_name, str)
        return "0" if target_type is str else 0
    # Provide an instance-compatible placeholder; the method only uses the table name
    pi.extract_informations_from_text(PaperTextTable, stub_extract_fn)  # type: ignore[arg-type]

