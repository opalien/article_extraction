import os
from sqlalchemy import create_engine
from tables.hardware_table import HardwareTable
from tables.epoch_table import EpochTable
from tables.country_table import CountryTable
from tables.paper_document_table import PaperDocumentTable, load_documents
from tables.paper_text_table import PaperTextTable, load_texts
from tables.paper_information_table import PaperInformation

def create_tables(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)

    os.makedirs(".cache", exist_ok=True)
        
    engine = create_engine(f"sqlite:///{db_path}")

    hardware_table = HardwareTable.create_table(engine)
    hardware_table.load_table()

    country_table = CountryTable.create_table(engine)
    country_table.load_table()

    epoch_table = EpochTable.create_table(engine)
    epoch_table.load_table()

    epoch_table.create_splits(test_ratio=0.99, dev_ratio=0.005)


    PaperDocumentTable.create_table(engine)
    load_documents(engine, build_splits=["train", "dev"])

    PaperTextTable.create_table(engine)
    load_texts(engine)
    

    # build epoch (and optionally other tables) before populating paper_information
    # build hardware and country first (lookups), then epoch
    #hardware_table = HardwareTable.create_table(engine)
    #hardware_table.load_table()

    #country_table = CountryTable.create_table(engine)
    #country_table.load_table()

    #epoch_table = EpochTable.create_table(engine)
    #epoch_table.load_table()

    variant_name = "paper_information_from_epoch"
    pi = PaperInformation.create_empty_table(engine, variant_name)
    pi.load_from_epoch(epoch_table, country_table, hardware_table)


    #type_map = pi.get_type_map()
    #def stub_extract_fn(text: str, column_name: str):
    #    target_type = type_map.get(column_name, str)
    #    return "0" if target_type is str else 0
        
    #pi.extract_informations_from_text(PaperTextTable, stub_extract_fn)  # type: ignore[arg-type]



if __name__ == "__main__":
    DB_PATH = "data/epoch.db"
    create_tables(DB_PATH)