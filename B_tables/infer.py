
import os
import re
from typing import Any, Callable

from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.orm import Session



class PaperDataset():
    def __init__(self, path: str):

        list_dirs = os.listdir(path)
        list_txt = list(filter(lambda x: re.fullmatch(r"\d+\.txt", x), list_dirs))
        sorted_list_txt = sorted(list_txt, key=lambda x: int(x.split(".")[0]))

        self.list_dirs = sorted_list_txt
        self.path = path        

    def __len__(self):
        return len(self.list_dirs)    

    def __getitem__(self, idx: int):
        return self.list_dirs[idx]

    def __iter__(self):
        for i, dir_name in enumerate(self.list_dirs):
            with open(os.path.join(self.path, dir_name), "r") as f:
                yield (f.read(), int(dir_name.split(".")[0]), i, dir_name)


def extract_fn(file_content: str, column_name: str) -> Any:
    return 10


NUMERAL_MULTIPLIERS: dict[str, float] = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}


SHORT_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}


def _parse_numeric(value: Any) -> float | None:
    """Attempt to coerce model answers like "1.5B" or "42 million" into floats."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None

    # Handle compact suffixes such as 1.2b or 300k.
    suffix_match = re.match(r"([-+]?\d*\.?\d+)\s*([kmbt])\b", text)
    if suffix_match:
        base = float(suffix_match.group(1))
        suffix = suffix_match.group(2)
        return base * SHORT_SUFFIX_MULTIPLIERS[suffix]

    cleaned = text.replace(",", "")
    # Capture scientific notation or decimal numbers inside the string.
    number_match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", cleaned)
    if number_match:
        number = float(number_match.group())
        for word, factor in NUMERAL_MULTIPLIERS.items():
            if re.search(rf"\b{word}\b", cleaned):
                return number * factor
        return number

    return None


def _coerce_value(value: Any, target_type: type) -> Any:
    """Return ``value`` cast to ``target_type`` or ``None`` when not possible."""

    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped

    if target_type is str:
        return str(value)

    if target_type is float:
        parsed = _parse_numeric(value)
        return float(parsed) if parsed is not None else None

    if target_type is int:
        parsed = _parse_numeric(value)
        return int(parsed) if parsed is not None else None

    # Fallback: try direct construction, silence conversion errors.
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return None


class InfertInformation():
    def __init__(self, engine: Engine, table_name: str, dataset: PaperDataset):
        self.engine = engine
        self.dataset = dataset
        metadata = MetaData()
        self.table = Table(table_name, metadata, autoload_with=engine)
        self.column_names = [column.name for column in self.table.columns]
        self.column_types = [column.type.python_type for column in self.table.columns]
        self._type_map = dict(zip(self.column_names, self.column_types))
        
    def empty_table(self):
        with Session(self.engine) as session:
            session.execute(self.table.delete())
            session.commit()


    def complete_table(self):
        pass 
        

    def infer(self, extract_fn: Callable[[str, str], Any]) -> None:
        with Session(self.engine) as session:
            for file_content, id_paper, i, dir_name in self.dataset:
                print(f"{i=}: Processing {dir_name}")
                columns: dict[str, Any] = {}
                for name in self.column_names[1:]:
                    res_brut = extract_fn(file_content, name)
                    res_typed = _coerce_value(res_brut, self._type_map[name])
                    columns[name] = res_typed
                print(columns)

                pk_name = self.column_names[0]
                pk_column = self.table.c[pk_name]

                update_stmt = (
                    self.table.update()
                    .where(pk_column == id_paper)
                    .values(**columns)
                )
                result = session.execute(update_stmt)

                if result.rowcount == 0:
                    session.execute(
                        self.table.insert().values({pk_name: id_paper, **columns})
                    )
            session.commit()



if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:////home/cloud/CDD_2025/code/exemple.db")
    dataset = PaperDataset("data/files/train")
    infer = InfertInformation(engine, "paper_information", dataset)
    infer.infer(extract_fn)
