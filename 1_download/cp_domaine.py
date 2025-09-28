# copie la colonne "Link" de data/train.csv dans .cache/domaine.txt
from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path("data/train.csv")
OUTPUT = Path(".cache/domaine.txt")
COLUMN = "Link"


def read_links(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {path} est introuvable.")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if COLUMN not in reader.fieldnames:
            raise KeyError(f"La colonne '{COLUMN}' est absente de {path}.")
        return [row[COLUMN] for row in reader if isinstance(row.get(COLUMN), str) and row[COLUMN].strip()]


def write_links(links: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(links) + ("\n" if links else ""), encoding="utf-8")


def main() -> None:
    links = read_links(SOURCE)
    write_links(links, OUTPUT)


if __name__ == "__main__":
    main()
