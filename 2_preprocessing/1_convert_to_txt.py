"""Convert PDF and HTML datasets into TXT files alongside existing data."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict

try:
    import fitz  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyMuPDF (fitz) is required. Install with: pip install pymupdf") from exc


def pdf_to_text(pdf_path: Path, txt_path: Path) -> None:
    with fitz.open(pdf_path) as doc, txt_path.open("w", encoding="utf-8") as out:
        for page in doc:
            out.write(page.get_text())
            out.write("\n")


class ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_p = False
        self._current_chunks: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: D401 - HTMLParser signature
        if tag.lower() == "p" and not self._inside_p:
            self._inside_p = True
            self._current_chunks = []

    def handle_endtag(self, tag: str) -> None:  # noqa: D401 - HTMLParser signature
        if tag.lower() == "p" and self._inside_p:
            paragraph = "".join(self._current_chunks).strip()
            if paragraph:
                self.paragraphs.append(paragraph)
            self._inside_p = False
            self._current_chunks = []

    def handle_data(self, data: str) -> None:  # noqa: D401 - HTMLParser signature
        if self._inside_p:
            self._current_chunks.append(data)


def html_to_text(html_path: Path, txt_path: Path, append: bool) -> None:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    parser = ParagraphExtractor()
    parser.feed(html)

    if not parser.paragraphs:
        return

    mode = "a" if append and txt_path.exists() else "w"
    with txt_path.open(mode, encoding="utf-8") as out:
        for paragraph in parser.paragraphs:
            out.write(paragraph)
            out.write("\n")


def collect_variants(split_dir: Path) -> Dict[str, Dict[str, Path]]:
    variants: Dict[str, Dict[str, Path]] = {}
    for entry in sorted(split_dir.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in {".pdf", ".html", ".txt"}:
            continue
        variants.setdefault(entry.stem, {})[suffix] = entry
    return variants


def process_split(split_dir: Path) -> None:
    if not split_dir.exists():
        print(f"Split not found, skipping: {split_dir}")
        return

    variants = collect_variants(split_dir)
    for stem, files in variants.items():
        if ".txt" in files:
            continue

        target = split_dir / f"{stem}.txt"
        wrote_anything = False

        pdf_path = files.get(".pdf")
        if pdf_path is not None:
            try:
                pdf_to_text(pdf_path, target)
                print(f"Wrote: {target}")
                wrote_anything = True
            except Exception as exc:  # pragma: no cover - runtime safeguard
                print(f"Error converting {pdf_path}: {exc}")

        html_path = files.get(".html")
        if html_path is not None:
            try:
                html_to_text(html_path, target, append=wrote_anything)
                action = "Appended" if wrote_anything else "Wrote"
                print(f"{action} HTML paragraphs to: {target}")
                wrote_anything = True
            except Exception as exc:  # pragma: no cover - runtime safeguard
                print(f"Error converting {html_path}: {exc}")

        if not wrote_anything:
            print(f"No convertible sources for: {split_dir / stem}")


def main(args: list[str] | None = None) -> None:
    args = args if args is not None else sys.argv[1:]

    if args:
        files_root = Path(args[0]).expanduser().resolve()
        splits = args[1:] if len(args) > 1 else ("train", "dev")
    else:
        project_root = Path(__file__).resolve().parent.parent
        files_root = project_root / "data" / "files"
        splits = ("train", "dev")

    for split in splits:
        process_split(files_root / split)


if __name__ == "__main__":
    main()
