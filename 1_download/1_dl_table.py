from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


link = "http://epoch.ai/data/generated/ai_models.zip"


CACHE_DIR = Path(".cache/1_download")
DEST_DIR = Path("data/brut")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"


def _download_zip(url: str, destination: Path) -> Path:
    """Download *url* into *destination*, overwriting any previous file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    tmp_path.replace(destination)
    return destination


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract the contents of *archive* into *destination*."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zip_file:
        members = [info for info in zip_file.infolist() if not info.is_dir()]
        filtered = [info for info in members if Path(info.filename).parts and Path(info.filename).parts[0] != "__MACOSX"]

        if not filtered:
            return

        top_levels = {Path(info.filename).parts[0] for info in filtered}
        drop_top = len(top_levels) == 1

        for info in filtered:
            parts = Path(info.filename).parts
            if drop_top and len(parts) > 1:
                parts = parts[1:]

            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(info) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def main() -> None:
    zip_name = Path(urlsplit(link).path).name or "download.zip"
    zip_path = CACHE_DIR / zip_name

    _download_zip(link, zip_path)
    _extract_zip(zip_path, DEST_DIR)


if __name__ == "__main__":
    main()
