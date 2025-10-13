from __future__ import annotations

import re
from urllib.parse import urlparse


_PAT_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]{0,2})?)")
_PAT_PDF = re.compile(r"^https?://.+\.pdf(?:$|[?#])", re.IGNORECASE)
_URL_FINDER = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)


def _arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if not arxiv_id.lower().endswith(".pdf") else f"https://arxiv.org/pdf/{arxiv_id}"


def _is_probable_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if any(ch.isspace() for ch in url):
        return False
    if ",http" in url.lower() or ",https" in url.lower():
        return False
    if "," in url:
        return False
    return True


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in _URL_FINDER.finditer(text or ""):
        candidate = match.group(0).rstrip(" \t\r\n).,;:!?]")
        if _is_probable_url(candidate) and candidate not in urls:
            urls.append(candidate)
    return urls


def _find_first_url(text: str) -> str | None:
    urls = _extract_urls(text)
    return urls[0] if urls else None


def solve_url(raw: str) -> str:
    # Normalise une valeur provenant de la colonne 'Link'
    raw = raw.strip()
    cleaned = raw.rstrip(" \t\r\n).,;:!?]")
    if not cleaned:
        raise ValueError("Empty link")

    # Règle connue: wiley -> PDF direct
    if "onlinelibrary.wiley.com/doi/full/" in cleaned:
        cleaned = cleaned.replace("/doi/full/", "/doi/pdf/", 1)

    # Choix du candidat principal
    if (ids := _PAT_ARXIV.findall(cleaned)):
        candidate = _arxiv_pdf_url(ids[0])
    elif _PAT_PDF.match(cleaned):
        candidate = cleaned
    else:
        candidate = cleaned.splitlines()[0]

    if candidate and _is_probable_url(candidate):
        return candidate

    fallback = _find_first_url(cleaned)
    if fallback:
        return fallback

    raise ValueError("No valid URL found")