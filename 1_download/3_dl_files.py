from __future__ import annotations

import csv
import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse

import requests
import zlib


LOG_FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


TRAIN_PATH = Path("data/tables/train.csv")
DEV_PATH = Path("data/tables/dev.csv")
ALL_MODELS_PATH = Path("data/brut/all_ai_models.csv")
OUTPUT_DIR = Path("data/files")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
PDF_ACCEPT = "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"
CHUNK_SIZE = 8192
TIMEOUT = 60
RETRY_STATUSES = {403, 429}


pat_arxiv = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]{0,2})?)")
pat_pdf = re.compile(r"^https?://.+\.pdf(?:$|[?#])", re.IGNORECASE)
URL_FINDER = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)


@dataclass(frozen=True)
class DownloadTarget:
    url: str
    referer: Optional[str] = None
    warmup: Optional[str] = None
    expects_pdf: Optional[bool] = None


@dataclass
class DownloadOutcome:
    path: Optional[Path]
    url: str


def iter_tables() -> Iterable[tuple[str, Path]]:
    found_any = False
    for split, path in ("train", TRAIN_PATH), ("dev", DEV_PATH):
        if path.exists():
            found_any = True
            yield split, path

    if not found_any and ALL_MODELS_PATH.exists():
        logging.info("Aucun split train/dev trouvé; utilisation de data/brut/all_ai_models.csv")
        yield "train", ALL_MODELS_PATH


def make_headers(accept: str, referer: Optional[str], target_url: str) -> dict[str, str]:
    headers = dict(BASE_HEADERS)
    headers["Accept"] = accept
    if referer:
        headers["Referer"] = referer

    parsed_target = urlparse(target_url)
    parsed_referer = urlparse(referer) if referer else None
    if parsed_referer and parsed_referer.netloc == parsed_target.netloc:
        headers.setdefault("Sec-Fetch-Site", "same-origin")
    else:
        headers.setdefault("Sec-Fetch-Site", "cross-site")
    headers.setdefault("Sec-Fetch-Mode", "navigate")
    headers.setdefault("Sec-Fetch-Dest", "document")
    headers.setdefault("Sec-Fetch-User", "?1")
    return headers


def arxiv_pdf_url(arxiv_id: str) -> str:
    if not arxiv_id.lower().endswith(".pdf"):
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return f"https://arxiv.org/pdf/{arxiv_id}"


def normalise_link(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None

    raw = raw.strip()
    cleaned = raw.rstrip(" \t\r\n).,;:!?]")
    if not cleaned:
        return None

    if "onlinelibrary.wiley.com/doi/full/" in cleaned:
        cleaned = cleaned.replace("/doi/full/", "/doi/pdf/", 1)

    match cleaned:
        case _ if (ids := pat_arxiv.findall(cleaned)):
            candidate = arxiv_pdf_url(ids[0])
        case _ if pat_pdf.match(cleaned):
            candidate = cleaned
        case _:
            candidate = cleaned.splitlines()[0]

    if candidate and is_probable_url(candidate):
        return candidate

    fallback = find_first_url(cleaned)
    if fallback:
        return fallback

    return None


def pick_extension(url: str, content_type: str) -> str:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct or "xml" in ct:
        return ".html"
    if ct.startswith("text/") or "charset" in ct:
        return ".txt"
    if "json" in ct:
        return ".json"

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm"}:
        return suffix

    return ".bin"


def is_probable_url(url: str) -> bool:
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


def find_first_url(text: str) -> Optional[str]:
    urls = extract_urls(text)
    return urls[0] if urls else None


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_FINDER.finditer(text or ""):
        candidate = match.group(0).rstrip(" \t\r\n).,;:!?]")
        if is_probable_url(candidate) and candidate not in urls:
            urls.append(candidate)
    return urls


def write_placeholder(dest_base: Path, split: str, reason: str | None = None) -> None:
    placeholder = dest_base.with_suffix(".txt")
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    existed = placeholder.exists()
    placeholder.write_text((reason or "").strip() + "\n" if reason else "", encoding="utf-8")
    if existed:
        logging.info("[%s] Placeholder conservé: %s", split, placeholder.as_posix())
    else:
        logging.info("[%s] Placeholder créé: %s", split, placeholder.as_posix())


def decompress_payload(data: bytes, encoding: str, url: str) -> bytes:
    encoding = (encoding or "").lower()
    if not encoding:
        return data

    encodings = [enc.strip() for enc in encoding.split(",") if enc.strip()]
    current = data

    for enc in encodings:
        try:
            if enc in {"gzip", "x-gzip"}:
                current = gzip.decompress(current)
                continue
            if enc == "deflate":
                try:
                    current = zlib.decompress(current)
                except zlib.error:
                    current = zlib.decompress(current, -zlib.MAX_WBITS)
                continue
            if enc == "br":
                brotli_mod = None
                try:
                    import brotli as _brotli  # type: ignore

                    brotli_mod = _brotli
                except ImportError:
                    try:
                        import brotlicffi as _brotli  # type: ignore

                        brotli_mod = _brotli
                    except ImportError:
                        logging.debug("Brotli indisponible pour %s; passage au plan B", url)
                if brotli_mod is not None:
                    try:
                        current = brotli_mod.decompress(current)
                        continue
                    except Exception as exc:  # brotli.error or others
                        logging.debug("Échec de décompression brotli pour %s: %s", url, exc)
                # If brotli lookup fails, stop trying further encodings
                break
            if enc in {"identity", "utf-8"}:
                continue
        except Exception as exc:
            logging.debug("Décompression impossible (%s) pour %s: %s", enc, url, exc)
            break
    return current
    return data


def resolve_download_targets(url: str) -> Sequence[DownloadTarget]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "onlinelibrary.wiley.com" and "/doi/" in parsed.path:
        doi_part = parsed.path.split("/doi/", 1)[1]
        doi = doi_part.split("/", 1)[-1] if "/" in doi_part else doi_part
        doi = doi.strip("/")
        doi = doi.split("?", 1)[0]
        doi = doi.split("#", 1)[0]
        if doi:
            referer = f"https://onlinelibrary.wiley.com/doi/{doi}"
            warmup = referer
            candidates = [
                (f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?downloadFile=1", True),
                (f"https://onlinelibrary.wiley.com/doi/pdf/{doi}", True),
                (f"https://onlinelibrary.wiley.com/doi/epdf/{doi}", True),
                (referer, False),
                (url, None),
            ]
            seen = set()
            targets: list[DownloadTarget] = []
            for candidate, expects_pdf in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    target_referer = referer if candidate != referer else None
                    targets.append(
                        DownloadTarget(candidate, referer=target_referer, warmup=warmup, expects_pdf=expects_pdf)
                    )
            return targets

    expects_pdf = True if pat_pdf.match(url) else False if url.lower().endswith((".html", ".htm")) else None
    targets = [DownloadTarget(url, expects_pdf=expects_pdf)]

    def _canonical_url(parsed_url):
        scheme = parsed_url.scheme or "https"
        base = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        if parsed_url.query:
            base += f"?{parsed_url.query}"
        return base

    if host.endswith(".github.io") or host == "github.io":
        jina_target = f"https://r.jina.ai/{_canonical_url(parsed)}"
        if jina_target != url:
            targets.append(DownloadTarget(jina_target, expects_pdf=False))

    if not host.startswith("r.jina.ai") and expects_pdf is not True and parsed.scheme in {"http", "https", ""}:
        jina_general = f"https://r.jina.ai/{_canonical_url(parsed)}"
        if all(target.url != jina_general for target in targets):
            targets.append(DownloadTarget(jina_general, expects_pdf=False))

    return targets


def lookup_semanticscholar_pdf(session: requests.Session, url: str) -> Optional[str]:
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "www.semanticscholar.org":
        return None

    segments = [segment for segment in parsed.path.split('/') if segment]
    if not segments:
        return None

    paper_id = segments[-1]
    if not paper_id:
        return None

    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=openAccessPdf"
    try:
        response = session.get(
            api_url,
            headers={**BASE_HEADERS, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logging.debug("Semantic Scholar API lookup failed for %s: %s", url, exc)
        return None
    except ValueError as exc:
        logging.debug("Semantic Scholar API JSON parse failed for %s: %s", url, exc)
        return None

    open_access = data.get("openAccessPdf") or {}
    pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
    if pdf_url and isinstance(pdf_url, str):
        return pdf_url

    return None


def open_stream(session: requests.Session, target: DownloadTarget) -> requests.Response:
    last_exc: requests.HTTPError | None = None
    tried: list[dict[str, str]] = []

    prefer_pdf = target.expects_pdf if target.expects_pdf is not None else bool(pat_pdf.match(target.url))

    if prefer_pdf:
        header_variants = [
            make_headers(PDF_ACCEPT, target.referer, target.url),
            make_headers(PDF_ACCEPT, None, target.url),
            make_headers(HTML_ACCEPT, target.referer, target.url),
            make_headers(HTML_ACCEPT, None, target.url),
        ]
    else:
        header_variants = [
            make_headers(HTML_ACCEPT, target.referer, target.url),
            make_headers(HTML_ACCEPT, None, target.url),
            make_headers(PDF_ACCEPT, target.referer, target.url),
            make_headers(PDF_ACCEPT, None, target.url),
        ]

    for headers in header_variants:
        if any(h == headers for h in tried):
            continue
        tried.append(headers)
        response = session.get(target.url, headers=headers, stream=True, timeout=TIMEOUT)
        try:
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            response.close()
            if status not in RETRY_STATUSES:
                raise
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Unexpected download failure")


def download(url: str, destination: Path, fallbacks: Sequence[str] | None = None) -> DownloadOutcome:
    last_error: requests.RequestException | None = None
    prefetched: set[str] = set()

    with requests.Session() as session:
        targets = list(resolve_download_targets(url))
        seen_urls = {target.url for target in targets}

        semanticscholar_pdf = lookup_semanticscholar_pdf(session, url)
        if semanticscholar_pdf:
            if semanticscholar_pdf not in seen_urls:
                targets.insert(0, DownloadTarget(semanticscholar_pdf, referer=url, warmup=None, expects_pdf=True))
                seen_urls.add(semanticscholar_pdf)

        if fallbacks:
            for alt in fallbacks:
                for extra in resolve_download_targets(alt):
                    if extra.url not in seen_urls:
                        targets.append(extra)
                        seen_urls.add(extra.url)

        for target in targets:
            if target.warmup and target.warmup not in prefetched:
                try:
                    session.get(
                        target.warmup,
                        headers=make_headers(HTML_ACCEPT, None, target.warmup),
                        timeout=TIMEOUT,
                        allow_redirects=True,
                    )
                except requests.RequestException as exc:
                    logging.debug("Warmup failed for %s: %s", target.warmup, exc)
                finally:
                    prefetched.add(target.warmup)

            try:
                response = open_stream(session, target)
            except requests.RequestException as exc:
                last_error = exc
                continue

            with response:
                ext = pick_extension(target.url, response.headers.get("Content-Type", ""))
                if ext not in {".pdf", ".html", ".txt", ".json"}:
                    last_error = requests.RequestException(
                        f"Unsupported content type for {target.url}: {response.headers.get('Content-Type')}"
                    )
                    continue

                out_path = destination.with_suffix(ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                raw_bytes = bytearray()
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        raw_bytes.extend(chunk)

                data_bytes = bytes(raw_bytes)
                encoding = response.headers.get("Content-Encoding", "")
                data_bytes = decompress_payload(data_bytes, encoding, target.url)

                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    handle.write(data_bytes)

            tmp_path.replace(out_path)
            if out_path.stat().st_size == 0:
                out_path.unlink(missing_ok=True)
                last_error = requests.RequestException(f"Empty response from {target.url}")
                continue
            return DownloadOutcome(path=out_path, url=target.url)

    if last_error is not None:
        raise last_error

    return DownloadOutcome(path=None, url=url)


def process_table(split: str, path: Path) -> None:
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "Link" not in reader.fieldnames:
            logging.warning("Le fichier %s ne contient pas de colonne 'Link'.", path)
            return

        for index, row in enumerate(reader):
            original_link = row.get("Link")
            dest_base = split_dir / f"{index}"
            link = normalise_link(original_link)
            if not link:
                write_placeholder(dest_base, split, reason=(original_link or None))
                continue
            fallback_urls: list[str] = []
            if isinstance(original_link, str):
                for candidate in extract_urls(original_link):
                    if link and candidate == link:
                        continue
                    fallback_urls.append(candidate)
            already_downloaded = False
            for candidate in (dest_base.with_suffix(".pdf"), dest_base.with_suffix(".html")):
                if candidate.exists():
                    logging.info("[%s] Ignoré, fichier déjà présent: %s", split, candidate.as_posix())
                    already_downloaded = True
                    break

            if not already_downloaded:
                try:
                    outcome = download(link, dest_base, fallbacks=fallback_urls)
                    if outcome.path:
                        if outcome.url != link:
                            logging.info("[%s] %s -> %s (via %s)", split, link, outcome.path.as_posix(), outcome.url)
                        else:
                            logging.info("[%s] %s -> %s", split, link, outcome.path.as_posix())
                    else:
                        logging.info("[%s] %s téléchargé mais ignoré (type non pris en charge).", split, link)
                except requests.RequestException as exc:
                    status = None
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        status = exc.response.status_code

                    if status in {401, 402, 403, 404, 406, 410, 451} or isinstance(exc, requests.ConnectionError):
                        message = f"Failure ({status}) for {link}: {exc}" if status else f"Failure for {link}: {exc}"
                        write_placeholder(dest_base, split, reason=message)
                    else:
                        logging.warning("[%s] Échec du téléchargement de %s: %s", split, link, exc)

def main() -> None:
    for split, path in iter_tables():
        logging.info("Traitement du split %s (%s)", split, path)
        process_table(split, path)


if __name__ == "__main__":
    main()
