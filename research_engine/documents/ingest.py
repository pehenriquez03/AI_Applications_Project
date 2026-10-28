"""Stage 1: read a PDF into page-numbered text.

Writes paper_meta.json and paper_text.json into the run folder. Pages are
1-based. Whitespace is normalized per page while line breaks are kept, so
quotes copied by the model can be matched against the page text later.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Optional

from research_engine.contracts import RunFiles

MIN_MEDIAN_CHARS_PER_PAGE = 200


class ScannedPDFError(ValueError):
    """The PDF has (almost) no extractable text; it is probably a scan."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_page(text: str) -> str:
    """Collapse runs of spaces/tabs, strip each line, drop >1 consecutive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace(" ", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _guess_title(first_page: str, pdf_title: Optional[str]) -> str:
    t = (pdf_title or "").strip()
    bad_meta = re.compile(r"(\\|/|\.(tex|wpd|doc|docx|pdf|dvi|ps)$|^pii\b|^microsoft|^untitled)", re.I)
    if len(t) > 8 and not bad_meta.search(t):
        return t
    skip = re.compile(r"(pii:|working paper|nber|http|www\.|journal|vol\.|volume|issn|doi|©|copyright|^\d+$|"
                      r"national bureau|cambridge|massachusetts|draft|^page)", re.I)
    for ln in first_page.split("\n"):
        s = ln.strip()
        if len(s) < 12 or skip.search(s):
            continue
        if sum(ch.isalpha() for ch in s) < 0.6 * len(s):
            continue
        return s
    return ""


def ingest(pdf_path, run_dir, selected_experiment: str | None = None) -> dict:
    """Extract page text from `pdf_path` into `run_dir`; return paper_meta."""
    from pypdf import PdfReader

    pdf_path = Path(pdf_path).resolve()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # a single broken page must not sink the paper
            raw = ""
        pages.append({"page": i, "text": normalize_page(raw)})

    n_pages = len(pages)
    if n_pages == 0:
        raise ScannedPDFError(f"{pdf_path.name}: the PDF has no pages")
    chars = [len(p["text"]) for p in pages]
    median_chars = statistics.median(chars)
    if median_chars < MIN_MEDIAN_CHARS_PER_PAGE:
        raise ScannedPDFError(
            f"{pdf_path.name}: median {median_chars:.0f} extractable characters per page "
            f"(< {MIN_MEDIAN_CHARS_PER_PAGE}); this looks like a scanned PDF. Run OCR first "
            f"or supply a text-based PDF."
        )

    pdf_title = None
    try:
        if reader.metadata is not None:
            pdf_title = reader.metadata.title
    except Exception:
        pdf_title = None

    meta = {
        "sha256": _sha256(pdf_path),
        "source_path": str(pdf_path),
        "file_name": pdf_path.name,
        "n_pages": n_pages,
        "title_guess": _guess_title(pages[0]["text"], pdf_title),
        "selected_experiment": selected_experiment,
        "median_chars_per_page": median_chars,
        "total_chars": sum(chars),
    }
    (run_dir / RunFiles.paper_text).write_text(json.dumps(pages, indent=1, ensure_ascii=False))
    (run_dir / RunFiles.paper_meta).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def load_paper_text(run_dir) -> list[dict]:
    return json.loads((Path(run_dir) / RunFiles.paper_text).read_text())


def page_tagged_text(paper_text: list[dict]) -> str:
    """Join pages as <page n="k">...</page> blocks for the prompts."""
    return "\n".join(f'<page n="{p["page"]}">\n{p["text"]}\n</page>' for p in paper_text)
