"""
ASTRA INTEL — Document Loader
===============================
PyMuPDF-based PDF ingestion with page-level text extraction,
scanned/empty PDF detection, and recursive text chunking
that preserves source + page metadata on every chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import BinaryIO, List, Optional, Union

import pymupdf  # PyMuPDF (modern import, replaces deprecated 'fitz')


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Raw text content of a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text.strip())


@dataclass
class DocumentChunk:
    """A text chunk with preserved metadata."""
    chunk_id: str
    text: str
    source: str       # original filename
    page: int         # 1-indexed page number
    chunk_index: int  # position within that page


@dataclass
class LoadedDocument:
    """Aggregated result of a successful document load."""
    doc_name: str
    total_pages: int
    pages: List[PageContent]
    chunks: List[DocumentChunk]
    word_count: int


# ── Exceptions ──────────────────────────────────────────────────────────────

class DocumentLoadError(Exception):
    """Raised when the document cannot be loaded or is unreadable."""


class ScannedOrEmptyPDFError(DocumentLoadError):
    """Raised when the PDF is scanned/image-only or near-empty."""


# ── Recursive Text Splitter ────────────────────────────────────────────────

def _recursive_text_split(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split *text* into chunks of roughly *chunk_size* characters with
    *chunk_overlap* overlap.  Tries paragraph -> line -> sentence -> word
    boundaries before falling back to a hard character cut.
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def _parts(txt: str, sep: str) -> List[str]:
        if sep == "":
            return list(txt)
        return [s for s in re.split(re.escape(sep), txt) if s.strip()]

    def _split(txt: str, seps: List[str]) -> List[str]:
        if not txt.strip():
            return []
        if len(txt) <= chunk_size:
            return [txt.strip()]
        if not seps:
            return [
                txt[i : i + chunk_size]
                for i in range(0, len(txt), chunk_size - chunk_overlap)
            ]

        sep = seps[0]
        parts = _parts(txt, sep)
        result: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            plen = len(part)
            if current_len + plen > chunk_size and current:
                joined = (" " if sep == " " else sep).join(current).strip()
                if joined:
                    result.append(joined)
                # keep overlap
                overlap: List[str] = []
                olen = 0
                for p in reversed(current):
                    if olen + len(p) <= chunk_overlap:
                        overlap.insert(0, p)
                        olen += len(p)
                    else:
                        break
                current = overlap
                current_len = olen
            current.append(part)
            current_len += plen

        if current:
            joined = (" " if sep == " " else sep).join(current).strip()
            if joined:
                result.append(joined)

        final: List[str] = []
        for chunk in result:
            if len(chunk) > chunk_size:
                final.extend(_split(chunk, seps[1:]))
            elif chunk.strip():
                final.append(chunk.strip())
        return final

    return _split(text, separators)


# ── Core Loader ─────────────────────────────────────────────────────────────

def load_pdf(
    file_input: Union[bytes, BinaryIO, str],
    doc_name: str = "document.pdf",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    min_word_threshold: int = 50,
) -> LoadedDocument:
    """
    Load a PDF, extract text page-by-page, and chunk it with metadata.

    Raises
    ------
    DocumentLoadError        – corrupt / password-protected / zero-page PDF.
    ScannedOrEmptyPDFError   – fewer than *min_word_threshold* words extracted.
    """
    # 1. Open
    try:
        if isinstance(file_input, (bytes, bytearray)):
            pdf = pymupdf.open(stream=file_input, filetype="pdf")
        elif isinstance(file_input, str):
            pdf = pymupdf.open(file_input)
        else:
            raw = file_input.read()
            pdf = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise DocumentLoadError(
            f"Failed to open PDF '{doc_name}'. "
            f"The file may be corrupt or password-protected. Detail: {exc}"
        ) from exc

    total_pages = pdf.page_count
    if total_pages == 0:
        pdf.close()
        raise DocumentLoadError(f"'{doc_name}' contains zero pages.")

    # 2. Extract per page
    pages: List[PageContent] = []
    for idx in range(total_pages):
        raw_text = pdf[idx].get_text("text")
        pages.append(PageContent(page_number=idx + 1, text=_clean(raw_text)))
    pdf.close()

    # 3. Scanned / empty guard
    all_text = " ".join(p.text for p in pages)
    word_count = len(all_text.split())
    if word_count < min_word_threshold:
        raise ScannedOrEmptyPDFError(
            f"Insufficient extractable text in '{doc_name}': "
            f"only {word_count} word(s) across {total_pages} page(s). "
            "The document may be scanned/image-only or largely empty. "
            "Please provide a text-layer PDF."
        )

    # 4. Chunk each page
    chunks: List[DocumentChunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = _recursive_text_split(page.text, chunk_size, chunk_overlap)
        for ci, ct in enumerate(page_chunks):
            if ct.strip():
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_name}::p{page.page_number}::c{ci}",
                    text=ct,
                    source=doc_name,
                    page=page.page_number,
                    chunk_index=ci,
                ))

    if not chunks:
        raise DocumentLoadError(
            f"No usable text chunks from '{doc_name}'. "
            "The document may contain only non-text elements."
        )

    return LoadedDocument(
        doc_name=doc_name,
        total_pages=total_pages,
        pages=pages,
        chunks=chunks,
        word_count=word_count,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _clean(raw: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(l.rstrip() for l in cleaned.splitlines())
    return cleaned.strip()


def get_page_text(doc: LoadedDocument, page_number: int) -> Optional[str]:
    for p in doc.pages:
        if p.page_number == page_number:
            return p.text
    return None


def get_doc_statistics(doc: LoadedDocument) -> dict:
    non_empty = sum(1 for p in doc.pages if p.text.strip())
    avg = (
        int(sum(len(c.text) for c in doc.chunks) / len(doc.chunks))
        if doc.chunks else 0
    )
    return {
        "doc_name": doc.doc_name,
        "total_pages": doc.total_pages,
        "non_empty_pages": non_empty,
        "total_chunks": len(doc.chunks),
        "total_words": doc.word_count,
        "avg_chunk_len": avg,
    }
