"""lithos_ingest.loaders.pdf — PDF → Document via pdfplumber.

Each PDF page is read with ``page.extract_text()`` and concatenated with a
form-feed sentinel between pages. ``page_breaks`` records the char offset
where each page begins. Empty / unreadable pages contribute an empty
string (and an entry in ``page_breaks``).

If a PDF refuses to yield text (image-only scans of older nodes), wrap it
with an OCR or VLM preprocessor and write the result through
:func:`lithos_ingest.loaders.text.load_text` instead.
"""
from __future__ import annotations

from pathlib import Path

from lithos_ingest.chunker import Document


_PAGE_SEP = "\n\f\n"
"""Form-feed sentinel between pages for human readability. The chunker
uses :attr:`Document.page_breaks` for page lookup, not the separator."""


def load_pdf(path: Path | str) -> Document:
    """Load a PDF from disk into a :class:`Document`.

    Raises :class:`ImportError` with an install hint when ``pdfplumber``
    isn't available.
    """
    try:
        import pdfplumber                        # type: ignore[import-not-found]
    except ImportError as exc:                   # pragma: no cover - install hint
        raise ImportError(
            "load_pdf requires pdfplumber. Install with: pip install pdfplumber"
        ) from exc

    path = Path(path)
    parts: list[str] = []
    page_breaks: list[int] = []
    cursor = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_breaks.append(cursor)
            page_text = page.extract_text() or ""
            parts.append(page_text)
            cursor += len(page_text)
            if i < n_pages - 1:
                parts.append(_PAGE_SEP)
                cursor += len(_PAGE_SEP)
    return Document(text="".join(parts), page_breaks=page_breaks, path=path)
