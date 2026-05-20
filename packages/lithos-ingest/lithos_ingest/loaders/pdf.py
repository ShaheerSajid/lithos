"""lithos_ingest.loaders.pdf — PDF → Document via pdfplumber.

Each PDF page is read with ``page.extract_text()`` and concatenated with a
form-feed sentinel between pages. ``page_breaks`` records the char offset
where each page begins. Empty / unreadable pages contribute an empty
string (and an entry in ``page_breaks``).

Vendor design-rule PDFs routinely stamp a confidentiality watermark over
every page — a vertical or diagonal string like ``"FOUNDRY Confidential
Information <id> ..."`` that pdfplumber emits as individual chars
interleaved with the body text (e.g. ``"(uTnion projection)"`` instead
of ``"(union projection)"``). Those interleaved letters break word
matching in the chunker and corrupt any text the LLM downstream sees.

We solve this at the char level: drop every char whose affine matrix is
not horizontal-upright (i.e. ``matrix[1] != 0 or matrix[2] != 0``)
before reassembling the page text. This excises both 90° vertical
watermarks (``upright=False`` chars) and 45° diagonal stamps
(``upright=True`` but skewed).

In addition, ``_HEADER_FOOTER_PATTERNS`` matches the recurring full-line
header / footer / confidentiality strings shipped by typical vendor
DRMs (document number, version, "Security B - ... Restricted Secret",
"Confidential – Do Not Copy", page-only-id lines, etc.). Matched lines
are dropped, then runs of blank lines are collapsed.

If a PDF refuses to yield text (image-only scans of older nodes), wrap it
with an OCR or VLM preprocessor and write the result through
:func:`lithos_ingest.loaders.text.load_text` instead.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lithos_ingest.chunker import Document


_PAGE_SEP = "\n\f\n"
"""Form-feed sentinel between pages for human readability. The chunker
uses :attr:`Document.page_breaks` for page lookup, not the separator."""


# ── Header / footer line patterns ──────────────────────────────────────────
#
# These are the recurring per-page noise strings vendor DRMs stamp on
# every page. Matched against full *stripped* lines (case-sensitive,
# regex). The patterns intentionally avoid the foundry name so they
# generalise.
_HEADER_FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in (
    r"^Security\s+[A-Z]\s*[-–]\s*.*Restricted\s+Secret\s*$",
    r"^Confidential\s*[-–]\s*Do\s+Not\s+Copy\s*$",
    r"^Technology\s+Document\s+No\.\s*:.+$",
    r"^Document\s+Number\s*:.+$",
    r"^\d+(?:\.\d+)?\s*[A-Z]+\s+CMOS\s+LOGIC\s+Version\s*:.+$",
    r"^\d+\s*nm\s+CMOS\s+LOGIC\s+Version\s*:.+$",
    r"^[A-Z]+\s+Confidential\s+Information\s*$",   # rotated stamp first line
    r"^Information\s*$",
    r"^LLC\s*$",
    r"^\d{2}/\d{2}/\d{4}\s*$",                      # date alone on a line
    r"^\d{6,}\s*$",                                  # 6+ digit document-id alone
    r"^Page\s+\d+\s+of\s+\d+\s*$",
))


def _looks_like_header_or_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _HEADER_FOOTER_PATTERNS:
        if pat.match(s):
            return True
    return False


def _strip_header_footer_lines(text: str) -> str:
    """Drop recurring per-page header / footer / watermark lines.

    Also collapses consecutive blank lines that result from the strip.
    """
    out: list[str] = []
    blank = False
    for line in text.splitlines():
        if _looks_like_header_or_footer(line):
            continue
        if line.strip() == "":
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(line)
    return "\n".join(out)


def _is_horizontal_char(c: dict[str, Any]) -> bool:
    """Return True for a horizontal-upright char (no rotation / skew).

    ``c["matrix"]`` is the PDF affine ``(a, b, c, d, e, f)`` where the
    upright case has ``b == 0`` and ``c == 0``. Anything with non-zero
    skew components is rotated text — typically a watermark stamp.
    """
    m = c.get("matrix")
    if m is None:
        # Older pdfplumber may not expose matrix; fall back to ``upright``.
        return bool(c.get("upright", True))
    # ``m`` is a 6-tuple; allow tiny FP slop.
    b, c2 = m[1], m[2]
    return abs(b) < 1e-3 and abs(c2) < 1e-3


def _page_text_clean(page: Any) -> str:
    """Extract a page's text after dropping rotated / skewed glyphs.

    Uses ``page.filter`` so pdfplumber's normal layout-aware text
    extraction still runs — we just hide the watermark chars from it.
    """
    try:
        filtered = page.filter(_is_horizontal_char)
        return filtered.extract_text() or ""
    except Exception:                            # pragma: no cover — defensive
        # If filtering blows up for any reason, fall back to the raw
        # extraction. Better noisy text than no text.
        return page.extract_text() or ""


def load_pdf(path: Path | str) -> Document:
    """Load a PDF from disk into a :class:`Document`.

    The loader strips rotated watermark chars at the page level (see
    module docstring) and removes recurring header / footer / page-id
    lines before assembling the final text. ``page_breaks`` indexes
    into the *cleaned* text.

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
            raw_text = _page_text_clean(page)
            page_text = _strip_header_footer_lines(raw_text)
            parts.append(page_text)
            cursor += len(page_text)
            if i < n_pages - 1:
                parts.append(_PAGE_SEP)
                cursor += len(_PAGE_SEP)
    return Document(text="".join(parts), page_breaks=page_breaks, path=path)
