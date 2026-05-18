"""lithos_ingest.chunker — code-anchored chunking, format-agnostic.

The chunker takes a list of rule codes (typically harvested from the deck
parser) and locates each one inside a foundry rule manual, returning a
context window per occurrence. This is the input the LLM extractor reads.

Why code-anchored, not heading-anchored: foundry rule docs vary in
structure (PDF, HTML, RST, Markdown, multi-column prose, tables) and
rarely share a consistent heading hierarchy across PDKs. But the rule
**code** (``M2.S.1``, ``poly.1a``, ``met2.6``) almost always appears as
text somewhere near its definition — the code itself is the most reliable
anchor. Heading-based chunking demands a per-PDK chunker; code-anchored
chunking generalises across both PDKs and source formats.

Layer separation
----------------
* :class:`Document` is a pure-Python value type (full text + soft "page"
  offsets). It has no I/O dependency. All chunking logic operates on it.
* Loaders for each format live in :mod:`lithos_ingest.loaders`
  (``load_pdf``, ``load_html``, ``load_text``, …). Each one turns a file
  on disk into a :class:`Document`. CSV-style structured rule tables
  bypass the chunker entirely — see :mod:`lithos_ingest.loaders.csv`.

Category scoping is supported through ``pdf_section_pattern`` on
:class:`lithos_core.categories.CategoryDef`: when a category declares one,
:func:`chunk_for_categories` restricts that category's code searches to
text windows whose surrounding region matches the pattern. The name
``pdf_section_pattern`` is historical — the pattern is applied to any
:class:`Document`'s text regardless of source format.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from lithos_core.categories import CategoryConfig


# ── Value types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Document:
    """A source document reduced to a single text string + section offsets.

    ``page_breaks`` is a list of char offsets at which "pages" (or other
    natural sections — chapters in HTML, top-level headings in RST,
    document breaks in concatenated text) begin. Names are historical —
    *page* is the PDF idiom but the same offset list works for any format.
    1-indexed via :meth:`page_at`.
    """
    text:        str
    page_breaks: list[int]
    path:        Optional[Path] = None

    def page_at(self, offset: int) -> int:
        """Return the 1-indexed section (page) that ``offset`` falls on."""
        if not self.page_breaks:
            return 1
        return max(1, bisect.bisect_right(self.page_breaks, offset))


@dataclass(frozen=True)
class Chunk:
    """One context window extracted around an anchor occurrence."""
    code:    str
    text:    str
    page:    int
    span:    tuple[int, int]     # (start, end) char offsets in Document.text
    anchor:  int                 # offset of the code itself within the doc
    section: Optional[str] = None


# ── Code-anchored search ─────────────────────────────────────────────────────

_CODE_WORD_BOUNDARY_CHARS = r"[A-Za-z0-9._]"
"""Characters considered part of a rule-code token for word-boundary purposes.

We can't use Python's ``\\b`` because rule codes contain dots, which ``\\b``
treats as boundaries — meaning ``\\bM2.S.1\\b`` matches the ``M2`` prefix of
``M2.S.1.W``. Instead we use negative lookbehind/lookahead on this character
class to ensure the match isn't part of a longer dotted identifier."""


def _code_pattern(code: str, word_boundary: bool) -> re.Pattern[str]:
    escaped = re.escape(code)
    if word_boundary:
        escaped = (
            f"(?<!{_CODE_WORD_BOUNDARY_CHARS}){escaped}"
            f"(?!{_CODE_WORD_BOUNDARY_CHARS})"
        )
    return re.compile(escaped)


def find_code_occurrences(
    doc: Document,
    code: str,
    *,
    word_boundary: bool = True,
) -> list[int]:
    """Return every character offset where ``code`` appears in ``doc``.

    With ``word_boundary=True`` (default), ``M2.S.1`` won't match inside
    ``M2.S.1.W``. Disable for prefix-style searches.
    """
    return [m.start() for m in _code_pattern(code, word_boundary).finditer(doc.text)]


def extract_window(
    doc: Document,
    anchor: int,
    *,
    context_before: int = 200,
    context_after:  int = 800,
    other_anchors:  Iterable[int] = (),
) -> tuple[str, tuple[int, int]]:
    """Extract a text window around ``anchor``.

    The window won't cross any offset in ``other_anchors`` — typically the
    occurrences of *other* rule codes in the doc, so each rule's chunk
    naturally ends where the next rule's chunk begins.

    Returns ``(text, (start, end))``.
    """
    raw_start = max(0, anchor - context_before)
    raw_end   = min(len(doc.text), anchor + context_after)

    others_before = [o for o in other_anchors if raw_start <= o < anchor]
    if others_before:
        raw_start = max(others_before)

    others_after = [o for o in other_anchors if anchor < o <= raw_end]
    if others_after:
        raw_end = min(others_after)

    return doc.text[raw_start:raw_end], (raw_start, raw_end)


# ── Top-level chunking entry points ──────────────────────────────────────────

def chunk_for_codes(
    doc: Document,
    codes: Iterable[str],
    *,
    context_before:     int  = 200,
    context_after:      int  = 800,
    bound_to_next_code: bool = True,
    word_boundary:      bool = True,
    min_separation:     int  = 50,
    section_pattern:    Optional[str] = None,
) -> dict[str, list[Chunk]]:
    """For each code in ``codes``, return the PDF chunks where it appears.

    Parameters
    ----------
    context_before, context_after
        Char counts before / after the anchor to include in the window.
    bound_to_next_code
        When True, a chunk for code A won't extend past the position of any
        other code's occurrence — giving each rule its own slice of the doc.
    word_boundary
        When True, a search for ``M2.S.1`` won't match inside ``M2.S.1.W``.
    min_separation
        Two anchors of the same code closer than this distance are merged
        into one chunk (deduplicates repeated mentions in the same paragraph).
    section_pattern
        Optional regex; when set, only matches whose extracted window
        contains this pattern are retained. Used by
        :func:`chunk_for_categories` to scope each category's searches.
    """
    code_list = list(codes)

    # First pass: collect every (code, offset) anchor so we can cross-bound.
    by_code: dict[str, list[int]] = {}
    all_offsets: list[int] = []
    for code in code_list:
        offsets = find_code_occurrences(doc, code, word_boundary=word_boundary)
        by_code[code] = offsets
        all_offsets.extend(offsets)
    all_offsets.sort()

    section_re = re.compile(section_pattern) if section_pattern else None

    out: dict[str, list[Chunk]] = {code: [] for code in code_list}
    for code, offsets in by_code.items():
        for offset in offsets:
            others = (
                [o for o in all_offsets if o != offset]
                if bound_to_next_code else []
            )
            text, span = extract_window(
                doc, offset,
                context_before=context_before,
                context_after =context_after,
                other_anchors =others,
            )
            if section_re is not None and not section_re.search(text):
                continue
            prior = out[code]
            if prior and abs(prior[-1].anchor - offset) < min_separation:
                continue
            out[code].append(Chunk(
                code   = code,
                text   = text,
                page   = doc.page_at(offset),
                span   = span,
                anchor = offset,
            ))
    return out


def chunk_for_categories(
    doc: Document,
    codes: Iterable[str],
    categories: CategoryConfig,
    *,
    context_before:     int  = 200,
    context_after:      int  = 800,
    bound_to_next_code: bool = True,
    word_boundary:      bool = True,
    min_separation:     int  = 50,
) -> dict[str, list[Chunk]]:
    """Category-scoped chunking.

    Codes are partitioned by their resolved category, and each partition is
    chunked independently. A category with a ``pdf_section_pattern`` only
    yields chunks whose window matches that pattern; categories without one
    pull from the entire document. Codes claimed by disabled categories
    (or none at all) are skipped entirely — efficiently scoping the
    ingestion to what the user actually wants this run.
    """
    by_category: dict[str, list[str]] = {}
    for code in codes:
        cat = categories.match(code)
        if cat is None:
            continue
        by_category.setdefault(cat.name, []).append(code)

    out: dict[str, list[Chunk]] = {}
    for cat_name, code_subset in by_category.items():
        cat = categories.by_name(cat_name)
        section_pat = cat.pdf_section_pattern if cat is not None else None
        chunks = chunk_for_codes(
            doc, code_subset,
            context_before     = context_before,
            context_after      = context_after,
            bound_to_next_code = bound_to_next_code,
            word_boundary      = word_boundary,
            min_separation     = min_separation,
            section_pattern    = section_pat,
        )
        for code, code_chunks in chunks.items():
            # Annotate each chunk with the category's section name (informational).
            tagged = [
                Chunk(**{**c.__dict__, "section": cat_name}) for c in code_chunks
            ]
            out[code] = tagged
    return out
