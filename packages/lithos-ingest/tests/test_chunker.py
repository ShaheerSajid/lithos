"""Chunker logic — pure-Python tests against synthetic Documents.

These tests don't use pdfplumber. The Document abstraction lets us
exercise the search/extraction logic without any PDF I/O.
"""
from __future__ import annotations

from pathlib import Path

from lithos_core import CategoryConfig, CategoryDef

from lithos_ingest.chunker import (
    Document,
    chunk_for_categories,
    chunk_for_codes,
    extract_window,
    find_code_occurrences,
)


def _doc(text: str, page_chars: int = 200) -> Document:
    """Build a synthetic document split into sections of `page_chars` chars each."""
    page_breaks = list(range(0, max(1, len(text)), page_chars))
    return Document(text=text, page_breaks=page_breaks, path=Path("fake.txt"))


# ── find_code_occurrences ───────────────────────────────────────────────────

def test_find_returns_offsets():
    doc = _doc("Rule M2.S.1 is here. Then M2.S.1 again.")
    offsets = find_code_occurrences(doc, "M2.S.1")
    assert offsets == [5, 26]


def test_word_boundary_excludes_longer_dotted_codes():
    """Searching for M2.S.1 should NOT match inside M2.S.1.W."""
    doc = _doc("See M2.S.1 and also M2.S.1.W.")
    offsets = find_code_occurrences(doc, "M2.S.1")
    # Only the first occurrence; the second is part of M2.S.1.W.
    assert offsets == [4]


def test_word_boundary_off_matches_substrings():
    doc = _doc("See M2.S.1 and also M2.S.1.W.")
    offsets = find_code_occurrences(doc, "M2.S.1", word_boundary=False)
    assert offsets == [4, 20]


def test_no_match():
    doc = _doc("Nothing here.")
    assert find_code_occurrences(doc, "X.1") == []


# ── extract_window ──────────────────────────────────────────────────────────

def test_window_basic():
    doc = _doc("a" * 100 + "MARK" + "b" * 100)
    text, span = extract_window(doc, anchor=100, context_before=10, context_after=20)
    # context_after is the char count *from the anchor*: anchor=100, +20 = 120.
    # text[100:120] = "MARK" (chars 100..103) + 16 b's (chars 104..119) = 20 chars.
    assert span == (90, 120)
    assert text == "a" * 10 + "MARK" + "b" * 16


def test_window_clipped_by_other_anchor_before():
    """If another anchor sits between window-start and our anchor, the
    window starts at the other anchor."""
    doc = _doc("a" * 50 + "X" + "b" * 50 + "Y" + "c" * 50)
    # Anchor at Y (offset 101). Other anchor X at offset 50.
    text, span = extract_window(
        doc, anchor=101, context_before=80, context_after=10, other_anchors=[50],
    )
    assert span == (50, 111)
    assert text.startswith("X")


def test_window_clipped_by_other_anchor_after():
    doc = _doc("a" * 50 + "X" + "b" * 50 + "Y" + "c" * 50)
    # Anchor at X (50). Other anchor Y at 101. Window should not extend past Y.
    text, span = extract_window(
        doc, anchor=50, context_before=10, context_after=80, other_anchors=[101],
    )
    assert span == (40, 101)
    assert text.endswith("b")


def test_window_at_doc_edges():
    doc = _doc("hello world")
    text, span = extract_window(doc, anchor=0, context_before=10, context_after=5)
    assert span == (0, 5)
    text2, span2 = extract_window(doc, anchor=len(doc.text), context_before=5, context_after=10)
    assert span2[1] == len(doc.text)


# ── chunk_for_codes ─────────────────────────────────────────────────────────

_SAMPLE_PDF_TEXT = """\
Chapter 5. Metal Layers.

M2.W.1 Minimum metal2 width.
The minimum drawn width of metal2 polygons shall be 0.14 micrometres.
Use shall be measured perpendicular to drawn edges.

M2.S.1 Minimum metal2 spacing.
The minimum drawn spacing between adjacent metal2 polygons shall be
0.14 micrometres. This prevents litho bridging during fabrication.

M2.S.1.W Wide metal2 spacing.
When both adjacent metal2 polygons exceed 0.30 micrometre width and
their parallel-run-length exceeds 1.0 micrometre, the minimum spacing
shall be 0.30 micrometres.

Chapter 6. Local Interconnect.

LI.E.1 Licon to li1 enclosure.
The minimum enclosure of licon1 by li1 shall be 0.04 micrometres on
all sides.
"""


def test_chunk_returns_one_window_per_code():
    doc = _doc(_SAMPLE_PDF_TEXT)
    chunks = chunk_for_codes(
        doc,
        ["M2.W.1", "M2.S.1", "M2.S.1.W", "LI.E.1"],
        context_before=50,
        context_after=200,
    )
    assert {code: len(cs) for code, cs in chunks.items()} == {
        "M2.W.1": 1, "M2.S.1": 1, "M2.S.1.W": 1, "LI.E.1": 1,
    }


def test_chunk_text_contains_rule_body():
    doc = _doc(_SAMPLE_PDF_TEXT)
    chunks = chunk_for_codes(
        doc, ["M2.S.1"], context_before=50, context_after=300,
    )
    [chunk] = chunks["M2.S.1"]
    assert "Minimum metal2 spacing" in chunk.text
    assert "0.14 micrometres" in chunk.text
    assert "litho bridging" in chunk.text


def test_chunk_bounded_by_neighbouring_codes():
    """M2.S.1's chunk should not bleed into M2.S.1.W's section."""
    doc = _doc(_SAMPLE_PDF_TEXT)
    chunks = chunk_for_codes(
        doc, ["M2.W.1", "M2.S.1", "M2.S.1.W", "LI.E.1"],
        context_before=50, context_after=500,
        bound_to_next_code=True,
    )
    m2_s1_text = chunks["M2.S.1"][0].text
    # M2.S.1.W is in a different rule's chunk now — shouldn't appear in M2.S.1's.
    assert "Wide metal2 spacing" not in m2_s1_text


def test_chunk_unbounded_can_overlap_neighbours():
    """With bounding off, chunks may overlap. Useful for prose-heavy manuals
    where the next code isn't a natural cut point."""
    doc = _doc(_SAMPLE_PDF_TEXT)
    chunks = chunk_for_codes(
        doc, ["M2.W.1"],
        context_before=50, context_after=2000,
        bound_to_next_code=False,
    )
    chunk = chunks["M2.W.1"][0]
    # Without bounding, M2.W.1's window reaches into the next sections.
    assert "Minimum metal2 spacing" in chunk.text


def test_chunk_page_assigned_correctly():
    """Page lookup via Document.page_breaks."""
    text = "PAGE1 content with X.1\f\nPAGE2 content with Y.1"
    page_breaks = [0, text.index("PAGE2")]
    doc = Document(text=text, page_breaks=page_breaks)
    chunks = chunk_for_codes(doc, ["X.1", "Y.1"], context_before=5, context_after=20)
    assert chunks["X.1"][0].page == 1
    assert chunks["Y.1"][0].page == 2


def test_chunk_missing_code_returns_empty_list():
    doc = _doc(_SAMPLE_PDF_TEXT)
    chunks = chunk_for_codes(doc, ["NOPE.99"])
    assert chunks == {"NOPE.99": []}


def test_chunk_section_pattern_filter():
    """section_pattern restricts to windows whose surrounding text matches.

    With bound_to_next_code=True the heading is only reachable from the
    *first* code under it (later codes' windows start at the prior anchor).
    Test with one rule per chapter to keep the scenario clean.
    """
    doc = _doc(_SAMPLE_PDF_TEXT)
    only_metal = chunk_for_codes(
        doc, ["M2.W.1", "LI.E.1"],
        context_before=200, context_after=150,
        section_pattern=r"Chapter 5\.",
    )
    assert len(only_metal["M2.W.1"]) == 1      # window reaches the Chapter 5 heading
    assert only_metal["LI.E.1"] == []          # Chapter 6, never matches


def test_chunk_min_separation_dedupes_close_matches():
    """Same code mentioned twice nearby produces one chunk."""
    text = "M2.S.1 first mention. Some more text. M2.S.1 second nearby mention."
    doc = _doc(text)
    chunks = chunk_for_codes(
        doc, ["M2.S.1"],
        context_before=10, context_after=80,
        min_separation=200,
    )
    assert len(chunks["M2.S.1"]) == 1


# ── chunk_for_categories ────────────────────────────────────────────────────

def test_chunk_for_categories_partitions_codes():
    doc = _doc(_SAMPLE_PDF_TEXT)
    cfg = CategoryConfig(
        categories=[
            CategoryDef(name="metal_low", code_prefixes=["M1.", "M2."], priority=10),
            CategoryDef(name="li",        code_prefixes=["LI."],         priority=20),
        ],
    )
    chunks = chunk_for_categories(
        doc, ["M2.W.1", "M2.S.1", "LI.E.1"], cfg,
        context_before=40, context_after=200,
    )
    assert set(chunks.keys()) == {"M2.W.1", "M2.S.1", "LI.E.1"}
    assert chunks["M2.W.1"][0].section == "metal_low"
    assert chunks["LI.E.1"][0].section == "li"


def test_chunk_for_categories_skips_unclaimed_codes():
    doc = _doc(_SAMPLE_PDF_TEXT)
    cfg = CategoryConfig(
        categories=[CategoryDef(name="metal_low", code_prefixes=["M2."], priority=10)],
    )
    chunks = chunk_for_categories(
        doc, ["M2.W.1", "LI.E.1"], cfg,        # LI.E.1 has no enabled category
        context_before=40, context_after=200,
    )
    assert "M2.W.1" in chunks
    assert "LI.E.1" not in chunks


def test_chunk_for_categories_honours_section_pattern():
    """A category's pdf_section_pattern restricts its codes' search."""
    doc = _doc(_SAMPLE_PDF_TEXT)
    cfg = CategoryConfig(
        categories=[
            CategoryDef(
                name="metal_low",
                code_prefixes=["M2."],
                pdf_section_pattern=r"Chapter 5\.",
                priority=10,
            ),
            CategoryDef(
                name="li",
                code_prefixes=["LI."],
                pdf_section_pattern=r"Chapter 99\.",        # nothing matches
                priority=20,
            ),
        ],
    )
    chunks = chunk_for_categories(
        doc, ["M2.W.1", "LI.E.1"], cfg,
        context_before=60, context_after=200,
    )
    assert len(chunks["M2.W.1"]) == 1
    assert chunks["LI.E.1"] == []     # category enabled but section never matches


# ── code_aliases (PDF placeholder fallback) ─────────────────────────────────

def test_code_aliases_used_when_exact_code_missing():
    """`chunk_for_codes` should fall back to an alias when the exact code
    doesn't appear. This unblocks docs that document a class of layers
    via a placeholder (e.g. ``Mx.W.1`` standing in for ``M3.W.1``)."""
    text = "4.5.14 Metal-2 to Metal-5 (Mx) Rules:\n  Mx.W.1 Minimum width >= 0.28\n  Mx.S.1 Minimum space >= 0.28\n"
    doc = _doc(text)
    chunks = chunk_for_codes(
        doc, ["M3.W.1", "M4.S.1"],
        context_before=20, context_after=100,
        code_aliases={
            "M3.W.1": ["Mx.W.1"],
            "M4.S.1": ["Mx.S.1"],
        },
    )
    assert len(chunks["M3.W.1"]) == 1
    assert "Mx.W.1" in chunks["M3.W.1"][0].text
    assert len(chunks["M4.S.1"]) == 1
    assert "Mx.S.1" in chunks["M4.S.1"][0].text


def test_code_aliases_only_used_when_exact_misses():
    """If the exact code is present, the alias must not override it."""
    text = "M3.W.1 explicitly mentioned. Also Mx.W.1 in a table."
    doc = _doc(text)
    chunks = chunk_for_codes(
        doc, ["M3.W.1"],
        context_before=10, context_after=80,
        code_aliases={"M3.W.1": ["Mx.W.1"]},
    )
    # Should find the explicit form, not chase the alias.
    assert len(chunks["M3.W.1"]) == 1
    assert "M3.W.1 explicitly" in chunks["M3.W.1"][0].text


def test_code_aliases_tries_in_order():
    """Aliases are tried left-to-right; first that hits wins."""
    text = "Section: Mz.W.1 only here.\n"
    doc = _doc(text)
    chunks = chunk_for_codes(
        doc, ["M8.W.1"],
        context_before=10, context_after=80,
        code_aliases={"M8.W.1": ["My.W.1", "Mz.W.1"]},   # My miss, Mz hit
    )
    assert len(chunks["M8.W.1"]) == 1
    assert "Mz.W.1" in chunks["M8.W.1"][0].text
