"""Tests for the loader modules: HTML, text/RST/MD, CSV.

PDF loading is exercised separately (it needs ``pdfplumber`` and a real
PDF). The chunker-bypassing CSV loader gets a full end-to-end test here
since it produces ``Chunk`` instances directly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_ingest.chunker import Chunk, Document
from lithos_ingest.loaders import csv_to_chunks, load_html, load_text
from lithos_ingest.loaders.pdf import (
    _is_horizontal_char,
    _strip_header_footer_lines,
)


# ── HTML ────────────────────────────────────────────────────────────────────

_HTML = """\
<html>
<head><title>SkyWater Sky130 Design Rules</title>
<style>body{color:black}</style></head>
<body>
<h1>Chapter 5. Metal Layers</h1>
<p>The metal layers below the redistribution layer carry signal nets.</p>
<h2>M2.W.1 Minimum metal2 width</h2>
<p>Minimum drawn width of metal2 shall be 0.14 micrometres.</p>
<h2>M2.S.1 Minimum metal2 spacing</h2>
<p>Minimum drawn spacing between adjacent metal2 polygons shall be 0.14 micrometres.</p>
<h1>Chapter 6. Local Interconnect</h1>
<p>The li1 layer connects the front-end to the metal stack.</p>
<h2>LI.E.1 Licon to li1 enclosure</h2>
<p>Minimum enclosure of licon1 by li1 shall be 0.04 micrometres.</p>
</body>
</html>
"""


def test_html_extracts_visible_text(tmp_path: Path):
    p = tmp_path / "rules.html"
    p.write_text(_HTML)
    doc = load_html(p)
    assert "Chapter 5. Metal Layers" in doc.text
    assert "M2.S.1" in doc.text
    assert "0.14 micrometres" in doc.text


def test_html_strips_script_and_style(tmp_path: Path):
    p = tmp_path / "rules.html"
    p.write_text(_HTML)
    doc = load_html(p)
    # The <style> content should not appear in extracted text.
    assert "color:black" not in doc.text
    # The <title> content sits in <head>, which is also skipped.
    assert "SkyWater Sky130 Design Rules" not in doc.text


def test_html_section_breaks_at_h1_h2(tmp_path: Path):
    p = tmp_path / "rules.html"
    p.write_text(_HTML)
    doc = load_html(p)
    # Five heading occurrences (1 × h1 for chapter 5, 2 × h2 in it,
    # 1 × h1 for chapter 6, 1 × h2 in it).
    assert len(doc.page_breaks) == 5
    # Section indices increase with offset.
    assert doc.page_breaks == sorted(doc.page_breaks)


def test_html_path_round_trip(tmp_path: Path):
    p = tmp_path / "x.html"
    p.write_text("<p>hello</p>")
    doc = load_html(p)
    assert doc.path == p
    assert "hello" in doc.text


# ── Text / RST / Markdown ───────────────────────────────────────────────────

_RST = """\
Chapter 5. Metal Layers
========================

The metal layers below the RDL carry signal nets.

M2.W.1 Minimum metal2 width
----------------------------

Minimum drawn width of metal2 shall be 0.14 micrometres.

M2.S.1 Minimum metal2 spacing
------------------------------

Minimum drawn spacing between adjacent metal2 polygons shall be 0.14 µm.
"""

_MARKDOWN = """\
# Chapter 5. Metal Layers

The metal layers below the RDL carry signal nets.

## M2.W.1 Minimum metal2 width

Minimum drawn width of metal2 shall be 0.14 micrometres.

## M2.S.1 Minimum metal2 spacing

Minimum drawn spacing between adjacent metal2 polygons shall be 0.14 µm.
"""


def test_load_text_rst_headings(tmp_path: Path):
    p = tmp_path / "rules.rst"
    p.write_text(_RST)
    doc = load_text(p)
    assert "Minimum drawn width" in doc.text
    # The three RST headings produce 3 section offsets.
    assert len(doc.page_breaks) == 3


def test_load_text_markdown_headings(tmp_path: Path):
    p = tmp_path / "rules.md"
    p.write_text(_MARKDOWN)
    doc = load_text(p)
    assert "M2.S.1" in doc.text
    # Three ATX headings → three section offsets.
    assert len(doc.page_breaks) == 3


def test_load_text_plain_no_headings(tmp_path: Path):
    p = tmp_path / "rules.txt"
    p.write_text("just text, no headings, but with a rule M2.S.1 in it.")
    doc = load_text(p)
    # No headings → single section starting at 0.
    assert doc.page_breaks == [0]


# ── CSV — bypasses the chunker ──────────────────────────────────────────────

def test_csv_to_chunks_basic(tmp_path: Path):
    csv_text = (
        "rule_code,layer,metric,value_um,description\n"
        "M2.W.1,met2,width,0.14,metal2 minimum width\n"
        "M2.S.1,met2,space,0.14,metal2 minimum spacing\n"
        "LI.E.1,licon1,enclosure,0.04,licon to li1 enclosure\n"
    )
    p = tmp_path / "rules.csv"
    p.write_text(csv_text)
    chunks = csv_to_chunks(p, code_column="rule_code")
    assert set(chunks) == {"M2.W.1", "M2.S.1", "LI.E.1"}
    [m2_s1] = chunks["M2.S.1"]
    assert isinstance(m2_s1, Chunk)
    assert "layer: met2" in m2_s1.text
    assert "value_um: 0.14" in m2_s1.text
    assert "metric: space" in m2_s1.text
    assert m2_s1.section == "csv"


def test_csv_to_chunks_skips_empty_codes(tmp_path: Path):
    csv_text = (
        "rule_code,layer\n"
        "M2.S.1,met2\n"
        ",met3\n"          # blank code — skip
        "   ,met4\n"        # whitespace-only — skip
        "P.1,poly\n"
    )
    p = tmp_path / "rules.csv"
    p.write_text(csv_text)
    chunks = csv_to_chunks(p, code_column="rule_code")
    assert set(chunks) == {"M2.S.1", "P.1"}


def test_csv_to_chunks_column_whitelist(tmp_path: Path):
    csv_text = (
        "rule_code,layer,metric,internal_note\n"
        "M2.S.1,met2,space,confidential\n"
    )
    p = tmp_path / "rules.csv"
    p.write_text(csv_text)
    chunks = csv_to_chunks(
        p, code_column="rule_code", columns={"layer", "metric"},
    )
    [chunk] = chunks["M2.S.1"]
    assert "layer: met2" in chunk.text
    assert "metric: space" in chunk.text
    assert "internal_note" not in chunk.text


def test_csv_to_chunks_missing_code_column(tmp_path: Path):
    csv_text = "name,value\nfoo,bar\n"
    p = tmp_path / "wrong.csv"
    p.write_text(csv_text)
    with pytest.raises(ValueError, match="rule_code"):
        csv_to_chunks(p, code_column="rule_code")


# ── PDF cleanup helpers ─────────────────────────────────────────────────────
#
# Unit-test the pure helpers used by load_pdf without needing pdfplumber
# (the real PDF call path is exercised by integration runs).

class TestStripHeaderFooterLines:
    def test_strips_restricted_secret_header(self):
        text = "Security B - Some Vendor Restricted Secret\nM1.W.1 width >= 0.23\n"
        out = _strip_header_footer_lines(text)
        assert "Restricted Secret" not in out
        assert "M1.W.1" in out

    def test_strips_confidential_do_not_copy(self):
        text = "Confidential – Do Not Copy\nrule body\n"
        out = _strip_header_footer_lines(text)
        assert "Do Not Copy" not in out

    def test_strips_technology_document_number(self):
        text = "Technology Document No. : T-XXX-XX-DR-001\nbody\n"
        out = _strip_header_footer_lines(text)
        assert "Document No" not in out
        assert "body" in out

    def test_strips_version_line(self):
        for line in (
            "0.18 UM CMOS LOGIC Version : 2.15_2",
            "65 nm CMOS LOGIC Version : 2.6",
        ):
            out = _strip_header_footer_lines(line + "\nbody\n")
            assert "Version" not in out
            assert "body" in out

    def test_strips_rotated_stamp_residue(self):
        """Single-token leftovers from the vertical confidentiality stamp.

        The stamp explodes into single-token lines once char-rotation
        is filtered out; the strip catches the recurring noise tokens
        (``Information``, ``LLC``, dates, multi-digit IDs).
        """
        text = "Information\nLLC\n01/30/2021\n2560523\nbody\n"
        out = _strip_header_footer_lines(text)
        assert "body" in out
        for token in ("Information", "LLC", "01/30/2021", "2560523"):
            assert token not in out

    def test_collapses_blank_runs(self):
        out = _strip_header_footer_lines("a\n\n\n\nb\n")
        assert out == "a\n\nb"

    def test_keeps_real_lines_intact(self):
        text = "M1.W.1 Minimum width of M1 region A >= 0.23\n"
        assert _strip_header_footer_lines(text).strip() == text.strip()


class TestIsHorizontalChar:
    def test_horizontal_passes(self):
        # Upright body text: matrix = (size, 0, 0, size, e, f).
        assert _is_horizontal_char({"matrix": (10.0, 0.0, 0.0, 10.0, 100.0, 200.0)}) is True

    def test_ninety_degree_filtered(self):
        # Vertical watermark: matrix = (0, -1, 1, 0, ...).
        assert _is_horizontal_char({"matrix": (0.0, -1.0, 1.0, 0.0, 0.0, 0.0)}) is False

    def test_forty_five_degree_filtered(self):
        # Diagonal stamp: matrix has 0.707 in both b and c.
        m = (0.707, -0.707, 0.707, 0.707, 0.0, 0.0)
        assert _is_horizontal_char({"matrix": m}) is False

    def test_missing_matrix_falls_back_to_upright(self):
        # Old pdfplumber: no matrix; trust `upright`.
        assert _is_horizontal_char({"upright": True}) is True
        assert _is_horizontal_char({"upright": False}) is False
