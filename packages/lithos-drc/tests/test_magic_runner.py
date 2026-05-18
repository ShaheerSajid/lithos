"""Magic runner tests — parser, rcfile discovery, GDS flattening."""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from lithos_core.metadata import PDKMetadata

from lithos_drc import MagicDRCRunner
from lithos_drc.magic_runner import (
    _flatten_gds,
    _guess_layer,
    _guess_rule_id,
    parse_magic_output,
)


def _metadata(tmp_path: Path, *, with_rcfile: bool = True) -> PDKMetadata:
    tech = tmp_path / "test_pdk.tech"
    tech.write_text("# stub tech\n")
    if with_rcfile:
        (tmp_path / "test_pdk.magicrc").write_text("# stub rcfile\n")
    return PDKMetadata(
        name      = "test_pdk",
        version   = "0.0.1",
        layers    = {"met2": (69, 20)},
        grid      = {},
        drc_decks = {"magic": tech},
    )


# ── Construction / discovery ───────────────────────────────────────────────

def test_tool_name_is_magic(tmp_path: Path):
    r = MagicDRCRunner(_metadata(tmp_path))
    assert r.tool_name == "magic"


def test_find_rcfile_via_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LITHOS_MAGIC_RCFILE", raising=False)
    monkeypatch.delenv("PDK_ROOT", raising=False)
    r = MagicDRCRunner(_metadata(tmp_path))
    rc = r._find_rcfile()
    assert rc == tmp_path / "test_pdk.magicrc"


def test_find_rcfile_env_override(tmp_path: Path, monkeypatch):
    override = tmp_path / "custom.magicrc"
    override.write_text("# override\n")
    monkeypatch.setenv("LITHOS_MAGIC_RCFILE", str(override))
    r = MagicDRCRunner(_metadata(tmp_path, with_rcfile=False))
    assert r._find_rcfile() == override


def test_find_rcfile_missing_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LITHOS_MAGIC_RCFILE", raising=False)
    monkeypatch.setenv("PDK_ROOT", str(tmp_path / "nonexistent_pdk_root"))
    r = MagicDRCRunner(_metadata(tmp_path, with_rcfile=False))
    with pytest.raises(FileNotFoundError, match="Cannot find Magic"):
        r._find_rcfile()


@pytest.mark.skipif(not shutil.which("magic"),
                    reason="magic binary not on PATH")
def test_is_available_when_magic_installed(tmp_path: Path):
    assert MagicDRCRunner(_metadata(tmp_path)).is_available() is True


def test_is_available_false_when_exe_missing(tmp_path: Path):
    r = MagicDRCRunner(_metadata(tmp_path), magic_exe="/nonexistent/magic")
    assert r.is_available() is False


# ── Heuristics ─────────────────────────────────────────────────────────────

def test_guess_layer_matches_logical_names():
    assert _guess_layer("Poly width < 0.15 (poly.1)") == "poly"
    assert _guess_layer("Met2 spacing < 0.14 (met2.2)") == "met2"
    assert _guess_layer("P-diff distance to N-tap (difftap.2)") == "diff"
    assert _guess_layer("unrelated text") == ""


def test_guess_rule_id_from_trailing_paren():
    assert _guess_rule_id("Poly width < 0.15 (poly.1)") == "poly.1"
    assert _guess_rule_id("P-diff distance to N-tap (difftap.2)") == "difftap.2"


def test_guess_rule_id_fallback():
    assert _guess_rule_id("met2 width 1") == "met2"     # letter+digit fallback


# ── Magic output parser ────────────────────────────────────────────────────

_MAGIC_OUTPUT = textwrap.dedent("""\
DRC errors for cell test_cell
--------------------------------------------

Poly width < 0.15um (poly.1)
0.000 0.000 0.100 0.020
0.500 0.500 0.600 0.520

Met2 spacing < 0.14um (met2.2)
1.000 1.000 1.140 1.020
""")


def test_parse_magic_output_basic():
    v = parse_magic_output(_MAGIC_OUTPUT)
    assert len(v) == 3
    # First two are poly.1.
    assert v[0].rule == "poly.1"
    assert v[0].layer == "poly"
    assert v[0].description.startswith("Poly width")
    # Centroid of first rect (0,0)-(0.1,0.02): (0.05, 0.01).
    assert v[0].x == pytest.approx(0.050)
    assert v[0].y == pytest.approx(0.010)
    # Third is met2.2.
    assert v[2].rule == "met2.2"
    assert v[2].layer == "met2"


def test_parse_magic_output_skips_headers_and_blanks():
    text = "DRC errors for cell foo\n---\n\n"
    assert parse_magic_output(text) == []


def test_parse_magic_output_no_coords_no_violation():
    """A rule description with no coord lines yields no violations."""
    text = "Some rule (foo.1)\n"
    assert parse_magic_output(text) == []


# ── GDS flatten helper (needs gdstk) ───────────────────────────────────────

def test_flatten_gds_round_trip(tmp_path: Path):
    """_flatten_gds picks the named cell and writes a clean single-cell GDS."""
    import gdstk

    src = tmp_path / "src.gds"
    cell = gdstk.Cell("my_cell")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=68, datatype=20))
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(str(src))

    dst = tmp_path / "flat.gds"
    name = _flatten_gds(src, dst, "my_cell")
    assert name == "my_cell"
    assert dst.exists()

    # Verify the written file has exactly one cell with the same polygons.
    lib2 = gdstk.read_gds(str(dst))
    assert len(lib2.cells) == 1
    assert lib2.cells[0].name == "my_cell"
    assert len(lib2.cells[0].polygons) == 1


def test_flatten_gds_sanitises_name(tmp_path: Path):
    """Magic can't load cells with $/dots etc.; flatten sanitises."""
    import gdstk

    src = tmp_path / "src.gds"
    cell = gdstk.Cell("bad.name$here")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=68, datatype=20))
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(str(src))

    dst = tmp_path / "flat.gds"
    name = _flatten_gds(src, dst, "bad.name$here")
    assert name == "bad_name_here"


# ── Registry self-registration ─────────────────────────────────────────────

def test_both_backends_self_register():
    """Importing lithos_drc registers klayout + magic by side effect."""
    import lithos_drc
    available = lithos_drc.registry.available()
    assert "klayout" in available
    assert "magic"   in available
