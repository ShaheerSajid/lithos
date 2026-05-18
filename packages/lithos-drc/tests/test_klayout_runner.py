"""KLayout runner tests — parser logic and command construction.

The pure functions (lyrdb XML parser, geometry helpers, command builder)
are exercised against synthetic inputs. ``is_available()`` is checked
when ``klayout`` is on PATH; the actual subprocess invocation is not
exercised here (covered by integration tests on a real PDK).
"""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from lithos_core.metadata import PDKMetadata

from lithos_drc import KLayoutDRCRunner
from lithos_drc.klayout_runner import parse_lyrdb


# ── Fixtures ────────────────────────────────────────────────────────────────

def _metadata(tmp_path: Path, deck_text: str = "# stub deck\n") -> PDKMetadata:
    deck = tmp_path / "deck.lydrc"
    deck.write_text(deck_text)
    return PDKMetadata(
        name      = "test_pdk",
        version   = "0.0.1",
        layers    = {"met2": (69, 20)},
        grid      = {"manufacturing_um": 0.005},
        drc_decks = {"klayout": deck},
    )


# ── Runner construction & metadata access ──────────────────────────────────

def test_tool_name_is_klayout(tmp_path: Path):
    r = KLayoutDRCRunner(_metadata(tmp_path))
    assert r.tool_name == "klayout"


def test_deck_resolution_from_metadata(tmp_path: Path):
    md = _metadata(tmp_path)
    r = KLayoutDRCRunner(md)
    assert r._resolve_deck() == md.drc_decks["klayout"]


def test_deck_resolution_env_override(tmp_path: Path, monkeypatch):
    override_deck = tmp_path / "override.lydrc"
    override_deck.write_text("# override\n")
    monkeypatch.setenv("LITHOS_KLAYOUT_DECK", str(override_deck))
    r = KLayoutDRCRunner(_metadata(tmp_path))
    assert r._resolve_deck() == override_deck


def test_deck_missing_raises(tmp_path: Path):
    md = PDKMetadata(
        name="x", version="0", layers={}, grid={},
        drc_decks={"klayout": tmp_path / "nope.drc"},
    )
    r = KLayoutDRCRunner(md)
    with pytest.raises(FileNotFoundError, match="KLayout DRC deck not found"):
        r._resolve_deck()


@pytest.mark.skipif(not shutil.which("klayout"),
                    reason="klayout binary not on PATH")
def test_is_available_when_klayout_installed(tmp_path: Path):
    assert KLayoutDRCRunner(_metadata(tmp_path)).is_available() is True


def test_is_available_false_when_exe_missing(tmp_path: Path):
    r = KLayoutDRCRunner(_metadata(tmp_path), klayout_exe="/nonexistent/klayout")
    assert r.is_available() is False


# ── Command construction ───────────────────────────────────────────────────

def test_build_command_sets_default_knobs(tmp_path: Path, monkeypatch):
    # Make sure no LITHOS_DRC_* env vars leak in from the host.
    for k in ("FEOL", "BEOL", "OFFGRID", "SEAL", "FLOATING_MET",
              "SRAM_EXCLUDE", "THR"):
        monkeypatch.delenv(f"LITHOS_DRC_{k}", raising=False)

    r = KLayoutDRCRunner(_metadata(tmp_path))
    cmd = r._build_command(
        gds_path=tmp_path / "in.gds",
        deck=tmp_path / "deck.lydrc",
        report=tmp_path / "out.lyrdb",
        cell_name="test_cell",
    )
    joined = " ".join(cmd)
    assert "klayout -b -r" in joined
    assert "-rd feol=true"          in joined
    assert "-rd beol=true"          in joined
    assert "-rd offgrid=true"       in joined
    assert "-rd seal=false"         in joined
    assert "-rd floating_met=false" in joined
    assert "-rd sram_exclude=false" in joined
    assert "-rd top_cell=test_cell" in joined
    assert "-rd topcell=test_cell"  in joined        # both names emitted


def test_build_command_env_override_individual_knob(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LITHOS_DRC_SEAL", "true")
    monkeypatch.setenv("LITHOS_DRC_THR", "16")
    r = KLayoutDRCRunner(_metadata(tmp_path))
    cmd = r._build_command(
        gds_path=tmp_path / "in.gds",
        deck=tmp_path / "deck.lydrc",
        report=tmp_path / "out.lyrdb",
        cell_name=None,
    )
    joined = " ".join(cmd)
    assert "-rd seal=true" in joined
    assert "-rd thr=16" in joined
    # No top-cell flag when cell_name is None.
    assert "top_cell" not in joined


# ── lyrdb parser ───────────────────────────────────────────────────────────

_LYRDB_BASIC = textwrap.dedent("""\
<?xml version="1.0" encoding="utf-8"?>
<report-database>
  <categories>
    <category>
      <name>'M2.S.1'</name>
      <description>metal2 minimum spacing</description>
    </category>
    <category>
      <name>'M2.W.1'</name>
      <description>metal2 minimum width</description>
    </category>
  </categories>
  <items>
    <item>
      <category>'M2.S.1'</category>
      <values>
        <value>edge-pair: (0,0;0,100)|(120,0;120,100)</value>
      </values>
    </item>
    <item>
      <category>'M2.W.1'</category>
      <polygon>(100,200;200,200;200,300;100,300)</polygon>
      <values>
        <value>0.13</value>
      </values>
    </item>
  </items>
</report-database>
""")


def test_parse_lyrdb_extracts_rule_and_description(tmp_path: Path):
    path = tmp_path / "report.lyrdb"
    path.write_text(_LYRDB_BASIC)
    v = parse_lyrdb(path)
    assert len(v) == 2
    assert v[0].rule == "M2.S.1"
    assert v[0].description == "metal2 minimum spacing"
    assert v[1].rule == "M2.W.1"
    assert v[1].description == "metal2 minimum width"


def test_parse_lyrdb_edge_pair_centroid_and_distance(tmp_path: Path):
    path = tmp_path / "report.lyrdb"
    path.write_text(_LYRDB_BASIC)
    v = parse_lyrdb(path)
    # First violation: edge-pair (0,0;0,100)|(120,0;120,100).
    # Coordinates are dbu (1nm); convert: 0.120 µm spacing.
    assert v[0].value == pytest.approx(0.120)
    # Centroid is the mean of the four endpoint coords.
    assert v[0].x == pytest.approx(0.060)
    assert v[0].y == pytest.approx(0.050)


def test_parse_lyrdb_polygon_centroid_and_numeric_value(tmp_path: Path):
    path = tmp_path / "report.lyrdb"
    path.write_text(_LYRDB_BASIC)
    v = parse_lyrdb(path)
    # Second violation: polygon (100,200;200,200;200,300;100,300), value 0.13.
    assert v[1].value == pytest.approx(0.13)
    assert v[1].x == pytest.approx(0.150)
    assert v[1].y == pytest.approx(0.250)


def test_parse_lyrdb_empty_returns_empty(tmp_path: Path):
    path = tmp_path / "empty.lyrdb"
    path.write_text(
        "<?xml version='1.0'?><report-database><items></items></report-database>"
    )
    assert parse_lyrdb(path) == []
