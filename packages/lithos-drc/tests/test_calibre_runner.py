"""Calibre runner tests — parser + docker command construction.

The pure functions (ASCII RVDB parser, command builder, deck
resolution, top-cell detection) are exercised against synthetic
inputs. The actual ``docker run`` is not exercised (covered by
integration tests against a real container + real deck).
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lithos_core.metadata import PDKMetadata

from lithos_drc import CalibreDRCRunner
from lithos_drc.calibre_runner import (
    _detect_top_cell,
    parse_rvdb_ascii,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _metadata(tmp_path: Path, deck_text: str = "// stub deck\n") -> PDKMetadata:
    deck = tmp_path / "deck.svrf"
    deck.write_text(deck_text)
    return PDKMetadata(
        name      = "test_pdk",
        version   = "0.0.1",
        layers    = {"poly": (66, 20)},
        grid      = {"manufacturing_um": 0.005},
        drc_decks = {"calibre": deck},
    )


# ── Tool identity + deck resolution ─────────────────────────────────────────

def test_tool_name_is_calibre(tmp_path: Path):
    r = CalibreDRCRunner(_metadata(tmp_path))
    assert r.tool_name == "calibre"


def test_deck_resolution_from_metadata(tmp_path: Path):
    md = _metadata(tmp_path)
    r = CalibreDRCRunner(md)
    assert r._resolve_deck() == md.drc_decks["calibre"]


def test_deck_resolution_env_override(tmp_path: Path, monkeypatch):
    override = tmp_path / "override.svrf"
    override.write_text("// override\n")
    monkeypatch.setenv("LITHOS_CALIBRE_DECK", str(override))
    r = CalibreDRCRunner(_metadata(tmp_path))
    assert r._resolve_deck() == override


def test_deck_missing_raises(tmp_path: Path):
    md = PDKMetadata(
        name="x", version="0", layers={}, grid={},
        drc_decks={"calibre": tmp_path / "nope.svrf"},
    )
    r = CalibreDRCRunner(md)
    with pytest.raises(FileNotFoundError):
        r._resolve_deck()


# ── Command construction ────────────────────────────────────────────────────

def test_docker_run_command_default(tmp_path: Path):
    """Default path: fresh ``docker run --rm`` with our standard mounts."""
    r = CalibreDRCRunner(_metadata(tmp_path), image="my/img",
                         network="my-net", cshrc="/etc/profile.d/foo.csh")
    cmd = r._build_command(tmp_path)
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--network" in cmd and "my-net" in cmd
    assert "my/img" in cmd
    # csh-style invocation: `csh -c "source <cshrc>; calibre ..."`
    csh_idx = cmd.index("csh")
    assert cmd[csh_idx + 1] == "-c"
    invocation = cmd[csh_idx + 2]
    assert "source /etc/profile.d/foo.csh" in invocation
    assert "calibre -drc -hier -turbo" in invocation
    assert str(tmp_path / "runset.svrf") in invocation


def test_docker_exec_when_container_set(tmp_path: Path):
    """Container override: ``docker exec`` against the named container."""
    r = CalibreDRCRunner(_metadata(tmp_path), exec_container="tools-running")
    cmd = r._build_command(tmp_path)
    assert cmd[:2] == ["docker", "exec"]
    assert "tools-running" in cmd
    assert "csh" in cmd                                  # invokes csh, not bash
    assert "--rm" not in cmd                             # no run flags


# ── ASCII RVDB parser ───────────────────────────────────────────────────────

def test_parse_rvdb_polygon_record(tmp_path: Path):
    """A polygon record under one rule produces one violation."""
    rvdb = tmp_path / "out.results"
    rvdb.write_text(
        "calibre header line\n"
        "test_cell\n"
        "POLY.SP.1\n"
        "07/01/2026 12:00:00 1 1\n"
        "Poly-to-poly spacing\n"
        "p 1 4 1 0 0\n"
        "1.000 1.000\n"
        "2.000 1.000\n"
        "2.000 2.000\n"
        "1.000 2.000\n"
    )
    vs = parse_rvdb_ascii(rvdb)
    assert len(vs) == 1
    v = vs[0]
    assert v.rule == "POLY.SP.1"
    assert "Poly-to-poly spacing" in v.description
    # Centroid of unit square at (1,1)→(2,2) = (1.5, 1.5).
    assert v.x == pytest.approx(1.5)
    assert v.y == pytest.approx(1.5)


def test_parse_rvdb_edge_record(tmp_path: Path):
    """Edge records pack both endpoints into the header line."""
    rvdb = tmp_path / "out.results"
    rvdb.write_text(
        "test_cell\n"
        "M1.W.1\n"
        "07/01/2026 12:00:00 1 1\n"
        "Metal-1 minimum width\n"
        "e 1 2 1 0.100 0.000 0.100 0.500\n"
    )
    vs = parse_rvdb_ascii(rvdb)
    assert len(vs) == 1
    v = vs[0]
    assert v.rule == "M1.W.1"
    assert v.x == pytest.approx(0.100)
    assert v.y == pytest.approx(0.250)


def test_parse_rvdb_multiple_rules_and_records(tmp_path: Path):
    """Multiple rules + multiple records per rule → flat violation list."""
    rvdb = tmp_path / "out.results"
    rvdb.write_text(
        "test_cell\n"
        "RULE.A\n"
        "07/01/2026 12:00:00 2 2\n"
        "Rule A description\n"
        "e 1 2 1 0.0 0.0 1.0 0.0\n"
        "e 1 2 1 2.0 2.0 3.0 2.0\n"
        "RULE.B\n"
        "07/01/2026 12:00:00 1 1\n"
        "Rule B description\n"
        "p 1 3 1 5 5\n"
        "5.000 5.000\n"
        "6.000 5.000\n"
        "5.500 6.000\n"
    )
    vs = parse_rvdb_ascii(rvdb)
    assert len(vs) == 3
    rules = [v.rule for v in vs]
    assert rules == ["RULE.A", "RULE.A", "RULE.B"]
    # First edge centroid = (0.5, 0); second = (2.5, 2.0); polygon centroid ≈ (5.5, 5.33).
    assert vs[0].x == pytest.approx(0.5)
    assert vs[1].x == pytest.approx(2.5)
    assert vs[2].x == pytest.approx(5.5)
    assert vs[2].y == pytest.approx(5.333, rel=1e-3)


def test_parse_rvdb_empty_returns_empty(tmp_path: Path):
    rvdb = tmp_path / "out.results"
    rvdb.write_text("calibre header\nno-rules-here\n")
    # `no-rules-here` matches our rule-name regex, but has no records under
    # it, so we expect zero violations.
    assert parse_rvdb_ascii(rvdb) == []


# ── Top-cell sniffer ────────────────────────────────────────────────────────

def test_detect_top_cell_minimal_gds(tmp_path: Path):
    """A minimal valid GDS with STRNAME ``TOP`` is detected correctly."""
    gds = tmp_path / "tiny.gds"
    # Build header records: HEADER(2-byte len, type, data type)+...
    # We just need a STRNAME record (type 0x06, dtype 0x06) followed by
    # the cell name. Earlier records can be anything as long as their
    # length fields are sane.
    def rec(rec_type: int, dtype: int, payload: bytes) -> bytes:
        rec_len = 4 + len(payload)
        return struct.pack(">H", rec_len) + bytes([rec_type, dtype]) + payload

    # HEADER (0x00,0x02 i2): version=600
    header = rec(0x00, 0x02, struct.pack(">h", 600))
    # STRNAME (0x06,0x06): "TOP" — even length required by GDS, so pad.
    strname = rec(0x06, 0x06, b"TOP\x00")
    gds.write_bytes(header + strname)
    assert _detect_top_cell(gds) == "TOP"


def test_detect_top_cell_missing_returns_none(tmp_path: Path):
    gds = tmp_path / "empty.gds"
    gds.write_bytes(b"")
    assert _detect_top_cell(gds) is None
