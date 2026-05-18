"""DRCViolation and DRCRunner base behaviour."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from lithos_core.metadata import PDKMetadata

from lithos_drc.base import DRCRunner, DRCViolation


def _fake_metadata(tmp_path: Path) -> PDKMetadata:
    return PDKMetadata(
        name      = "test_pdk",
        version   = "0.0.1",
        layers    = {"met2": (69, 20)},
        grid      = {"manufacturing_um": 0.005},
        drc_decks = {"klayout": tmp_path / "deck.lydrc"},
    )


def test_violation_defaults_and_repr():
    v = DRCViolation(rule="M2.S.1", x=1.5, y=2.25, value=0.123)
    assert v.severity == "error"
    assert v.layer == ""
    assert v.description == ""
    r = repr(v)
    assert "M2.S.1" in r
    assert "1.500" in r and "2.250" in r
    assert "measured=0.1230" in r


def test_violation_repr_without_value():
    v = DRCViolation(rule="X.1")
    r = repr(v)
    assert "measured=" not in r


class _FakeRunner(DRCRunner):
    """Minimal concrete DRCRunner for testing the base class plumbing."""

    @property
    def tool_name(self) -> str:
        return "klayout"

    def run(self, gds_path, cell_name=None):
        return [
            DRCViolation(rule="M2.S.1", layer="met2", x=0.0, y=0.0, value=0.10),
            DRCViolation(rule="m2_sp_70", layer="met2", x=1.0, y=0.5, value=0.12),
        ]


def test_count_delegates_to_run(tmp_path: Path):
    runner = _FakeRunner(_fake_metadata(tmp_path))
    assert runner.count(tmp_path / "nope.gds") == 2


def test_deck_path_lookup(tmp_path: Path):
    runner = _FakeRunner(_fake_metadata(tmp_path))
    assert runner.deck_path() == tmp_path / "deck.lydrc"


def test_deck_path_missing(tmp_path: Path):
    md = PDKMetadata(
        name="x", version="0", layers={}, grid={}, drc_decks={},  # no klayout entry
    )
    runner = _FakeRunner(md)
    with pytest.raises(KeyError, match="klayout"):
        runner.deck_path()
