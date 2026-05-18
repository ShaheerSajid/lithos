"""DRC runner registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_core.metadata import PDKMetadata

from lithos_drc import registry
from lithos_drc.base import DRCRunner, DRCViolation


class _Stub(DRCRunner):
    @property
    def tool_name(self) -> str:
        return "stub"

    def run(self, gds_path, cell_name=None):
        return []


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty registry; bundled backends are restored
    after each test so subsequent test modules (or test ordering changes) see
    the registry in its post-import state."""
    snapshot = dict(registry._RUNNERS)              # capture name → class
    for name in list(snapshot):
        registry.unregister(name)
    try:
        yield
    finally:
        for name in list(registry._RUNNERS):        # drop anything the test added
            registry.unregister(name)
        for name, cls in snapshot.items():
            registry.register(name, cls)


def test_register_and_get():
    registry.register("stub", _Stub)
    assert "stub" in registry.available()
    md = PDKMetadata(name="x", version="0", layers={}, grid={}, drc_decks={})
    inst = registry.get("stub", metadata=md)
    assert isinstance(inst, _Stub)


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="missing"):
        registry.get("missing")


def test_register_overwrites_existing():
    registry.register("stub", _Stub)
    class _Stub2(_Stub):
        pass
    registry.register("stub", _Stub2)
    md = PDKMetadata(name="x", version="0", layers={}, grid={}, drc_decks={})
    inst = registry.get("stub", metadata=md)
    assert type(inst) is _Stub2
