"""Shared pytest setup for lithos-layout tests.

gdsfactory ≥ 9 refuses to draw on a Component without an active PDK.
Several lithos modules — :func:`draw_transistor`, :func:`draw_via_stack`,
the router style handlers — activate the generic PDK lazily on first
call, but unit tests that call lower-level primitives
(:func:`_rect`, ``Component.add_polygon``, …) may hit the bare-Component
path before any of those guards. Activate the generic PDK once at
session start so every test sees a usable global.
"""
from __future__ import annotations


def pytest_sessionstart(session) -> None:           # noqa: ARG001
    """Activate gdsfactory's generic PDK so Component drawing works."""
    try:
        import gdsfactory as gf
    except ImportError:                              # pragma: no cover — opt-in
        return
    try:
        gf.get_active_pdk()
        return                                       # already active
    except ValueError:
        pass
    try:
        from gdsfactory.gpdk import get_generic_pdk
        get_generic_pdk().activate()
    except ImportError:                              # pragma: no cover
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()
