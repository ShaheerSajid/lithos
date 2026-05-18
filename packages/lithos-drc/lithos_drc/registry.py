"""lithos_drc.registry — tool name → DRCRunner class mapping.

Backends register here so the rest of the codebase can stay tool-agnostic::

    from lithos_drc import registry
    registry.register("calibre", CalibreDRCRunner)
    runner = registry.get("calibre", metadata=my_pdk_metadata)
"""
from __future__ import annotations

from typing import Type

from lithos_drc.base import DRCRunner

_RUNNERS: dict[str, Type[DRCRunner]] = {}


def register(name: str, cls: Type[DRCRunner]) -> None:
    """Register ``cls`` under ``name``. Overwrites any existing entry."""
    _RUNNERS[name] = cls


def get(name: str, **kwargs) -> DRCRunner:
    """Instantiate the runner registered under ``name``, forwarding ``kwargs``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _RUNNERS:
        raise KeyError(
            f"No DRC runner registered for tool {name!r}. "
            f"Available: {sorted(_RUNNERS)}"
        )
    return _RUNNERS[name](**kwargs)


def available() -> list[str]:
    """Sorted list of registered tool names."""
    return sorted(_RUNNERS)


def unregister(name: str) -> None:
    """Remove ``name`` from the registry. Safe to call when absent."""
    _RUNNERS.pop(name, None)
