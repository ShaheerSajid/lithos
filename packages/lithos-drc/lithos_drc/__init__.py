"""lithos_drc — DRC runner abstraction + per-tool backends.

Exposes a tool-agnostic ``DRCRunner`` interface (subclassed per tool:
KLayout, Magic, Calibre, ...), a small registry of named backends, and a
resolver that maps tool-emitted check names to canonical foundry rule
codes via :class:`lithos_core.RuleDB`.

Concrete backends self-register at import time. Currently registered:

* ``"klayout"`` → :class:`lithos_drc.klayout_runner.KLayoutDRCRunner`
* ``"magic"``   → :class:`lithos_drc.magic_runner.MagicDRCRunner`
"""

from lithos_drc.base import DRCRunner, DRCViolation
from lithos_drc.resolver import (
    ResolvedViolation,
    partition_unresolved,
    resolve_violations,
)
from lithos_drc.klayout_runner import KLayoutDRCRunner
from lithos_drc.magic_runner   import MagicDRCRunner
from lithos_drc import registry

# Self-register the bundled backends.
registry.register("klayout", KLayoutDRCRunner)
registry.register("magic",   MagicDRCRunner)

__all__ = [
    "DRCRunner",
    "DRCViolation",
    "ResolvedViolation",
    "partition_unresolved",
    "resolve_violations",
    "registry",
    "KLayoutDRCRunner",
    "MagicDRCRunner",
]
