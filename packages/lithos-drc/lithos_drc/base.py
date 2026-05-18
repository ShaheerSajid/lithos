"""lithos_drc.base — tool-agnostic DRC runner interface.

Every DRC backend (KLayout, Magic, Calibre, PVS, ICV, ...) subclasses
:class:`DRCRunner` and returns a list of :class:`DRCViolation`. Downstream
callers only ever see this interface; the resolver in
:mod:`lithos_drc.resolver` is the layer that maps each violation's
tool-emitted ``rule`` string to a canonical foundry code via
:class:`lithos_core.RuleDB`.

A new backend:

    1. Subclass :class:`DRCRunner`.
    2. Implement :attr:`tool_name` and :meth:`run`.
    3. Register with :func:`lithos_drc.registry.register`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lithos_core.metadata import PDKMetadata


@dataclass(frozen=True)
class DRCViolation:
    """One DRC violation as returned by any backend.

    ``rule`` is the **raw tool-emitted check name** — whatever string the
    DRC tool wrote into its report (the SVRF ``RULECHECK`` title, a Magic
    feedback label, a KLayout rule object's name, etc.). Use the resolver
    in :mod:`lithos_drc.resolver` to map this to a canonical foundry code.

    Attributes
    ----------
    rule
        Tool-emitted check name. Looks up in :class:`lithos_core.RuleDB`'s
        ``rule_alias`` table.
    description
        Human-readable description from the deck (often a copy of the
        rule's title; tool-dependent).
    layer
        Logical PDK layer name involved, when determinable from the report.
    severity
        ``"error"`` (blocks signoff) or ``"warning"`` (informational).
    x, y
        Centroid of the violating geometry in µm (cell coordinates).
    value
        Measured value (µm) that caused the violation, or ``None`` if the
        backend does not report it.
    """
    rule:        str
    description: str = ""
    layer:       str = ""
    severity:    str = "error"
    x:           float = 0.0
    y:           float = 0.0
    value:       Optional[float] = None

    def __repr__(self) -> str:
        loc = f"({self.x:.3f}, {self.y:.3f})"
        val = f" measured={self.value:.4f}" if self.value is not None else ""
        return f"DRCViolation({self.rule!r}{val} @ {loc})"


class DRCRunner(ABC):
    """Abstract base for DRC tool backends.

    Parameters
    ----------
    metadata
        PDK metadata. The runner uses it to locate the deck file
        (``metadata.drc_decks[tool_name]``) and to resolve logical layer
        names when parsing violation reports.
    """

    def __init__(self, metadata: PDKMetadata):
        self.metadata = metadata

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Short identifier for this backend (e.g. ``"klayout"``)."""

    @abstractmethod
    def run(
        self,
        gds_path: Path,
        cell_name: Optional[str] = None,
    ) -> list[DRCViolation]:
        """Run DRC on ``gds_path`` and return all violations.

        Parameters
        ----------
        gds_path
            Path to the GDS file under test.
        cell_name
            Top-cell name to check. If ``None`` the runner picks the last
            cell in the file (tool-dependent behaviour).
        """

    def deck_path(self) -> Path:
        """The DRC deck file path for this tool, from PDK metadata."""
        try:
            return self.metadata.drc_decks[self.tool_name]
        except KeyError as exc:
            raise KeyError(
                f"PDK metadata {self.metadata.name!r} has no DRC deck "
                f"configured for tool {self.tool_name!r}. "
                f"Available: {sorted(self.metadata.drc_decks)}"
            ) from exc

    def is_available(self) -> bool:
        """Return ``True`` if the tool executable is reachable.

        Default returns ``True``. Concrete backends should probe for their
        binary (``which klayout``, etc.) and return ``False`` when missing
        so callers can gracefully fall back or skip tests.
        """
        return True

    def count(
        self,
        gds_path: Path,
        cell_name: Optional[str] = None,
    ) -> int:
        """Convenience: return the number of DRC violations."""
        return len(self.run(gds_path, cell_name))
