"""lithos_repair — DRC repair primitives and fix-graph engine.

Given a list of violations from ``lithos-drc`` and the rule DB from
``lithos-core``, this package applies fixes. It reads each violated
rule's ``fix_metadata`` (allowed/forbidden action classes), evaluates
the constraint AST against the local geometry, and chooses a remedy.
The ``rule_relation`` cross-reference graph drives anticipating
downstream violations.

Public surface (M2 — action library + feature schema):

* :class:`~lithos_repair.features.Polygon`,
  :class:`~lithos_repair.features.PolygonRef`,
  :class:`~lithos_repair.features.FreeSpace`,
  :class:`~lithos_repair.features.Neighbor`,
  :class:`~lithos_repair.features.ViolationContext` — schema the
  agent / policy sees per violation.
* :func:`~lithos_repair.actions.extract_polygons`,
  :func:`~lithos_repair.actions.rebuild_component` — the gdsfactory ↔
  polygon-list bridge.
* :data:`REGISTRY` — the populated :class:`ActionRegistry` exposing the
  v1 verb vocabulary (widen, narrow, shift_*, extend, shrink,
  snap_to_grid, remove, redraw).
"""
from __future__ import annotations

from .actions import (
    EdgeParams,
    NarrowParams,
    RedrawParams,
    RemoveParams,
    ShiftParams,
    SnapParams,
    WidenParams,
    extract_polygons,
    find_polygon,
    rebuild_component,
)
from .agent import AgentConfig, AgentProposalError, LLMRepairAgent, ProposedAction
from .analyzer import analyze
from .fix_log import FixLog, FixOutcome, FixRow, FixSource
from .loop import RepairStep, RepairTrace, repair_cell
from .features import (
    FreeSpace,
    LayerTuple,
    Neighbor,
    Polygon,
    PolygonRef,
    ViolationContext,
    polygon_ref,
)
from .registry import REGISTRY, ActionDef, ActionRegistry

__all__ = [
    "ActionDef",
    "ActionRegistry",
    "AgentConfig",
    "AgentProposalError",
    "EdgeParams",
    "FixLog",
    "FixOutcome",
    "FixRow",
    "FixSource",
    "LLMRepairAgent",
    "ProposedAction",
    "RepairStep",
    "RepairTrace",
    "repair_cell",
    "FreeSpace",
    "LayerTuple",
    "NarrowParams",
    "Neighbor",
    "Polygon",
    "PolygonRef",
    "REGISTRY",
    "RedrawParams",
    "RemoveParams",
    "ShiftParams",
    "SnapParams",
    "ViolationContext",
    "WidenParams",
    "analyze",
    "extract_polygons",
    "find_polygon",
    "polygon_ref",
    "rebuild_component",
]
