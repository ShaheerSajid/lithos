"""lithos_layout.synth — topology-driven cell synthesis.

Pipeline (in roadmap order):

* :mod:`lithos_layout.synth.loader` — topology YAML → typed dataclasses.
  No PDK knowledge, no numeric evaluation; every symbolic expression
  stays a string for downstream evaluation.
* placer / router / synthesizer / netlist — not yet ported.

This subpackage is intentionally PDK-agnostic. Anything that touches
the rule DB goes through :class:`lithos_layout.BootstrapRules`; anything
that touches GDS layers uses the canonical lithos stack vocabulary
(``m0``, ``m1``, ``m2``, …, ``contact``, ``via_m0_m1``, …).
"""
from __future__ import annotations

from lithos_layout.synth.loader import (
    AbutmentSpec,
    CellDimensions,
    CellTemplate,
    DeviceSpec,
    LabelLayerSpec,
    NetSpec,
    PlacementDirective,
    PortSpec,
    RoutingHint,
    RoutingSpec,
    RowPairSpec,
    load_template,
)

__all__ = [
    "AbutmentSpec",
    "CellDimensions",
    "CellTemplate",
    "DeviceSpec",
    "LabelLayerSpec",
    "NetSpec",
    "PlacementDirective",
    "PortSpec",
    "RoutingHint",
    "RoutingSpec",
    "RowPairSpec",
    "load_template",
]
