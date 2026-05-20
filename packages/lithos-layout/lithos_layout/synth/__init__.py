"""lithos_layout.synth — topology-driven cell synthesis.

Pipeline (in roadmap order):

* :mod:`lithos_layout.synth.loader` — topology YAML → typed dataclasses.
  No PDK knowledge, no numeric evaluation; every symbolic expression
  stays a string for downstream evaluation.
* :mod:`lithos_layout.synth.constraints` — symbolic expression
  evaluator. Resolves ``"rules.diff.spacing_min_um - …"`` style
  expressions to floats given a :class:`BootstrapRules` and per-device
  :class:`TransistorGeom` map.
* :mod:`lithos_layout.synth.netlist` — connectivity graph
  (:class:`NetGraph`) built from device terminals. The auto-router
  uses it to decide what needs routing.
* :mod:`lithos_layout.synth.euler` — Euler-path device ordering so
  adjacent transistors share diffusion (smaller, denser cells).
* :mod:`lithos_layout.synth.placer` — resolves floorplan directives
  (and the named-spacing-rule registry) into global ``(x, y)``
  device origins.
* :mod:`lithos_layout.synth.port_resolver` — compass-side port
  placement on the cell bounding box; emits ``expose_terminal``
  routing specs for the auto-router.
* router / auto_router / synthesizer — not yet ported.

This subpackage is intentionally PDK-agnostic. Anything that touches
the rule DB goes through :class:`lithos_layout.BootstrapRules`; anything
that touches GDS layers uses the canonical lithos stack vocabulary
(``m0``, ``m1``, ``m2``, …, ``contact``, ``via_m0_m1``, …).
"""
from __future__ import annotations

from lithos_layout.synth.constraints import (
    build_namespace,
    eval_expr,
    resolve_named_constraints,
)
from lithos_layout.synth.euler import (
    build_diffusion_graph,
    common_euler_order,
    euler_order,
    euler_path,
    has_euler_path,
)
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
from lithos_layout.synth.netlist import (
    NetGraph,
    NetInfo,
    TerminalRef,
    build_net_graph,
)
from lithos_layout.synth.placer import (
    Placer,
    PlacedDevice,
    SPACING_RULES,
    TerminalGeom,
    global_diff_y,
    global_gate_x,
    global_poly_bottom,
    global_poly_top,
    global_sd_x,
    resolve_spacing_rule,
    resolve_terminal,
)
from lithos_layout.synth.port_resolver import (
    PortCandidate,
    generate_expose_specs,
    resolve_ports,
)

__all__ = [
    # Loader
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
    # Constraints
    "build_namespace",
    "eval_expr",
    "resolve_named_constraints",
    # Netlist
    "NetGraph",
    "NetInfo",
    "TerminalRef",
    "build_net_graph",
    # Euler
    "build_diffusion_graph",
    "common_euler_order",
    "euler_order",
    "euler_path",
    "has_euler_path",
    # Placer
    "Placer",
    "PlacedDevice",
    "SPACING_RULES",
    "TerminalGeom",
    "global_diff_y",
    "global_gate_x",
    "global_poly_bottom",
    "global_poly_top",
    "global_sd_x",
    "resolve_spacing_rule",
    "resolve_terminal",
    # Port resolver
    "PortCandidate",
    "generate_expose_specs",
    "resolve_ports",
]
