"""lithos_layout — layout generation (cells, primitives, synth, templates).

The "draw" side. Consumes the bootstrap subset of the rule DB
(``usage_class = "geometry_primitive"``) via :class:`BootstrapRules` plus
topology templates to produce candidate GDS for the repair loop to refine.

lithos uses a PDK-agnostic metal stack (``m0``, ``m1``, ``m2``, …) with
``contact`` for poly/diff → m0 cuts and ``via_mX_mY`` for inter-metal
cuts. The per-PDK YAML maps these abstract layer names to physical
(gds_layer, datatype) pairs.

Current public surface:

* :class:`BootstrapMapping`, :func:`load_bootstrap_mapping` — per-PDK
  translation from semantic dotted-keys to canonical rule codes.
* :class:`BootstrapRules` — wraps PDKMetadata + RuleDB + BootstrapMapping
  and exposes both ``rules.get("poly.width_min_um")`` and the
  ``rules.poly["width_min_um"]`` dict idiom.
* :class:`TransistorGeom`, :func:`finger_count`, :func:`transistor_geom`,
  :func:`sd_contact_columns`, :func:`draw_transistor` — single-transistor
  dimension math + GDS emitter.
* :mod:`lithos_layout.cells` — atomic via cell factories
  (``via_poly_m0`` / ``via_m0_m1`` / …) and the tap cell.
* :func:`load_template` and the dataclasses in :mod:`lithos_layout.synth`
  — topology-YAML loader (zero PDK dependency).
"""

from lithos_layout.cells import (
    draw_tap_cell,
    via_diff_m0,
    via_m0_m1,
    via_m0_m2,
    via_m1_m2,
    via_poly_m0,
    via_poly_m1,
    via_poly_m2,
)
from lithos_layout.rules import (
    BootstrapMapping,
    BootstrapRules,
    load_bootstrap_mapping,
)
from lithos_layout.synth import (
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
from lithos_layout.transistor import (
    TransistorGeom,
    draw_transistor,
    finger_count,
    sd_contact_columns,
    transistor_geom,
)

__all__ = [
    "BootstrapMapping",
    "BootstrapRules",
    "load_bootstrap_mapping",
    "TransistorGeom",
    "draw_transistor",
    "finger_count",
    "sd_contact_columns",
    "transistor_geom",
    "draw_tap_cell",
    "via_poly_m0",
    "via_diff_m0",
    "via_m0_m1",
    "via_m1_m2",
    "via_poly_m1",
    "via_poly_m2",
    "via_m0_m2",
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
