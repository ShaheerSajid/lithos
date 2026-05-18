"""lithos_layout — layout generation (cells, primitives, synth, templates).

The "draw" side. Consumes the bootstrap subset of the rule DB
(``usage_class = "geometry_primitive"``) via :class:`BootstrapRules` plus
topology templates to produce candidate GDS for the repair loop to refine.

Current public surface (geometry/dimension math half; GDS emitter lands
in a follow-up):

* :class:`BootstrapMapping`, :func:`load_bootstrap_mapping` — per-PDK
  translation from semantic dotted-keys to canonical rule codes.
* :class:`BootstrapRules` — wraps PDKMetadata + RuleDB + BootstrapMapping
  and exposes both ``rules.get("poly.width_min_um")`` and the
  ``rules.poly["width_min_um"]`` dict idiom.
* :class:`TransistorGeom`, :func:`finger_count`, :func:`transistor_geom`,
  :func:`sd_contact_columns` — pure-data transistor dimensioning.
"""

from lithos_layout.rules import (
    BootstrapMapping,
    BootstrapRules,
    load_bootstrap_mapping,
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
]
