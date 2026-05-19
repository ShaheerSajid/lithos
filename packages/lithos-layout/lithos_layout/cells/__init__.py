"""lithos_layout.cells — standard-cell geometry helpers + via/tap primitives.

* :mod:`lithos_layout.cells.standard` — pure-math helpers (S/D extents,
  gate extents, diff-Y span, inter-cell gap, routing gap, grid snap, rect).
* :mod:`lithos_layout.cells.vias` — atomic contact and via-stack
  ``gdsfactory.Component`` factories (licon/poly/diff, mcon, via1, and
  the composite stacks used by the synthesizer/router).
* :mod:`lithos_layout.cells.tap` — standalone well/substrate tap cell.

Public surface re-exports the GDS-emitting cell factories; the pure-math
helpers in :mod:`standard` are imported via their module path.
"""
from __future__ import annotations

from lithos_layout.cells.tap import draw_tap_cell
from lithos_layout.cells.vias import (
    via_diff_m0,
    via_m0_m1,
    via_m0_m2,
    via_m1_m2,
    via_poly_m0,
    via_poly_m1,
    via_poly_m2,
)

__all__ = [
    "draw_tap_cell",
    "via_poly_m0",
    "via_diff_m0",
    "via_m0_m1",
    "via_m1_m2",
    "via_poly_m1",
    "via_poly_m2",
    "via_m0_m2",
]
