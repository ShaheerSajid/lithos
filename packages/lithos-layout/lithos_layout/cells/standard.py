"""lithos_layout.cells.standard — geometry helpers for CMOS standard cells.

Shared helper functions used by the synthesizer, placer, and router for
computing transistor source/drain and gate extents, diffusion Y spans,
inter-cell gaps, and rectangle emission with manufacturing-grid snap.

All dimensions in µm, all rules pulled from the rule DB via
:class:`lithos_layout.BootstrapRules`.
"""
from __future__ import annotations

from lithos_layout.rules import BootstrapRules
from lithos_layout.transistor import TransistorGeom


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _sd_x(
    j: int,
    geom: TransistorGeom,
    rules: BootstrapRules | None = None,
) -> tuple[float, float]:
    """(x0, x1) of the j-th source/drain li1 region in local transistor coords.

    When *rules* is provided, returns the licon-width li1 strip extent
    (zero X enclosure, matching the reference transistor geometry).
    """
    cx = j * (geom.sd_length_um + geom.l_um) + geom.sd_length_um / 2

    if rules is not None:
        c_half = rules.contact["size_um"] / 2
        return cx - c_half, cx + c_half

    x0 = j * (geom.sd_length_um + geom.l_um)
    return x0, x0 + geom.sd_length_um


def _gate_x(i: int, geom: TransistorGeom) -> tuple[float, float]:
    """(x0, x1) of the i-th poly gate finger in local transistor coords."""
    x0 = (i + 1) * geom.sd_length_um + i * geom.l_um
    return x0, x0 + geom.l_um


def _diff_y(geom: TransistorGeom, rules: BootstrapRules) -> tuple[float, float]:
    """(y0, y1) of diffusion in local transistor Y coords.

    Diff is contained within the poly gate in Y — poly overhangs diff by
    ``poly.endcap_over_diff_um`` on each side.
    """
    endcap = rules.get("poly.endcap_over_diff_um")
    return endcap, endcap + geom.w_finger_um


def _inter_cell_gap(rules: BootstrapRules) -> float:
    """Minimum Y gap between NMOS poly top edge and PMOS poly bottom edge
    such that the diff-to-diff spacing rule is satisfied.
    """
    endcap  = rules.get("poly.endcap_over_diff_um")
    min_sep = rules.get("diff.spacing_min_um")
    return max(0.0, min_sep - 2 * endcap)


def _routing_gap(rules: BootstrapRules) -> float:
    """Y gap large enough to fit one horizontal m0 routing track between
    NMOS and PMOS (needed for multi-input cells).
    """
    m0_sp   = rules.get("m0.spacing_min_um")
    m0_w    = rules.get("m0.width_min_um")
    endcap  = rules.get("poly.endcap_over_diff_um")
    ext     = rules.get("diff.extension_past_poly_um")
    min_sep = rules.get("diff.spacing_min_um")
    needed  = 2 * m0_sp + m0_w
    base    = max(0.0, min_sep - 2 * endcap + 2 * ext)
    gap_needed = needed - 2 * endcap + 2 * ext
    return max(base, gap_needed)


def _snap(value: float, grid: float = 0.005) -> float:
    """Snap *value* to nearest manufacturing grid point (default 5 nm)."""
    if grid <= 0:
        return value
    return round(round(value / grid) * grid, 6)


def _rect(c, x0: float, x1: float, y0: float, y1: float, layer,
           snap_grid: float = 0.005) -> None:
    """Add a rectangle polygon to component c, snapped to mfg grid."""
    x0, x1 = _snap(x0, snap_grid), _snap(x1, snap_grid)
    y0, y1 = _snap(y0, snap_grid), _snap(y1, snap_grid)
    c.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )
