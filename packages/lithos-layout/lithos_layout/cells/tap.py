"""lithos_layout.cells.tap — standalone well/substrate tap cell generator.

Generates a tap cell containing a P+ substrate tap (VSS) and an N+ nwell
tap (VDD). Designed to tile alongside logic cells at regular intervals
for latch-up prevention and well biasing.

All dimensions are derived from the rule DB via :class:`BootstrapRules`;
nothing is hardcoded. Layer naming uses lithos's PDK-agnostic stack
(``contact``, ``m0``, ``m1``, ``via_m0_m1``); each PDK YAML maps these
to its physical (gds_layer, datatype) pairs.

Usage::

    from lithos_layout import draw_tap_cell, BootstrapRules
    rules = BootstrapRules(metadata, db, mapping)
    comp = draw_tap_cell(rules, cell_height=2.72)
"""
from __future__ import annotations

import itertools
from typing import Any

from lithos_layout.rules import BootstrapRules

_CELL_COUNTER = itertools.count(1)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _rect(comp: Any, x0: float, x1: float, y0: float, y1: float, layer: Any) -> None:
    comp.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )


def _activate_pdk() -> None:
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except ValueError:
        try:
            from gdsfactory.gpdk import get_generic_pdk
            get_generic_pdk().activate()
        except ImportError:
            from gdsfactory.generic_tech import PDK as _GENERIC
            _GENERIC.activate()


def _layer_or(rules: BootstrapRules, name: str, fallback: str) -> tuple[int, int] | None:
    """Return ``rules.layer(name)`` if defined, else ``rules.layer(fallback)`` if
    defined, else ``None``.
    """
    try:
        return rules.layer(name)
    except KeyError:
        pass
    try:
        return rules.layer(fallback)
    except KeyError:
        return None


# ── Main generator ──────────────────────────────────────────────────────────

def draw_tap_cell(
    rules: BootstrapRules,
    cell_height: float | None = None,
) -> Any:
    """Generate a standalone tap cell with substrate and nwell taps.

    Parameters
    ----------
    rules :
        Bootstrap-rule accessor over the PDK metadata + rule DB.
    cell_height :
        Total cell height in µm. If ``None``, computed from minimum tap
        geometry + spacing requirements.

    Returns
    -------
    gdsfactory.Component
        The generated tap cell with ``VDD`` (north) and ``GND`` (south)
        ports on m1.
    """
    _activate_pdk()
    import gdsfactory as gf

    _n = next(_CELL_COUNTER)
    _name = "tap_cell" if _n == 1 else f"tap_cell${_n}"
    comp = gf.Component(_name)

    # ── Extract rule values ──────────────────────────────────────────────
    c_size   = rules.contact["size_um"]
    impl_enc = rules.implant.get("enclosure_of_diff_um", 0.125)

    tap_enc      = rules.tap.get("enclosure_of_contact_um", c_size * 0.7)
    tap_to_diff  = rules.tap.get("spacing_to_diff_um", rules.diff["spacing_min_um"])

    via_cut_sz   = rules.via_m0_m1.get("size_um", c_size)
    m1_enc       = rules.m1.get(
        "enclosure_of_via_m0_m1_2adj_um",
        rules.via_m0_m1.get("enclosure_in_m1_um", 0.0),
    )

    enc_m0_2adj  = rules.contact.get(
        "enclosure_in_m0_2adj_um",
        rules.contact.get("enclosure_in_m0_um", 0.0),
    )

    nw_enc   = rules.nwell.get("enclosure_of_pdiff_um", 0.18)
    nw_min_w = rules.nwell.get("width_min_um", 0.84)

    rail_h = rules.m1["width_min_um"]

    # Tap diffusion size (square, enclosing one contact cut)
    tap_w = c_size + 2 * tap_enc
    tap_h = tap_w

    # m0 pad half-width (must enclose the contact cut per the 2-adj rule)
    m0_min_w = rules.m0.get("width_min_um", c_size)
    m0_hw    = max(c_size / 2 + enc_m0_2adj, m0_min_w / 2)

    # ── Resolve cell height ──────────────────────────────────────────────
    min_height = 2 * (tap_h + impl_enc) + tap_to_diff
    if cell_height is None:
        cell_height = max(min_height, nw_min_w + tap_h + tap_to_diff)

    # ── Layer lookups ────────────────────────────────────────────────────
    # The tap diffusion layer is PDK-dependent. Many decks use a dedicated
    # "tap" GDS layer; for stacks without one, fall back to the regular
    # diffusion layer.
    lyr_tap     = _layer_or(rules, "tap", "diff")
    if lyr_tap is None:
        raise KeyError("draw_tap_cell needs either a 'tap' or 'diff' layer in the PDK metadata.")
    lyr_contact = rules.layer("contact")
    lyr_m0      = rules.layer("m0")
    lyr_m1      = rules.layer("m1")

    # via_m0_m1 may be absent on PDKs that collapse m0 into m1.
    lyr_via_m0_m1: tuple[int, int] | None = None
    if not rules.m0_is_m1:
        try:
            lyr_via_m0_m1 = rules.layer("via_m0_m1")
        except KeyError:
            lyr_via_m0_m1 = None

    lyr_nimplant = rules.layer("nimplant")
    lyr_pimplant = rules.layer("pimplant")

    lyr_nwell: tuple[int, int] | None = None
    try:
        lyr_nwell = rules.layer("nwell")
    except KeyError:
        pass

    # ── Cell width: enough for one tap + implant enclosure ───────────────
    cell_w = max(tap_w + 2 * impl_enc, nw_min_w)
    cx = cell_w / 2  # centre X for all taps

    # ── Helper: draw one complete tap contact stack ──────────────────────
    def _draw_tap(tap_cx: float, tap_cy: float, implant_lyr: tuple[int, int]) -> None:
        half = tap_w / 2
        ch   = c_size / 2
        vch  = via_cut_sz / 2

        # 1. Tap diffusion
        _rect(comp, tap_cx - half, tap_cx + half,
              tap_cy - half, tap_cy + half, lyr_tap)

        # 2. Contact cut (centred)
        _rect(comp, tap_cx - ch, tap_cx + ch,
              tap_cy - ch, tap_cy + ch, lyr_contact)

        # 3. m0 pad
        _rect(comp, tap_cx - m0_hw, tap_cx + m0_hw,
              tap_cy - m0_hw, tap_cy + m0_hw, lyr_m0)

        # 4. m0 → m1 cut (skipped if m0 and m1 collapse)
        if lyr_via_m0_m1 is not None:
            _rect(comp, tap_cx - vch, tap_cx + vch,
                  tap_cy - vch, tap_cy + vch, lyr_via_m0_m1)

        # 5. m1 landing pad
        if lyr_via_m0_m1 is not None:
            m1_hw = vch + m1_enc
            _rect(comp, tap_cx - m1_hw, tap_cx + m1_hw,
                  tap_cy - m1_hw, tap_cy + m1_hw, lyr_m1)

        # 6. Implant enclosure
        _rect(comp, tap_cx - half - impl_enc, tap_cx + half + impl_enc,
              tap_cy - half - impl_enc, tap_cy + half + impl_enc,
              implant_lyr)

    # ── P+ substrate tap (bottom, VSS bias) ──────────────────────────────
    ptap_cy = rail_h + tap_h / 2
    _draw_tap(cx, ptap_cy, lyr_pimplant)

    # m1 bottom rail (GND)
    _rect(comp, 0, cell_w, 0, rail_h, lyr_m1)

    # ── N+ nwell tap (top, VDD bias) ────────────────────────────────────
    ntap_cy = cell_height - rail_h - tap_h / 2
    _draw_tap(cx, ntap_cy, lyr_nimplant)

    # m1 top rail (VDD)
    _rect(comp, 0, cell_w, cell_height - rail_h, cell_height, lyr_m1)

    # ── Nwell (enclose ntap + n-implant) ────────────────────────────────
    if lyr_nwell is not None:
        nw_y0 = ntap_cy - tap_h / 2 - impl_enc - nw_enc
        nw_y1 = cell_height
        nw_x0 = cx - max(tap_w / 2 + impl_enc + nw_enc, nw_min_w / 2)
        nw_x1 = cx + max(tap_w / 2 + impl_enc + nw_enc, nw_min_w / 2)
        # Enforce minimum nwell width in Y
        if nw_y1 - nw_y0 < nw_min_w:
            nw_y0 = nw_y1 - nw_min_w
        _rect(comp, nw_x0, nw_x1, nw_y0, nw_y1, lyr_nwell)

    # ── Ports ────────────────────────────────────────────────────────────
    comp.add_port(
        name="GND", center=(cell_w / 2, rail_h / 2),
        width=cell_w, orientation=270, layer=lyr_m1,
    )
    comp.add_port(
        name="VDD", center=(cell_w / 2, cell_height - rail_h / 2),
        width=cell_w, orientation=90, layer=lyr_m1,
    )

    return comp
