"""lithos_layout.cells.vias — via/contact cell generators.

Naming convention: lithos uses a PDK-agnostic metal stack ``m0``, ``m1``,
``m2`` (and a ``contact`` layer for poly/diff → m0 cuts). Each PDK YAML
maps these abstract names to its physical (gds_layer, datatype) pairs —
so the same cell code works against sky130 (where m0 is "li1") or a
foundry stack that calls its first metal "m1".

Each function returns a ``gdsfactory.Component`` representing one atomic
contact or via stack. All dimensions are derived from the rule DB via
:class:`BootstrapRules`.

Single-cut cells
----------------
via_poly_m0  — poly contact: poly pad + contact + m0 pad
via_diff_m0  — diff contact:           contact + m0 pad
via_m0_m1    — m0 → m1: via_m0_m1 + m1 pad
via_m1_m2    — m1 → m2: via_m1_m2 + m1 pad + m2 pad

Composite stacks
----------------
via_poly_m1  — poly pad + contact + m0 + via_m0_m1 + m1
via_poly_m2  — poly pad + contact + m0 + via_m0_m1 + m1 + via_m1_m2 + m2
via_m0_m2    — m0 + via_m0_m1 + m1 + via_m1_m2 + m2
"""
from __future__ import annotations

import itertools
from typing import Any

from lithos_layout.rules import BootstrapRules

_counter = itertools.count()


def _uname(base: str) -> str:
    return f"{base}_{next(_counter)}"


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


def _rect(comp: Any, x0: float, x1: float, y0: float, y1: float, layer: Any) -> None:
    comp.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )


# ── Single-cut cells ──────────────────────────────────────────────────────────

def via_poly_m0(rules: BootstrapRules) -> Any:
    """Poly contact: poly enclosure + contact + m0 pad.

    Centred at (0, 0). The poly pad provides the required enclosure of
    the contact cut, and the m0 pad provides m0 enclosure.
    """
    _activate_pdk()
    import gdsfactory as gf

    c_size = rules.contact["size_um"]
    ch = c_size / 2

    poly_enc      = rules.contact.get("poly_enclosure_um", 0.05)
    poly_enc_2adj = rules.contact.get("poly_enclosure_2adj_um", 0.08)

    m0_enc      = rules.contact.get("enclosure_in_m0_um", 0.0)
    m0_enc_2adj = rules.contact.get("enclosure_in_m0_2adj_um", 0.08)

    lyr_poly    = rules.layer("poly")
    lyr_contact = rules.layer("contact")
    lyr_m0      = rules.layer("m0")

    comp = gf.Component(_uname("via_poly_m0"))

    _rect(comp, -ch - poly_enc, ch + poly_enc,
                -ch - poly_enc_2adj, ch + poly_enc_2adj, lyr_poly)

    _rect(comp, -ch, ch, -ch, ch, lyr_contact)

    m0_min_w = rules.m0.get("width_min_um", c_size)
    m0_hx = max(ch + m0_enc_2adj, m0_min_w / 2)
    m0_hy = max(ch + m0_enc,      m0_min_w / 2)
    _rect(comp, -m0_hx, m0_hx, -m0_hy, m0_hy, lyr_m0)

    return comp


def via_diff_m0(rules: BootstrapRules) -> Any:
    """Diff contact: contact + m0 pad (no diff drawn — that's the transistor's job).

    Centred at (0, 0).
    """
    _activate_pdk()
    import gdsfactory as gf

    c_size = rules.contact["size_um"]
    ch = c_size / 2

    m0_enc      = rules.contact.get("enclosure_in_m0_um", 0.0)
    m0_enc_2adj = rules.contact.get("enclosure_in_m0_2adj_um", 0.08)

    lyr_contact = rules.layer("contact")
    lyr_m0      = rules.layer("m0")

    comp = gf.Component(_uname("via_diff_m0"))

    _rect(comp, -ch, ch, -ch, ch, lyr_contact)

    m0_min_w = rules.m0.get("width_min_um", c_size)
    m0_hx = max(ch + m0_enc,      m0_min_w / 2)
    m0_hy = max(ch + m0_enc_2adj, m0_min_w / 2)
    _rect(comp, -m0_hx, m0_hx, -m0_hy, m0_hy, lyr_m0)

    return comp


def via_m0_m1(rules: BootstrapRules) -> Any:
    """m0 → m1 via: cut + m1 pad. Centred at (0, 0).

    When m0 and m1 collapse to the same GDS layer (some foundries do this
    for their bottom routing layers), the cut is a no-op — only the m1
    landing pad is drawn.
    """
    _activate_pdk()
    import gdsfactory as gf

    cut_sz = rules.via_m0_m1.get("size_um", rules.contact["size_um"])
    ch     = cut_sz / 2

    m1_enc = rules.m1.get(
        "enclosure_of_via_m0_m1_2adj_um",
        rules.via_m0_m1.get("enclosure_in_m1_um", 0.03),
    )
    m1h = ch + m1_enc

    lyr_m1 = rules.layer("m1")

    comp = gf.Component(_uname("via_m0_m1"))

    if not rules.m0_is_m1:
        lyr_cut = rules.layer("via_m0_m1")
        _rect(comp, -ch, ch, -ch, ch, lyr_cut)
    _rect(comp, -m1h, m1h, -m1h, m1h, lyr_m1)

    return comp


def via_m1_m2(rules: BootstrapRules) -> Any:
    """m1 → m2 via: cut + m1 pad + m2 pad. Centred at (0, 0)."""
    _activate_pdk()
    import gdsfactory as gf

    cut_sz = rules.via_m1_m2.get("size_um", 0.15)
    ch     = cut_sz / 2

    m1_enc = rules.via_m1_m2.get(
        "enclosure_in_m1_2adj_um",
        rules.via_m1_m2.get("enclosure_in_m1_um", 0.055),
    )
    m2_enc = rules.m2.get(
        "enclosure_of_via_m1_m2_2adj_um",
        rules.via_m1_m2.get("enclosure_in_m2_um", 0.055),
    )
    m1h = ch + m1_enc
    m2h = ch + m2_enc

    lyr_cut = rules.layer("via_m1_m2")
    lyr_m1  = rules.layer("m1")
    lyr_m2  = rules.layer("m2")

    comp = gf.Component(_uname("via_m1_m2"))

    _rect(comp, -ch, ch, -ch, ch, lyr_cut)
    _rect(comp, -m1h, m1h, -m1h, m1h, lyr_m1)
    _rect(comp, -m2h, m2h, -m2h, m2h, lyr_m2)

    return comp


# ── Composite stacks ──────────────────────────────────────────────────────────

def via_poly_m1(rules: BootstrapRules) -> Any:
    """Poly → m1 stack: poly pad + contact + m0 + via_m0_m1 + m1.

    Centred at (0, 0). Used for gate connections to the first routing
    metal.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("via_poly_m1"))
    comp.add_ref(via_poly_m0(rules))
    comp.add_ref(via_m0_m1(rules))
    return comp


def via_poly_m2(rules: BootstrapRules) -> Any:
    """Poly → m2 stack. Centred at (0, 0). Used for cross-couple gate
    connections jumping a routing layer.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("via_poly_m2"))
    comp.add_ref(via_poly_m0(rules))
    comp.add_ref(via_m0_m1(rules))
    comp.add_ref(via_m1_m2(rules))
    return comp


def via_m0_m2(rules: BootstrapRules) -> Any:
    """m0 → m2 stack: via_m0_m1 + m1 + via_m1_m2 + m2. Centred at (0, 0).
    Used for bitline / signal handoffs from local interconnect up to m2.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("via_m0_m2"))
    comp.add_ref(via_m0_m1(rules))
    comp.add_ref(via_m1_m2(rules))
    return comp
