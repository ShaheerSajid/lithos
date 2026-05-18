"""lithos_layout.transistor — single-transistor primitive (dimensions + GDS).

Two layers in one module:

  *  **Pure dimension math** — :class:`TransistorGeom`, :func:`finger_count`,
     :func:`transistor_geom`, :func:`sd_contact_columns`. No gdsfactory
     dependency at import time; cheap to use from sizing / placement code.
  *  **GDS emitter** — :func:`draw_transistor` returns a fully-drawn
     ``gdsfactory.Component`` with named ``G``, ``D``, ``S`` ports. The
     gdsfactory import is deferred to the function body so the dimension
     math is reachable without paying gdsfactory's import cost.

Orientation convention
----------------------
- X axis : channel length (L) direction — poly fingers run vertically
- Y axis : channel width  (W) direction — diffusion runs horizontally

Multi-finger layout (n=2 shown, shared source/drain)::

    ←  sd  →← L →←  sd  →← L →←  sd  →
    ┌───────┐┌────┐┌───────┐┌────┐┌───────┐   ─ poly_endcap
    │  S/D  ││poly││  S/D  ││poly││  S/D  │
    │ (li1) ││    ││ (li1) ││    ││ (li1) │
    └───────┘└────┘└───────┘└────┘└───────┘   ─ poly_endcap
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from lithos_layout.rules import BootstrapRules


@dataclass(frozen=True)
class TransistorGeom:
    """Computed transistor geometry — all dimensions in µm.

    Produced by :func:`transistor_geom`; consumed by the GDS emitter.

    Attributes
    ----------
    w_um, l_um
        Drawn channel width and length.
    device_type
        Logical device name (``"nmos"`` / ``"pmos"``).
    n_fingers
        Number of parallel gate fingers.
    w_finger_um
        Per-finger channel width = ``w_um / n_fingers``.
    sd_length_um
        Length of each source/drain contact region in the X direction.
    n_contacts_y
        Number of contact rows in the Y (width) direction per S/D column.
    total_x_um, total_y_um
        Overall bounding-box dimensions.
    """
    w_um:         float
    l_um:         float
    device_type:  str
    n_fingers:    int
    w_finger_um:  float
    sd_length_um: float
    n_contacts_y: int
    total_x_um:   float
    total_y_um:   float


# ── Helpers ─────────────────────────────────────────────────────────────────

def _min_channel_width(rules: BootstrapRules, device_type: str) -> float:
    """PDK minimum channel width (µm) for ``device_type``.

    Prefers an explicit ``channel_width_min_um`` from the device record;
    falls back to the diffusion width minimum from the rule DB.
    """
    dev = rules.device(device_type)
    w = float(dev.get("channel_width_min_um", 0.0) or 0.0)
    if w > 0:
        return w
    return rules.get("diff.width_min_um") if rules.has("diff.width_min_um") else 0.0


def finger_count(
    w_um: float, rules: BootstrapRules, device_type: str = "nmos",
) -> int:
    """Minimum number of gate fingers for total channel width ``w_um``.

    Fingers are added when ``w_um`` exceeds ``w_finger_max_um`` from the
    device record. Always returns at least 1; clamps so the per-finger
    width never drops below the PDK minimum channel width.
    """
    dev   = rules.device(device_type)
    w_max = float(dev.get("w_finger_max_um", 2.0))
    n = max(1, math.ceil(w_um / w_max))
    w_min = _min_channel_width(rules, device_type)
    if w_min > 0:
        n = min(n, max(1, int(w_um / w_min)))
    return n


def sd_contact_columns(w_finger_um: float, rules: BootstrapRules) -> int:
    """Number of contact rows that fit in an S/D of width ``w_finger_um``.

    Contacts array along the channel-width (Y) direction; the count is
    bounded by ``size_um + spacing_um`` plus the per-side enclosure.
    Returns at least 1 even if the math says 0.
    """
    c_size  = rules.get("contacts.size_um")
    c_space = rules.get("contacts.spacing_um")
    enc     = rules.get("contacts.enclosure_in_diff_um")
    usable  = w_finger_um - 2 * enc
    if usable < c_size:
        return 1
    return max(1, int((usable + c_space) / (c_size + c_space)))


def transistor_geom(
    w_um: float, l_um: float, device_type: str, rules: BootstrapRules,
) -> TransistorGeom:
    """Compute all transistor geometry parameters from PDK rules.

    Returns a fully-populated :class:`TransistorGeom`; pure data, no GDS.
    """
    dev = rules.device(device_type)
    n   = finger_count(w_um, rules, device_type)
    w_f = w_um / n

    c_size = rules.get("contacts.size_um")
    c_enc  = rules.get("contacts.enclosure_in_diff_um")
    sd     = max(float(dev.get("sd_length_min_um", 0.29)), c_size + 2 * c_enc)

    n_cy   = sd_contact_columns(w_f, rules)

    endcap = rules.get("poly.endcap_over_diff_um")
    total_x = (n + 1) * sd + n * l_um
    total_y = w_f + 2 * endcap

    return TransistorGeom(
        w_um         = w_um,
        l_um         = l_um,
        device_type  = device_type,
        n_fingers    = n,
        w_finger_um  = w_f,
        sd_length_um = sd,
        n_contacts_y = n_cy,
        total_x_um   = total_x,
        total_y_um   = total_y,
    )


# ── GDS emitter ─────────────────────────────────────────────────────────────

_CELL_COUNTER: dict[str, int] = {}
"""Tracks per-base-name instance count so repeated calls in one Python
session don't collide on gdsfactory's global cell registry (kfactory
raises on duplicate names)."""


def draw_transistor(
    w_um: float,
    l_um: float,
    device_type: str,
    rules: BootstrapRules,
    *,
    n_fingers: int | None = None,
    skip_sd:   set[int] | None = None,
):
    """Draw a single transistor and return a ``gdsfactory.Component``.

    Orientation: poly fingers run vertically (along Y), diffusion along X.
    The component origin is the lower-left of the poly bounding box.

    Parameters
    ----------
    w_um, l_um
        Drawn channel width and length (µm).
    device_type
        Logical device type. Must be a key under ``PDKMetadata.devices``.
    rules
        :class:`BootstrapRules` wrapping the PDK's metadata + rule DB +
        bootstrap mapping.
    n_fingers
        Override the computed finger count. Useful when the placer wants
        a specific count for matching constraints.
    skip_sd
        Set of S/D indices (0-based) to leave unpopulated — no contacts /
        no li1 strip on those S/D regions. Used by logic-gate generators
        where an internal node is connected only by shared diffusion (e.g.
        the middle S/D of a NAND stack).

    Returns
    -------
    gdsfactory.Component
        Component with named ports ``G`` (gate, top edge of finger 0),
        ``S`` (leftmost S/D, west-facing), ``D`` (rightmost S/D,
        east-facing).
    """
    import gdsfactory as gf

    # Ensure a PDK is active so layer tuples resolve correctly. The generic
    # PDK accepts arbitrary (layer, datatype) integer pairs. Prefer the
    # post-9.x ``gpdk`` import path; fall back to the deprecated location
    # so older gdsfactory installs keep working.
    try:
        gf.get_active_pdk()
    except ValueError:
        try:
            from gdsfactory.gpdk import get_generic_pdk
            get_generic_pdk().activate()
        except ImportError:
            from gdsfactory.generic_tech import PDK as _GENERIC
            _GENERIC.activate()

    geom = transistor_geom(w_um, l_um, device_type, rules)
    if n_fingers is not None and n_fingers != geom.n_fingers:
        from dataclasses import replace as _replace
        n   = max(1, int(n_fingers))
        w_f = w_um / n
        geom = _replace(
            geom,
            n_fingers    = n,
            w_finger_um  = w_f,
            total_x_um   = (n + 1) * geom.sd_length_um + n * l_um,
            total_y_um   = w_f + 2 * rules.get("poly.endcap_over_diff_um"),
            n_contacts_y = sd_contact_columns(w_f, rules),
        )

    dev = rules.device(device_type)

    # Unique cell name so repeated calls don't clash with kfactory.
    _base = f"{device_type}_W{w_um:.3f}_L{l_um:.3f}_f{geom.n_fingers}"
    _CELL_COUNTER[_base] = _CELL_COUNTER.get(_base, 0) + 1
    _n = _CELL_COUNTER[_base]
    _name = _base if _n == 1 else f"{_base}${_n}"
    c = gf.Component(name=_name)

    endcap  = rules.get("poly.endcap_over_diff_um")
    c_size  = rules.get("contacts.size_um")
    c_space = rules.get("contacts.spacing_um")
    c_enc   = rules.get("contacts.enclosure_in_diff_um")
    li_w    = rules.get("li1.width_min_um")

    lyr_diff    = rules.layer(dev["diff_layer"])
    lyr_gate    = rules.layer(dev["gate_layer"])
    lyr_contact = rules.layer("licon1")
    lyr_li1     = rules.layer("li1")
    lyr_implant = rules.layer(dev["implant_layer"])

    # ── Diffusion rectangle (covers all fingers in X; poly endcap in Y) ──
    diff_y0 = endcap
    diff_y1 = endcap + geom.w_finger_um
    diff_x0 = 0.0
    diff_x1 = geom.total_x_um
    c.add_polygon(
        [(diff_x0, diff_y0), (diff_x1, diff_y0),
         (diff_x1, diff_y1), (diff_x0, diff_y1)],
        layer=lyr_diff,
    )

    # ── Implant (S/D select — encloses diff) ─────────────────────────────
    impl_enc = rules.get("implant.enclosure_of_diff_um") if rules.has(
        "implant.enclosure_of_diff_um"
    ) else 0.0
    c.add_polygon(
        [(diff_x0 - impl_enc, diff_y0 - impl_enc),
         (diff_x1 + impl_enc, diff_y0 - impl_enc),
         (diff_x1 + impl_enc, diff_y1 + impl_enc),
         (diff_x0 - impl_enc, diff_y1 + impl_enc)],
        layer=lyr_implant,
    )

    # ── N-well (PMOS only; encloses diff with PDK enclosure rule) ────────
    if dev.get("nwell", False) and rules.has("nwell.enclosure_of_pdiff_um"):
        nw_enc = rules.get("nwell.enclosure_of_pdiff_um")
        c.add_polygon(
            [(diff_x0 - nw_enc, diff_y0 - nw_enc),
             (diff_x1 + nw_enc, diff_y0 - nw_enc),
             (diff_x1 + nw_enc, diff_y1 + nw_enc),
             (diff_x0 - nw_enc, diff_y1 + nw_enc)],
            layer=rules.layer("nwell"),
        )

    # ── Poly gate fingers ────────────────────────────────────────────────
    gate_port_x: list[float] = []
    for i in range(geom.n_fingers):
        gx0 = (i + 1) * geom.sd_length_um + i * geom.l_um
        gx1 = gx0 + geom.l_um
        c.add_polygon(
            [(gx0, 0.0), (gx1, 0.0),
             (gx1, geom.total_y_um), (gx0, geom.total_y_um)],
            layer=lyr_gate,
        )
        gate_port_x.append((gx0 + gx1) / 2)

    # ── NPC (Nitride Poly Cut) on poly endcaps ──────────────────────────
    # Prevents silicide on poly stubs extending beyond diffusion. Only
    # drawn if the PDK defines an ``npc`` layer.
    try:
        lyr_npc = rules.layer("npc")
        npc_enc = (
            rules.get("npc.enclosure_of_poly_um")
            if rules.has("npc.enclosure_of_poly_um") else 0.10
        )
        for i in range(geom.n_fingers):
            gx0 = (i + 1) * geom.sd_length_um + i * geom.l_um
            gx1 = gx0 + geom.l_um
            c.add_polygon(
                [(gx0 - npc_enc, 0.0 - npc_enc),
                 (gx1 + npc_enc, 0.0 - npc_enc),
                 (gx1 + npc_enc, diff_y0 + npc_enc),
                 (gx0 - npc_enc, diff_y0 + npc_enc)],
                layer=lyr_npc,
            )
            c.add_polygon(
                [(gx0 - npc_enc, diff_y1 - npc_enc),
                 (gx1 + npc_enc, diff_y1 - npc_enc),
                 (gx1 + npc_enc, geom.total_y_um + npc_enc),
                 (gx0 - npc_enc, geom.total_y_um + npc_enc)],
                layer=lyr_npc,
            )
    except KeyError:
        pass    # PDK doesn't define an npc layer — skip

    # ── Contacts + li1 rails per S/D region ─────────────────────────────
    n_cy  = geom.n_contacts_y
    c_mid = (diff_y0 + diff_y1) / 2
    total_span = n_cy * c_size + (n_cy - 1) * c_space
    cy_start   = c_mid - total_span / 2 + c_size / 2
    c_y_centres = [cy_start + k * (c_size + c_space) for k in range(n_cy)]

    sd_x_centres: list[float] = [
        j * (geom.sd_length_um + geom.l_um) + geom.sd_length_um / 2
        for j in range(geom.n_fingers + 1)
    ]

    enc_li_2adj, _enc_li_opp = rules.enclosure("contacts", "enclosure_in_li1")
    li1_half_w = max(c_size / 2, li_w / 2)

    source_li1_x0: float | None = None
    drain_li1_x0:  float | None = None
    _skip = skip_sd or set()

    for j, cx in enumerate(sd_x_centres):
        if j == 0:
            source_li1_x0 = cx
        if j == geom.n_fingers:
            drain_li1_x0 = cx

        if j in _skip:
            continue

        # Contact column.
        for cy in c_y_centres:
            c.add_polygon(
                [(cx - c_size / 2, cy - c_size / 2),
                 (cx + c_size / 2, cy - c_size / 2),
                 (cx + c_size / 2, cy + c_size / 2),
                 (cx - c_size / 2, cy + c_size / 2)],
                layer=lyr_contact,
            )

        # Li1 strip — li.5 asymmetric: enc_li_2adj on north+south.
        li_x0 = cx - li1_half_w
        li_x1 = cx + li1_half_w
        li_y0 = min(diff_y0, c_y_centres[0]  - c_size / 2 - enc_li_2adj)
        li_y1 = max(diff_y1, c_y_centres[-1] + c_size / 2 + enc_li_2adj)
        c.add_polygon(
            [(li_x0, li_y0), (li_x1, li_y0),
             (li_x1, li_y1), (li_x0, li_y1)],
            layer=lyr_li1,
        )

        if j == 0:
            source_li1_x0 = (li_x0 + li_x1) / 2
        if j == geom.n_fingers:
            drain_li1_x0 = (li_x0 + li_x1) / 2

    # ── Ports ───────────────────────────────────────────────────────────
    diff_y_mid = (diff_y0 + diff_y1) / 2
    c.add_port(
        name="G",
        center=(gate_port_x[0], geom.total_y_um),
        width=geom.l_um,
        orientation=90,
        layer=lyr_gate,
    )
    c.add_port(
        name="S",
        center=(source_li1_x0, diff_y_mid),
        width=geom.w_finger_um,
        orientation=180,
        layer=lyr_li1,
    )
    c.add_port(
        name="D",
        center=(drain_li1_x0, diff_y_mid),
        width=geom.w_finger_um,
        orientation=0,
        layer=lyr_li1,
    )

    return c
