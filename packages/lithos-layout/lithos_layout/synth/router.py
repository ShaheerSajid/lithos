"""lithos_layout.synth.router — routing style dispatch for synthesised cells.

Each routing style maps a :class:`RoutingSpec` (from the topology
template YAML — or, more commonly, emitted by the auto-router) to a
set of polygons drawn on the output :class:`gdsfactory.Component`.
Style handlers are registered by name and invoked by
:meth:`Router.route`.

Adding a new style
------------------

Define a function with the signature::

    def _my_style(
        comp:   gdsfactory.Component,
        spec:   RoutingSpec,
        placed: dict[str, PlacedDevice],
        rules:  BootstrapRules,
    ) -> list[PortCandidate]:
        ...

Then register it::

    register_style("my_style", _my_style)

Or use the decorator form at module level::

    @_style("my_style")
    def _my_style(...): ...

Currently registered
--------------------

* ``horizontal_power_rail`` — full-width VDD / GND rail (top or bottom).
* ``shared_gate_poly``     — vertical poly bridge tying NMOS and PMOS
  gates of the same net (e.g. inverter ``IN``).
* ``intra_device_sd``      — horizontal strap connecting all S or D
  fingers of a single multi-finger device.
* ``m0_bridge``            — narrow m0 strip between two S/D terminals
  at the same Y band (SRAM Q / Q_ and similar local taps).
* ``drain_bridge``         — N.D ↔ P.D bridge across the N-P gap, with
  per-finger stubs on m0 and a horizontal bus on ``spec.layer``.
* ``source_to_rail``       — source / drain strips bonded to a power
  rail via :func:`draw_via_stack` and an upper-layer strap.
* ``expose_terminal``      — exposes a device terminal as a port
  without drawing any routing geometry.
* ``gate_to_drain``        — same-row gate-to-drain route through the
  N-P gap: poly-contact stub on the gate side, horizontal m0 (or
  upper-layer) trunk to the drain X, then vertical down/up to the
  drain S/D centre. Handles AOI/OAI stage chaining.

Remaining style handlers (cross-couple-gate, vertical-bus,
cross-row-connect, poly-stub-m1-bus, vertical-m2-bus) land in
subsequent commits.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable

from lithos_layout.cells.standard import _rect as _rect_raw
from lithos_layout.rules          import BootstrapRules
from lithos_layout.stack          import via_stack_between
from lithos_layout.synth.loader   import RoutingSpec
from lithos_layout.synth.placer   import (
    PlacedDevice,
    global_diff_y,
    global_gate_x,
    global_poly_bottom,
    global_poly_top,
    global_sd_x,
    resolve_terminal,
)
from lithos_layout.synth.port_resolver import PortCandidate


# ── Drawing helpers ─────────────────────────────────────────────────────────

# Poly contacts already drawn this routing pass. Style handlers consult
# the map so a gate shared across the NMOS/PMOS rows gets only one
# poly-contact pad, not two. Cleared at the start of every
# :meth:`Router.route` call.
_drawn_poly_contacts: dict[tuple[float, str], tuple[float, float]] = {}


def _rect(comp: Any, x0: float, x1: float, y0: float, y1: float,
          layer: Any, snap_grid: float = 0.005) -> None:
    """Draw a rectangle (delegates to :func:`cells.standard._rect`)."""
    _rect_raw(comp, x0, x1, y0, y1, layer, snap_grid)


# ── Style registry ──────────────────────────────────────────────────────────

# Handler signature: (comp, spec, placed, rules) -> list of PortCandidate.
_Handler = Callable[
    [Any, RoutingSpec, dict[str, PlacedDevice], BootstrapRules],
    list[PortCandidate],
]
_REGISTRY: dict[str, _Handler] = {}


def register_style(name: str, fn: _Handler) -> None:
    """Register ``fn`` as the handler for routing style ``name``."""
    _REGISTRY[name] = fn


def _style(name: str):
    """Decorator: ``@_style("name")`` registers the decorated function."""
    def _dec(fn: _Handler) -> _Handler:
        _REGISTRY[name] = fn
        return fn
    return _dec


def registered_styles() -> list[str]:
    """Return the currently-registered style names (sorted)."""
    return sorted(_REGISTRY)


# ── Router ──────────────────────────────────────────────────────────────────

class Router:
    """Applies all routing specs from a template to a :class:`gdsfactory.Component`.

    Parameters
    ----------
    rules :
        Bootstrap rules.
    """

    def __init__(self, rules: BootstrapRules):
        self.rules = rules

    def route(
        self,
        comp:    Any,                                  # gdsfactory.Component
        routing: list[RoutingSpec],
        placed:  dict[str, PlacedDevice],
    ) -> list[PortCandidate]:
        """Route all specs and return collected port candidates."""
        _drawn_poly_contacts.clear()
        candidates: list[PortCandidate] = []
        for spec in routing:
            handler = _REGISTRY.get(spec.style)
            if handler is None:
                warnings.warn(
                    f"No handler registered for routing style "
                    f"{spec.style!r} (net={spec.net!r}); skipping.",
                    stacklevel=2,
                )
                continue
            result = handler(comp, spec, placed, self.rules)
            if result:
                candidates.extend(result)
        return candidates


# ── Geometry helpers ────────────────────────────────────────────────────────

def _collect_gate_poly_ranges(
    placed: dict[str, PlacedDevice],
) -> list[tuple[float, float, str]]:
    """Return ``(x0, x1, device_name)`` for every gate poly in the cell."""
    ranges: list[tuple[float, float, str]] = []
    for name, dev in placed.items():
        for j in range(dev.geom.n_fingers):
            gx0, gx1 = global_gate_x(dev, j)
            ranges.append((gx0, gx1, name))
    return ranges


def _nudge_for_poly_spacing(
    cx:              float,
    pad_half_x:      float,
    own_gate_range:  tuple[float, float],
    all_gate_ranges: list[tuple[float, float, str]],
    poly_sp:         float,
) -> float:
    """Shift contact centre ``cx`` so its poly pad keeps ``poly_sp``
    from every other gate. The pad still has to overlap ``own_gate_range``
    so it's electrically tied to its own gate.

    PDK-agnostic: works with any minimum poly spacing.
    """
    own_x0, own_x1 = own_gate_range
    eps = 0.005                                       # 5 nm extra clearance

    for gx0, gx1, _ in all_gate_ranges:
        # Skip the gate this contact belongs to.
        if abs(gx0 - own_x0) < 0.001 and abs(gx1 - own_x1) < 0.001:
            continue

        pad_left  = cx - pad_half_x
        pad_right = cx + pad_half_x

        if pad_right <= gx0:
            gap = gx0 - pad_right
            if gap < poly_sp:
                cx -= (poly_sp - gap + eps)
        elif pad_left >= gx1:
            gap = pad_left - gx1
            if gap < poly_sp:
                cx += (poly_sp - gap + eps)

    # Clamp: pad must still overlap its own gate for connectivity.
    cx = max(cx, own_x0 - pad_half_x + eps)
    cx = min(cx, own_x1 + pad_half_x - eps)
    return cx


def _power_rail_gap(rules: BootstrapRules) -> float:
    """Extra Y gap between a power rail and the transistor body when
    ``m0`` and ``m1`` collapse onto the same GDS layer (e.g. GF180).

    On such PDKs the device's m0 S/D strips and the m1 rail share a
    layer in DRC, so the natural gap (poly endcap) may be smaller than
    the m1 spacing rule. We add the shortfall plus a small margin.
    """
    if not getattr(rules, "m0_is_m1", False):
        return 0.0
    m1_sp  = rules.m1.get("spacing_min_um", 0.14) if hasattr(rules.m1, "get") \
             else rules.get("m1.spacing_min_um")
    endcap = rules.poly.get("endcap_over_diff_um", 0.0) if hasattr(rules.poly, "get") \
             else rules.get("poly.endcap_over_diff_um")
    return max(0.0, m1_sp - endcap) + 0.01           # 10 nm margin


def _min_area_half(rules: BootstrapRules, layer_name: str) -> float:
    """Return the half-extent for a square pad that meets ``layer_name``'s
    minimum-area rule, or ``0.0`` when no ``area_min_um2`` is mapped.
    """
    try:
        area_min = rules.section(layer_name).get("area_min_um2", 0.0) or 0.0
    except Exception:                                # pragma: no cover — defensive
        return 0.0
    if area_min <= 0:
        return 0.0
    return math.sqrt(area_min) / 2


# ── General-purpose via stack drawing ──────────────────────────────────────

def draw_via_stack(
    comp:       Any,
    rules:      BootstrapRules,
    cx:         float,
    cy:         float,
    from_layer: str,
    to_layer:   str,
    direction:  str = "horizontal",
) -> float:
    """Draw every via cut + metal landing needed to connect ``from_layer``
    to ``to_layer``.

    Uses :func:`lithos_layout.stack.via_stack_between` to decide which
    cuts to insert. ``direction`` controls which axis gets the larger
    2-adjacent-edge enclosure:

    * ``"horizontal"`` — 2adj on X, opposite on Y (route runs left/right).
    * ``"vertical"``   — 2adj on Y, opposite on X (route runs up/down).

    Only the *bottommost* (lower side of the first transition) and
    *topmost* (upper side of the last transition) landing pads use the
    direction. Intermediate landings are square (2adj on both axes) so
    no other route covers them.

    Returns the half-extent of the topmost landing pad — useful when
    the caller wants to extend a wire to overlap the via stack.
    Returns ``0.0`` when both layers resolve to the same stack position.
    """
    transitions = via_stack_between(rules, from_layer, to_layer)
    if not transitions:
        return 0.0

    vertical = direction == "vertical"
    n        = len(transitions)

    def _metal_w_min(metal: str) -> float:
        try:
            return rules.section(metal).get("width_min_um", 0.0) or 0.0
        except Exception:                            # pragma: no cover
            return 0.0

    def _dir_halves(enc_2adj: float, enc_opp: float, wmin: float, vh: float,
                    use_direction: bool) -> tuple[float, float]:
        """Return ``(hx, hy)`` for a landing pad.

        ``use_direction``: True → orient per caller's direction; False
        → square pad using 2adj on both axes (intermediate landings).
        """
        if not use_direction:
            h = max(vh + enc_2adj, wmin / 2)
            return h, h
        if vertical:
            hx = max(vh + enc_opp,  wmin / 2)
            hy = max(vh + enc_2adj, wmin / 2)
        else:
            hx = max(vh + enc_2adj, wmin / 2)
            hy = max(vh + enc_opp,  wmin / 2)
        return hx, hy

    top_half = 0.0
    for i, t in enumerate(transitions):
        vh        = t.via_size / 2
        is_first  = (i == 0)
        is_last   = (i == n - 1)

        lower_w   = _metal_w_min(t.lower_metal)
        upper_w   = _metal_w_min(t.upper_metal)

        lower_hx, lower_hy = _dir_halves(
            t.enc_lower, t.enc_lower_opp or t.enc_lower, lower_w, vh, is_first,
        )
        upper_hx, upper_hy = _dir_halves(
            t.enc_upper, t.enc_upper_opp or t.enc_upper, upper_w, vh, is_last,
        )

        lyr_via   = rules.layer(t.via_layer)
        lyr_lower = rules.layer(t.lower_metal)
        lyr_upper = rules.layer(t.upper_metal)

        _rect(comp, cx - vh,       cx + vh,       cy - vh,       cy + vh,       lyr_via)
        _rect(comp, cx - lower_hx, cx + lower_hx, cy - lower_hy, cy + lower_hy, lyr_lower)
        _rect(comp, cx - upper_hx, cx + upper_hx, cy - upper_hy, cy + upper_hy, lyr_upper)

        top_half = max(upper_hx, upper_hy)

    return top_half


# ── Style handlers ──────────────────────────────────────────────────────────

@_style("horizontal_power_rail")
def _horizontal_power_rail(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Full-width VDD / GND rail across the cell on ``spec.layer`` (default
    ``m1``).

    ``spec.edge`` selects the rail position:

    * ``"bottom"`` — GND rail below the lowest device, oriented south.
    * ``"top"``    — VDD rail above the highest device, oriented north.

    Alternatively, ``spec.extra["y_pos"]`` places an intermediate rail
    centred at the given Y (used by stacked multi-row cells).
    """
    if not placed:
        return []

    route_layer = spec.layer or "m1"
    try:
        rail_w_min = rules.section(route_layer).get("width_min_um", 0.0) or 0.0
    except Exception:                                # pragma: no cover
        rail_w_min = 0.0
    try:
        m0_w_min   = rules.section("m0").get("width_min_um", 0.0) or 0.0
    except Exception:                                # pragma: no cover
        m0_w_min   = 0.0
    rail_h = max(rail_w_min, m0_w_min, 0.14)         # 0.14 µm fallback
    lyr    = rules.layer(route_layer)

    # Bounding box of the placed devices.
    dev_x0    = min(d.x for d in placed.values())
    dev_x1    = max(d.x + d.geom.total_x_um for d in placed.values())
    cell_ytop = max(d.y + d.geom.total_y_um for d in placed.values())

    # Honour fixed-width cells (e.g. SRAM tile pitch).
    fixed_w = spec.extra.get("cell_width", 0) if spec.extra else 0
    if fixed_w > 0:
        dev_cx  = (dev_x0 + dev_x1) / 2
        cell_x0 = dev_cx - fixed_w / 2
        cell_x1 = dev_cx + fixed_w / 2
    else:
        cell_x0 = dev_x0
        cell_x1 = dev_x1
    cell_w  = cell_x1 - cell_x0
    cell_cx = (cell_x0 + cell_x1) / 2

    # ── Intermediate rail at explicit Y position ──────────────────
    y_pos = spec.extra.get("y_pos") if spec.extra else None
    if y_pos is not None:
        y_center = float(y_pos)
        y0, y1   = y_center, y_center + rail_h
        _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
        return [PortCandidate(
            net          = spec.net,
            location_key = f"rail_{spec.net}_{y_center:.3f}",
            x            = cell_cx,
            y            = (y0 + y1) / 2,
            layer        = route_layer,
            width        = cell_w,
            orientation  = 90,
        )]

    # ── Edge rails (top / bottom) ─────────────────────────────────
    edge = spec.edge or "bottom"
    gap  = _power_rail_gap(rules)

    if edge == "bottom":
        y0, y1 = -rail_h - gap, -gap
        _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
        return [PortCandidate(
            net          = spec.net,
            location_key = "bottom_rail_center",
            x            = cell_cx,
            y            = (y0 + y1) / 2,
            layer        = route_layer,
            width        = cell_w,
            orientation  = 270,
        )]

    # edge == "top"
    y0, y1 = cell_ytop + gap, cell_ytop + gap + rail_h
    _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
    return [PortCandidate(
        net          = spec.net,
        location_key = "top_rail_center",
        x            = cell_cx,
        y            = (y0 + y1) / 2,
        layer        = route_layer,
        width        = cell_w,
        orientation  = 90,
    )]


# ── shared_gate_poly ────────────────────────────────────────────────────────

@_style("shared_gate_poly")
def _shared_gate_poly(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Vertical poly bridge from the NMOS gate top to the PMOS gate bottom.

    Expected path: ``[N.G, P.G]``. For multi-finger devices, bridges
    *every* gate-finger pair and ties them together with a horizontal
    poly strap in the N-P gap.
    """
    if len(spec.path) < 2:
        return []
    n_name = spec.path[0].split(".")[0]
    p_name = spec.path[1].split(".")[0]
    dev_n  = placed[n_name]
    dev_p  = placed[p_name]

    lyr       = rules.layer("poly")
    n_fingers = max(dev_n.geom.n_fingers, dev_p.geom.n_fingers)
    y_bot     = global_poly_top(dev_n)
    y_top     = global_poly_bottom(dev_p)

    poly_w_min = rules.poly.get("width_min_um", 0.15) if hasattr(rules.poly, "get") \
                 else rules.get("poly.width_min_um")
    gap        = max(y_top - y_bot, poly_w_min)
    gate_mid_y = (y_bot + y_top) / 2
    strap_hy   = poly_w_min / 2

    leftmost_x0:  float | None = None
    rightmost_x1: float | None = None
    for i in range(n_fingers):
        if i < dev_n.geom.n_fingers:
            ngx0, ngx1 = global_gate_x(dev_n, i)
            if y_top > y_bot:
                _rect(comp, ngx0, ngx1, y_bot, y_top, lyr)
            if leftmost_x0 is None:
                leftmost_x0 = ngx0
            rightmost_x1 = ngx1
        if i < dev_p.geom.n_fingers:
            pgx0, pgx1 = global_gate_x(dev_p, i)
            if y_top > y_bot:
                _rect(comp, pgx0, pgx1, y_bot, y_top, lyr)
            if leftmost_x0 is None:
                leftmost_x0 = pgx0
            rightmost_x1 = max(rightmost_x1 or pgx1, pgx1)

    if n_fingers > 1 and leftmost_x0 is not None:
        _rect(
            comp, leftmost_x0, rightmost_x1,
            gate_mid_y - strap_hy, gate_mid_y + strap_hy, lyr,
        )

    gx0, _ = global_gate_x(dev_n, 0)
    return [
        PortCandidate(
            net=spec.net, location_key=f"{spec.net}_gate_left_edge_mid_y",
            x=gx0, y=gate_mid_y, layer="poly", width=gap, orientation=180,
        ),
        PortCandidate(
            net=spec.net, location_key="gate_left_edge_mid_y",
            x=gx0, y=gate_mid_y, layer="poly", width=gap, orientation=180,
        ),
    ]


# ── intra_device_sd ─────────────────────────────────────────────────────────

@_style("intra_device_sd")
def _intra_device_sd(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Tie all S (or all D) strips of one multi-finger device together.

    Expected path: ``[Dev.D]`` or ``[Dev.S]``. ``spec.extra["terminal"]``
    selects ``"D"`` (drain strips at odd j) or ``"S"`` (source strips at
    even j). The terminal index is flipped when the device's
    ``sd_flip`` is set.

    Draws a horizontal strap on ``spec.layer`` (default ``m0``)
    spanning every selected strip's X range, at the diffusion centre
    Y. When ``spec.layer`` is above ``m0``, drops a via stack at each
    strip so the upper-layer strap is bonded down to m0.
    """
    if not spec.path:
        return []

    dev_name  = spec.path[0].split(".")[0]
    term      = (spec.extra or {}).get("terminal", "D")
    dev       = placed[dev_name]
    n_fingers = dev.geom.n_fingers
    is_drain  = (term == "D")

    # j=0 → source, j=1 → drain, j=2 → source, …  (sd_flip reverses parity)
    sd_indices: list[int] = []
    for j in range(n_fingers + 1):
        j_is_drain = (j % 2 == 1) if not dev.spec.sd_flip else (j % 2 == 0)
        if j_is_drain == is_drain:
            sd_indices.append(j)
    if len(sd_indices) < 2:
        return []

    route_layer = spec.layer or "m0"
    lyr         = rules.layer(route_layer)
    try:
        route_w_min = rules.section(route_layer).get("width_min_um", 0.0) or 0.0
    except Exception:                                # pragma: no cover
        route_w_min = 0.0
    try:
        m0_w_min = rules.section("m0").get("width_min_um", 0.0) or 0.0
    except Exception:                                # pragma: no cover
        m0_w_min = 0.0
    route_w = route_w_min or m0_w_min or 0.17
    rhw     = route_w / 2

    dy0, dy1 = global_diff_y(dev, rules)
    d_cy     = (dy0 + dy1) / 2

    all_x0: list[float] = []
    all_x1: list[float] = []
    for j in sd_indices:
        sx0, sx1 = global_sd_x(dev, j, rules)
        all_x0.append(sx0)
        all_x1.append(sx1)
        # Bond every strip when the strap sits on a layer above m0.
        if route_layer != "m0":
            s_cx = (sx0 + sx1) / 2
            draw_via_stack(comp, rules, s_cx, d_cy, "m0", route_layer)

    strap_x0 = min(all_x0)
    strap_x1 = max(all_x1)
    _rect(comp, strap_x0, strap_x1, d_cy - rhw, d_cy + rhw, lyr)
    return []


# ── m0_bridge ───────────────────────────────────────────────────────────────

@_style("m0_bridge")
def _m0_bridge(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Horizontal m0 strip between two S/D terminals at the same Y band.

    Expected path: ``[DevA.Term, DevB.Term]``. Used for SRAM Q / Q_
    style local taps where two devices' source/drain strips need a
    short bridge but no via stack to a higher metal.
    """
    if len(spec.path) < 2:
        return []

    t0 = resolve_terminal(spec.path[0], placed, rules)
    t1 = resolve_terminal(spec.path[1], placed, rules)

    route_layer = spec.layer or "m0"
    lyr         = rules.layer(route_layer)
    try:
        route_w = rules.section(route_layer).get("width_min_um", 0.17) or 0.17
    except Exception:                                # pragma: no cover
        route_w = 0.17

    # Bridge spans the gap between the two terminals' inside edges.
    if t0.x1 < t1.x0:
        bridge_x0, bridge_x1 = t0.x1, t1.x0
    else:
        bridge_x0, bridge_x1 = t1.x1, t0.x0

    y_mid = (max(t0.y0, t1.y0) + min(t0.y1, t1.y1)) / 2
    y0    = y_mid - route_w / 2
    y1    = y_mid + route_w / 2

    if bridge_x1 > bridge_x0:
        _rect(comp, bridge_x0, bridge_x1, y0, y1, lyr)

    mid_x = (bridge_x0 + bridge_x1) / 2
    return [PortCandidate(
        net          = spec.net,
        location_key = f"{spec.net}_bridge_center",
        x            = mid_x,
        y            = y_mid,
        layer        = route_layer,
        width        = route_w,
        orientation  = 90,
    )]


# ── drain_bridge ────────────────────────────────────────────────────────────

@_style("drain_bridge")
def _drain_bridge(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Bridge NMOS drain(s) to PMOS drain(s) across the N-P gap.

    Expected path: ``[N.D, P.D]``. The bridge has three parts:

    * Vertical m0 stubs from each NMOS drain strip up into the gap.
    * Horizontal bus on ``spec.layer`` (default ``m0``) centred in the
      N-P gap, spanning the leftmost-to-rightmost drain X.
    * Vertical m0 stubs from the bus down to each PMOS drain strip.

    When ``spec.layer`` is above ``m0`` (e.g. ``m1`` for wider output
    bridges), a via stack is dropped at every drain so the upper-layer
    bus is bonded down to the S/D contact.
    """
    if len(spec.path) < 2:
        return []
    n_name = spec.path[0].split(".")[0]
    p_name = spec.path[1].split(".")[0]
    dev_n  = placed[n_name]
    dev_p  = placed[p_name]

    m0_w_min   = rules.m0.get("width_min_um", 0.17) or 0.17
    route_layer = spec.layer or "m0"
    lyr         = rules.layer(route_layer)
    try:
        route_w = rules.section(route_layer).get("width_min_um", m0_w_min) or m0_w_min
    except Exception:                                # pragma: no cover
        route_w = m0_w_min
    rhw = route_w / 2

    nd_y0, nd_y1 = global_diff_y(dev_n, rules)
    pd_y0, pd_y1 = global_diff_y(dev_p, rules)
    nd_cy        = (nd_y0 + nd_y1) / 2
    pd_cy        = (pd_y0 + pd_y1) / 2
    bus_y        = (nd_y1 + pd_y0) / 2

    def _drain_indices(dev: PlacedDevice) -> list[int]:
        out = []
        for j in range(dev.geom.n_fingers + 1):
            j_is_drain = (j % 2 == 0) if dev.spec.sd_flip else (j % 2 == 1)
            if j_is_drain:
                out.append(j)
        return out

    m0_is_m1 = getattr(rules, "m0_is_m1", False)
    m1_w_min = rules.m1.get("width_min_um", 0.14) or 0.14

    def _enforce_min_w(x0: float, x1: float) -> tuple[float, float]:
        """When m0 == m1 (collapsed PDK), the m0 strap is on the m1 layer
        in DRC and must hit m1's minimum width.
        """
        if not m0_is_m1:
            return x0, x1
        w = x1 - x0
        if w < m1_w_min:
            cx = (x0 + x1) / 2
            x0 = cx - m1_w_min / 2
            x1 = cx + m1_w_min / 2
        return x0, x1

    all_x0: list[float] = []
    all_x1: list[float] = []

    # NMOS stubs.
    for j in _drain_indices(dev_n):
        sx0, sx1 = global_sd_x(dev_n, j, rules)
        sx0, sx1 = _enforce_min_w(sx0, sx1)
        all_x0.append(sx0)
        all_x1.append(sx1)
        _rect(comp, sx0, sx1, nd_y1, bus_y + rhw, lyr)
        if route_layer != "m0":
            draw_via_stack(comp, rules, (sx0 + sx1) / 2, nd_cy,
                           "m0", route_layer, direction="vertical")

    # PMOS stubs.
    for j in _drain_indices(dev_p):
        sx0, sx1 = global_sd_x(dev_p, j, rules)
        sx0, sx1 = _enforce_min_w(sx0, sx1)
        all_x0.append(sx0)
        all_x1.append(sx1)
        _rect(comp, sx0, sx1, bus_y - rhw, pd_y0, lyr)
        if route_layer != "m0":
            draw_via_stack(comp, rules, (sx0 + sx1) / 2, pd_cy,
                           "m0", route_layer, direction="vertical")

    # Horizontal bus spanning every drain.
    if all_x0:
        _rect(comp, min(all_x0), max(all_x1),
              bus_y - rhw, bus_y + rhw, lyr)

    bridge_height = max(pd_y0 - nd_y1, dev_n.geom.l_um)
    rightmost_x   = max(all_x1) if all_x1 else 0.0
    return [PortCandidate(
        net          = spec.net,
        location_key = "drain_bridge_right_edge_mid_y",
        x            = rightmost_x,
        y            = bus_y,
        layer        = route_layer,
        width        = bridge_height,
        orientation  = 0,
    )]


# ── source_to_rail ──────────────────────────────────────────────────────────

@_style("source_to_rail")
def _source_to_rail(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Connect source / drain terminals to a power rail.

    Expected path: ``[Dev.S, Dev.S, …]`` (or ``Dev.D`` for layouts where
    the rail-side terminal is the drain).  ``spec.edge`` picks the rail:

    * ``"bottom"`` — GND rail just below the lowest device.
    * ``"top"``    — VDD rail just above the highest device.

    For each terminal strip the handler either:

    * draws an m0 strap from the strip's diffusion edge to the rail Y
      boundary when ``spec.layer == "m0"`` (no via stack needed); or
    * drops a via stack at the strip's diffusion centre and draws a
      ``spec.layer`` strap from the via to the rail Y boundary
      otherwise.
    """
    if not spec.path:
        return []

    edge       = spec.edge or "bottom"
    rail_layer = spec.layer or "m1"

    m0_w_min = rules.m0.get("width_min_um", 0.17) or 0.17
    try:
        rail_w_min = rules.section(rail_layer).get("width_min_um", 0.14) or 0.14
    except Exception:                                # pragma: no cover
        rail_w_min = 0.14
    rail_h = max(rail_w_min, m0_w_min)

    lyr_m0 = rules.layer("m0")

    gap = _power_rail_gap(rules)
    if edge == "bottom":
        rail_y0, rail_y1 = -rail_h - gap, -gap
    else:
        cell_ytop = max(d.y + d.geom.total_y_um for d in placed.values())
        rail_y0, rail_y1 = cell_ytop + gap, cell_ytop + gap + rail_h

    for ref in spec.path:
        parts = ref.split(".", 1)
        if len(parts) != 2:
            continue
        dev_name, term = parts
        dev = placed.get(dev_name)
        if dev is None:
            continue

        # j=0 → source, j=1 → drain, … (sd_flip reverses parity).
        n_fingers = dev.geom.n_fingers
        is_source = (term == "S")
        sd_indices: list[int] = []
        for j in range(n_fingers + 1):
            j_is_source = (j % 2 == 0) if not dev.spec.sd_flip else (j % 2 == 1)
            if j_is_source == is_source:
                sd_indices.append(j)

        dy0, dy1 = global_diff_y(dev, rules)
        d_cy     = (dy0 + dy1) / 2

        for j in sd_indices:
            sx0, sx1 = global_sd_x(dev, j, rules)
            tx_mid   = (sx0 + sx1) / 2
            m0_hx    = max(m0_w_min / 2, (sx1 - sx0) / 2)

            if rail_layer != "m0":
                # Bond down to m0 at the strip centre, then strap on the
                # rail layer from via to rail edge.
                draw_via_stack(comp, rules, tx_mid, d_cy,
                               "m0", rail_layer, direction="vertical")
                lyr_rail = rules.layer(rail_layer)
                rail_hx  = max(rail_w_min / 2, m0_hx)
                if edge == "bottom":
                    _rect(comp, tx_mid - rail_hx, tx_mid + rail_hx,
                          rail_y0, d_cy, lyr_rail)
                else:
                    _rect(comp, tx_mid - rail_hx, tx_mid + rail_hx,
                          d_cy, rail_y1, lyr_rail)
            else:
                # m0-only path: single m0 strap from terminal edge to rail.
                if edge == "bottom":
                    _rect(comp, tx_mid - m0_hx, tx_mid + m0_hx,
                          rail_y0, dy1, lyr_m0)
                else:
                    _rect(comp, tx_mid - m0_hx, tx_mid + m0_hx,
                          dy0, rail_y1, lyr_m0)

    return []


# ── expose_terminal ─────────────────────────────────────────────────────────

@_style("expose_terminal")
def _expose_terminal(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Expose a device terminal as a port without drawing any routing.

    Use this to make terminals that are not connected internally
    (e.g. SRAM BL / BL_, or an inverter's IN that only drives gates)
    reachable from outside the cell.

    Expected path: ``[Dev.Terminal]`` (single element).

    Extra fields
    ------------
    orientation : int
        Port orientation in degrees (default ``90`` = north).
    location_key : str
        Explicit location key for the emitted candidate. Defaults to
        ``"<dev>_<term>_center"``.
    """
    if not spec.path:
        return []

    try:
        t = resolve_terminal(spec.path[0], placed, rules)
    except (KeyError, ValueError) as exc:
        warnings.warn(
            f"expose_terminal (net={spec.net!r}): {exc}; skipped.",
            stacklevel=3,
        )
        return []

    mid_x = (t.x0 + t.x1) / 2
    mid_y = (t.y0 + t.y1) / 2

    orientation = int((spec.extra or {}).get("orientation", 90))
    if orientation in (90, 270):
        width = t.x1 - t.x0
    else:
        width = t.y1 - t.y0

    m0_w_min = rules.m0.get("width_min_um", 0.17) or 0.17
    width    = max(width, m0_w_min)

    location_key = (spec.extra or {}).get(
        "location_key",
        f"{spec.path[0].replace('.', '_')}_center",
    )

    return [PortCandidate(
        net          = spec.net,
        location_key = location_key,
        x            = mid_x,
        y            = mid_y,
        layer        = t.layer,
        width        = width,
        orientation  = orientation,
    )]


# ── gate_to_drain ───────────────────────────────────────────────────────────

@_style("gate_to_drain")
def _gate_to_drain(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> list[PortCandidate]:
    """Same-row gate ↔ drain route through the N-P gap.

    Expected path: ``[Dev_A.G, Dev_B.D]``. Draws an explicit poly
    contact (cut + poly pad + m0 pad) at the gate's poly endcap and
    runs an m0 (or upper-layer) trunk from there to the drain S/D
    centre, then vertically into the drain diffusion. Used by AOI21 /
    OAI21 / cross-couple chains where one stage's output drives the
    next stage's gate without going through a higher metal.

    ``spec.layer`` defaults to ``m0``. When set higher (e.g. ``m1``
    for crossing-avoidance), a via stack is dropped at each end so
    the trunk lives on the requested layer.
    """
    if len(spec.path) < 2:
        return []

    gate_name  = spec.path[0].split(".")[0]
    drain_name = spec.path[1].split(".")[0]
    gate_dev   = placed[gate_name]
    drain_dev  = placed[drain_name]
    is_nmos_gate = gate_dev.spec.device_type == "nmos"

    # ── Gate poly position ───────────────────────────────────────────
    gx0, gx1 = global_gate_x(gate_dev, 0)
    gate_cx  = (gx0 + gx1) / 2

    # ── Drain S/D position ───────────────────────────────────────────
    j_d = 0 if drain_dev.spec.sd_flip else 1
    dx0, dx1 = global_sd_x(drain_dev, j_d, rules)
    drain_cx = (dx0 + dx1) / 2

    # ── Contact sizing ──────────────────────────────────────────────
    c_size        = rules.contact["size_um"]
    ch            = c_size / 2
    poly_enc      = rules.contact.get("enclosure_in_poly_um", 0.05)
    poly_enc_2adj = rules.contact.get("enclosure_in_poly_2adj_um", poly_enc)

    m0_enc_2adj = rules.m0.get("enclosure_of_contact_2adj_um", 0.08)
    m0_enc      = rules.m0.get("enclosure_of_contact_um", 0.0)
    m0_w_min    = rules.m0.get("width_min_um", 0.17)
    m0_sp       = rules.m0.get("spacing_min_um", 0.17)

    # ── Contact Y: just above/below the poly endcap ──────────────────
    pc_half_y = ch + poly_enc_2adj
    if is_nmos_gate:
        pc_y = global_poly_top(gate_dev) + pc_half_y
    else:
        pc_y = global_poly_bottom(gate_dev) - pc_half_y

    # ── Contact X: shifted toward the connecting drain ──────────────
    # The poly contact sits between two S/D strips. The m0 pad must
    # clear the strip on the far side from the drain so the trunk
    # doesn't short an adjacent net's S/D.
    m0_enc_route = max(ch + m0_enc_2adj, m0_w_min / 2)
    m0_enc_far   = max(ch + m0_enc_2adj, m0_w_min / 2)
    m0_hy_val    = max(ch + m0_enc,      m0_w_min / 2)

    pc_x = gate_cx
    drain_is_left = drain_cx < gate_cx

    if drain_is_left:
        m0_hx_left  = m0_enc_route   # toward drain
        m0_hx_right = m0_enc_far     # away from drain
    else:
        m0_hx_left  = m0_enc_far
        m0_hx_right = m0_enc_route

    # Shift contact toward the drain until the far-side m0 edge
    # clears the nearest opposite-side S/D strip.
    n_fingers = gate_dev.geom.n_fingers
    for j in range(n_fingers + 1):
        sdx0, sdx1 = global_sd_x(gate_dev, j, rules)
        sd_cx = (sdx0 + sdx1) / 2
        if drain_is_left and sd_cx > gate_cx:
            max_pc_x = sdx0 - m0_sp - m0_hx_right
            pc_x = min(pc_x, max_pc_x)
            break
        if not drain_is_left and sd_cx < gate_cx:
            min_pc_x = sdx1 + m0_sp + m0_hx_left
            pc_x = max(pc_x, min_pc_x)
            break

    # Snap contact centre to manufacturing grid.
    _grid = rules.mfg_grid
    if _grid > 0:
        pc_x = round(round(pc_x / _grid) * _grid, 6)
        pc_y = round(round(pc_y / _grid) * _grid, 6)

    # ── Draw poly contact (skip if another route already laid one) ──
    contact_key = (round(gate_cx, 4), "poly_contact")
    prev = _drawn_poly_contacts.get(contact_key)

    lyr_poly    = rules.layer("poly")
    lyr_contact = rules.layer("contact")
    lyr_m0      = rules.layer("m0")

    poly_pad_hx = ch + poly_enc
    poly_pad_hy = ch + poly_enc_2adj

    if prev is not None:
        pc_x, pc_y = prev
    else:
        _drawn_poly_contacts[contact_key] = (pc_x, pc_y)
        # 1. Contact cut
        _rect(comp, pc_x - ch, pc_x + ch, pc_y - ch, pc_y + ch, lyr_contact)
        # 2. Poly pad
        _rect(comp, pc_x - poly_pad_hx, pc_x + poly_pad_hx,
                    pc_y - poly_pad_hy, pc_y + poly_pad_hy, lyr_poly)
        # 3. m0 pad — asymmetric: 2adj toward route, min elsewhere
        _rect(comp, pc_x - m0_hx_left, pc_x + m0_hx_right,
                    pc_y - m0_hy_val,  pc_y + m0_hy_val, lyr_m0)

    # 4. Gate poly stub: connect transistor poly edge to the contact pad
    if is_nmos_gate:
        poly_top = global_poly_top(gate_dev)
        _rect(comp, gx0, gx1, poly_top, pc_y + poly_pad_hy, lyr_poly)
    else:
        poly_bot = global_poly_bottom(gate_dev)
        _rect(comp, gx0, gx1, pc_y - poly_pad_hy, poly_bot, lyr_poly)

    # ── Trunk on spec.layer ──────────────────────────────────────────
    route_layer = spec.layer or "m0"
    lyr_route   = rules.layer(route_layer)
    try:
        route_w = rules.section(route_layer).get("width_min_um", 0.0) or m0_w_min
    except Exception:                                # pragma: no cover
        route_w = m0_w_min
    rhw = route_w / 2

    route_y      = pc_y
    dd_y0, dd_y1 = global_diff_y(drain_dev, rules)
    dd_cy        = (dd_y0 + dd_y1) / 2

    # Bond contact-side pad to the trunk layer when routing above m0.
    if route_layer != "m0":
        draw_via_stack(comp, rules, pc_x, pc_y, "m0", route_layer)

    # Horizontal: contact pad → drain X
    _rect(comp,
          min(pc_x - m0_hx_left,  drain_cx - rhw),
          max(pc_x + m0_hx_right, drain_cx + rhw),
          route_y - rhw, route_y + rhw, lyr_route)

    # Vertical: trunk → drain S/D centre
    if is_nmos_gate:
        _rect(comp, drain_cx - rhw, drain_cx + rhw,
                    dd_cy, route_y + rhw, lyr_route)
    else:
        _rect(comp, drain_cx - rhw, drain_cx + rhw,
                    route_y - rhw, dd_cy, lyr_route)

    # Bond trunk back down at the drain when routing above m0.
    if route_layer != "m0":
        draw_via_stack(comp, rules, drain_cx, dd_cy,
                       route_layer, "m0", direction="vertical")

    return []
