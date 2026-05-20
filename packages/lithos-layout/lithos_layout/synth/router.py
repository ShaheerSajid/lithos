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

Initial slice
-------------

This is the *first* slice of the router port. Currently registered:

* ``horizontal_power_rail`` — full-width VDD / GND rail (top or bottom).

Remaining style handlers (shared-gate-poly, drain-bridge,
intra-device-sd, source-to-rail, m0-bridge, cross-couple, vertical
bus, expose-terminal, …) land in subsequent commits. The Router
itself, the registry, the geometry helpers, and the general
:func:`draw_via_stack` primitive are wired up so each handler can
land independently.
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
    global_gate_x,
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
