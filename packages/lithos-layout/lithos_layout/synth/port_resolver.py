"""lithos_layout.synth.port_resolver — compass-side port placement.

Resolves each :class:`PortSpec` in a template to a concrete ``(x, y)``
on the cell's bounding-box edge. Three resolution paths, in order of
preference:

1. **Explicit terminal** — the YAML names ``terminal: N.G``. We
   look up that terminal's :class:`TerminalGeom` via
   :func:`resolve_terminal` and place the port at its centre with the
   terminal's logical layer.
2. **Routing candidate match** — a routing style handler emitted a
   :class:`PortCandidate` for this net. We pick the candidate closest
   to the requested compass side.
3. **Fallback** — for signal nets with gate connections we use the
   first gate terminal as a last-resort anchor.

Also generates ``expose_terminal`` :class:`RoutingSpec` entries via
:func:`generate_expose_specs` for ports that name a specific terminal
(consumed by the auto-router's "phase E").
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing      import Any

from lithos_layout.rules            import BootstrapRules
from lithos_layout.synth.loader     import CellTemplate, RoutingSpec
from lithos_layout.synth.netlist    import NetGraph
from lithos_layout.synth.placer     import PlacedDevice, resolve_terminal


# ── PortCandidate (co-located so port_resolver doesn't import the router) ──

@dataclass
class PortCandidate:
    """Port location hint emitted by a routing style handler.

    The synthesizer matches ``location_key`` against each port's
    ``PortSpec.location`` to know where to place the output port.

    Lives in this module (rather than the router) so the placer and
    the port resolver can share it without depending on the much
    larger router module.
    """
    net:          str
    location_key: str           # matches the port's ``location:`` field in YAML
    x:            float
    y:            float
    layer:        str           # logical layer name (``"m0"``, ``"m1"``, …)
    width:        float
    orientation:  int           # gdsfactory port orientation (degrees)


# ── Side → orientation mapping ──────────────────────────────────────────────

_SIDE_ORIENTATION: dict[str, int] = {
    "west":   180,
    "east":   0,
    "north":  90,
    "south":  270,
    "left":   180,
    "right":  0,
    "top":    90,
    "bottom": 270,
}


# ── Port resolver ───────────────────────────────────────────────────────────

def resolve_ports(
    comp:       Any,                # gf.Component
    template:   CellTemplate,
    net_graph:  NetGraph,
    placed:     dict[str, PlacedDevice],
    candidates: list[PortCandidate],
    rules:      BootstrapRules,
) -> None:
    """Add ports to ``comp`` according to the template's compass-side specs.

    Falls back to matching routing candidates by net name when the
    compass-side resolution doesn't find a direct terminal match.
    """
    if not template.ports:
        return

    # Cell bounding box (used to score candidates on side proximity).
    cell_x0 = min(d.x for d in placed.values())
    cell_x1 = max(d.x + d.geom.total_x_um for d in placed.values())
    cell_y0 = min(d.y for d in placed.values())
    cell_y1 = max(d.y + d.geom.total_y_um for d in placed.values())

    cand_by_net: dict[str, list[PortCandidate]] = {}
    for c in candidates:
        cand_by_net.setdefault(c.net, []).append(c)

    for port_name, pspec in template.ports.items():
        side          = pspec.side
        orientation   = _SIDE_ORIENTATION.get(side, 0)
        terminal_ref  = pspec.terminal

        x = y = width = 0.0
        layer: str = ""

        net_info = net_graph.nets.get(port_name)

        # ── Path 1: explicit terminal ─────────────────────────────────
        if terminal_ref:
            try:
                t = resolve_terminal(terminal_ref, placed, rules)
                x = (t.x0 + t.x1) / 2
                y = (t.y0 + t.y1) / 2
                layer = t.layer
                width = (t.x1 - t.x0) if orientation in (90, 270) else (t.y1 - t.y0)
            except (KeyError, ValueError) as exc:
                warnings.warn(
                    f"Port {port_name!r}: cannot resolve terminal "
                    f"{terminal_ref!r}: {exc}. Skipped.",
                    stacklevel=3,
                )
                continue

        # ── Path 2a: power-net rail candidate ─────────────────────────
        elif net_info and net_info.is_power:
            rail_cands = cand_by_net.get(port_name, [])
            if not rail_cands:
                continue
            c = rail_cands[0]
            x, y, layer, width, orientation = c.x, c.y, c.layer, c.width, c.orientation

        # ── Path 2b: signal net with gate connections ─────────────────
        elif net_info and net_info.gate_terminals:
            gate_cands = cand_by_net.get(port_name, [])
            if gate_cands:
                c = _best_candidate_for_side(
                    gate_cands, side, cell_x0, cell_x1, cell_y0, cell_y1,
                )
                x, y, layer, width = c.x, c.y, c.layer, c.width
            else:
                # Last resort: anchor on the first gate terminal.
                gt = net_info.gate_terminals[0]
                try:
                    t = resolve_terminal(gt.ref, placed, rules)
                    x     = (t.x0 + t.x1) / 2
                    y     = (t.y0 + t.y1) / 2
                    layer = t.layer
                    width = (
                        (t.y1 - t.y0) if orientation in (0, 180)
                        else (t.x1 - t.x0)
                    )
                except (KeyError, ValueError):
                    continue

        # ── Path 3: any candidate matching by net name ────────────────
        else:
            net_cands = cand_by_net.get(port_name, [])
            if not net_cands:
                warnings.warn(
                    f"Port {port_name!r}: no candidate or terminal found. Skipped.",
                    stacklevel=3,
                )
                continue
            c = _best_candidate_for_side(
                net_cands, side, cell_x0, cell_x1, cell_y0, cell_y1,
            )
            x, y, layer, width = c.x, c.y, c.layer, c.width

        # Enforce minimum port width (use m0 minimum as the fallback).
        try:
            min_w = rules.m0["width_min_um"]
        except (KeyError, AttributeError):
            min_w = 0.0
        width = max(width, min_w)

        # Explicit port layer from YAML overrides the auto-detected one.
        if pspec.layer:
            layer = pspec.layer

        # Logical layer name (e.g. "m0") survives gdsfactory's LayerEnum
        # integer mapping so downstream code (label placement, LVS) can
        # recover the matching pin layer from the PDK.
        layer_name = layer if isinstance(layer, str) else ""
        try:
            lyr = rules.layer(layer) if isinstance(layer, str) else layer
        except (KeyError, TypeError):
            lyr = (1, 0)

        new_port = comp.add_port(
            port_name,
            center      = (x, y),
            width       = width,
            orientation = orientation,
            layer       = lyr,
        )
        if layer_name and new_port is not None:
            try:
                new_port.info["layer_name"] = layer_name
            except Exception:                        # pragma: no cover — defensive
                pass


def generate_expose_specs(
    template:  CellTemplate,
    net_graph: NetGraph,
    placed:    dict[str, PlacedDevice],
) -> list[RoutingSpec]:
    """Generate ``expose_terminal`` :class:`RoutingSpec` entries.

    These drive the auto-router's terminal-exposure phase: every port
    that names a specific terminal in the template ends up here so the
    router knows to bring that terminal out to the cell edge.
    """
    specs: list[RoutingSpec] = []
    if not template.ports:
        return specs

    for port_name, pspec in template.ports.items():
        if not pspec.terminal:
            continue
        orientation = _SIDE_ORIENTATION.get(pspec.side, 0)
        specs.append(RoutingSpec(
            net   = port_name,
            style = "expose_terminal",
            layer = "m0",
            path  = [pspec.terminal],
            extra = {
                "orientation":  orientation,
                "location_key": f"{port_name}_port",
            },
        ))
    return specs


# ── Helpers ─────────────────────────────────────────────────────────────────

def _best_candidate_for_side(
    cands:   list[PortCandidate],
    side:    str,
    cell_x0: float,
    cell_x1: float,
    cell_y0: float,
    cell_y1: float,
) -> PortCandidate:
    """Pick the candidate closest to the requested cell edge."""
    if len(cands) == 1:
        return cands[0]

    def _dist(c: PortCandidate) -> float:
        if side in ("west", "left"):
            return abs(c.x - cell_x0)
        if side in ("east", "right"):
            return abs(c.x - cell_x1)
        if side in ("north", "top"):
            return abs(c.y - cell_y1)
        if side in ("south", "bottom"):
            return abs(c.y - cell_y0)
        return 0.0

    return min(cands, key=_dist)
