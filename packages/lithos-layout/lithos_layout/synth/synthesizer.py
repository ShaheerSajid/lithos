"""lithos_layout.synth.synthesizer — top-level template → GDS orchestrator.

The :class:`Synthesizer` glues every other piece of the cell-generation
pipeline together:

1. :class:`Placer` resolves the template's placement directives →
   global ``(x, y)`` offsets for every device.
2. :func:`draw_transistor` emits each device as a sub-Component;
   :class:`Synthesizer` instantiates them at their placed origins.
3. :func:`_merge_implants` + :func:`_merge_nwells` fuse per-device
   implant / N-well boxes into the per-row regions a real cell needs
   (single-device boxes routinely violate implant / N-well spacing).
4. :class:`Router` applies the caller-supplied
   :class:`RoutingSpec` list — power rails, gate ties, drain bridges,
   source-to-rail straps, etc.
5. :func:`resolve_ports` places gdsfactory ports on the cell's
   bounding-box edges using the candidates the router emitted.

What this skeleton does **not** do (deferred to follow-up commits):

* **DRC iteration** — runs once, returns. No fix loop, no ML model,
  no geometric-fix agent.
* **LVS** — no connectivity verification.
* **GDS labels / well taps / per-row labels** — defer until the
  Magic / netgen LVS pipeline is wired in.

The auto-router runs by default when the caller omits
``routing_specs``; callers can still hand-roll a spec list to override
the planner (useful for tests or hand-tuned cells).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Any

from lithos_layout.rules                import BootstrapRules
from lithos_layout.transistor           import draw_transistor
from lithos_layout.synth.auto_router    import AutoRouter
from lithos_layout.synth.loader         import CellTemplate, RoutingSpec
from lithos_layout.synth.netlist        import build_net_graph
from lithos_layout.synth.placer         import Placer, PlacedDevice
from lithos_layout.synth.port_resolver  import generate_expose_specs, resolve_ports
from lithos_layout.synth.router         import Router


@dataclass
class SynthResult:
    """Result of one :meth:`Synthesizer.synthesize` call.

    Attributes
    ----------
    component :
        Final :class:`gdsfactory.Component` (no DRC verification —
        the skeleton does not iterate to clean).
    placed :
        Device name → :class:`PlacedDevice` map.
    params :
        Parameter dict used for placement / sizing.
    iterations :
        Always ``1`` for the skeleton (no DRC loop yet).
    candidates :
        Port candidates emitted by the router style handlers. Useful
        for downstream tooling that wants to re-resolve ports against
        a different policy.
    """
    component:   Any
    placed:      dict[str, PlacedDevice]
    params:      dict[str, Any]
    iterations:  int = 1
    candidates:  list = field(default_factory=list)


class Synthesizer:
    """Drive the placer → router → port-resolver pipeline.

    Parameters
    ----------
    rules :
        Bootstrap rules.
    """

    def __init__(self, rules: BootstrapRules):
        self.rules = rules

    def synthesize(
        self,
        template:      CellTemplate,
        params:        dict[str, Any]      | None = None,
        routing_specs: list[RoutingSpec]   | None = None,
        *,
        component_name: str | None = None,
    ) -> SynthResult:
        """Build a fully-instantiated cell from a template.

        Parameters
        ----------
        template :
            Cell topology template (from :func:`load_template`).
        params :
            Device sizing keyed as ``"w_<dev_name>"`` / ``"l_<dev_name>"``
            (lower-case) with ``"w"`` / ``"l"`` falling back as defaults.
        routing_specs :
            Routing specs to feed the :class:`Router`. When
            ``None``, :class:`AutoRouter` generates the specs from the
            :class:`NetGraph` + placement. Pass an explicit list to
            override (e.g. for hand-tuned cells or testing).
            ``expose_terminal`` specs for ports that name a specific
            terminal are added automatically via
            :func:`generate_expose_specs`.
        component_name :
            Optional override for the :class:`gdsfactory.Component`
            name. Defaults to ``template.name``.

        Returns
        -------
        SynthResult
        """
        _activate_pdk()

        # ── Placement ─────────────────────────────────────────────────
        placer = Placer(self.rules, params)
        placed = placer.place(template)

        import gdsfactory as gf
        # gdsfactory's kfactory backend enforces a globally-unique cell
        # name. Pass a name only when the caller asked for one; otherwise
        # let gdsfactory auto-generate, which is unique per Component
        # instance and survives back-to-back synthesis calls in tests.
        comp = gf.Component(name=component_name) if component_name else gf.Component()

        skip_map = _compute_skip_sd(template, placed)

        for dev in placed.values():
            tc = draw_transistor(
                w_um         = dev.geom.w_um,
                l_um         = dev.geom.l_um,
                device_type  = dev.spec.device_type,
                rules        = self.rules,
                n_fingers    = dev.geom.n_fingers,
                skip_sd      = skip_map.get(dev.name),
            )
            ref = comp.add_ref(tc)
            ref.move((dev.x, dev.y))
            dev.component = tc

        # ── Per-row implant / nwell merging ──────────────────────────
        _merge_implants(comp, placed, self.rules)
        _merge_nwells(comp, placed, self.rules)

        # ── Routing ──────────────────────────────────────────────────
        net_graph = build_net_graph(template)
        if routing_specs is None:
            specs = AutoRouter(self.rules).plan(net_graph, placed, template)
        else:
            specs = list(routing_specs)
        specs.extend(generate_expose_specs(template, net_graph, placed))

        router = Router(self.rules)
        candidates = router.route(comp, specs, placed)

        # ── Ports ────────────────────────────────────────────────────
        resolve_ports(
            comp, template, net_graph, placed, candidates, self.rules,
        )

        return SynthResult(
            component  = comp,
            placed     = placed,
            params     = dict(params or {}),
            candidates = candidates,
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _activate_pdk() -> None:
    """Ensure gdsfactory has an active PDK so Components can be drawn."""
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
        return
    except ValueError:
        pass
    try:
        from gdsfactory.gpdk import get_generic_pdk
        get_generic_pdk().activate()
    except ImportError:                              # pragma: no cover
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()


def _compute_skip_sd(
    template: CellTemplate,
    placed:   dict[str, PlacedDevice],
) -> dict[str, set[int]]:
    """Decide which S/D indices on abutted devices can skip contacts.

    At an ``abut_x`` boundary the two devices share one diffusion
    region. If the shared net is *internal* (not power, not a port),
    the contacts + m0 strip at that index are unnecessary — the
    abutting diffusion already conducts. Skipping them keeps the cell
    DRC-clean on PDKs where m0 collapses onto m1.

    Returns a ``{device_name: {sd_index, …}}`` map for
    :func:`draw_transistor` to consult via its ``skip_sd`` arg.
    """
    skip: dict[str, set[int]] = {}
    if not template.placement_directives:
        return skip

    port_nets   = set(template.ports.keys()) if template.ports else set()
    power_nets  = {
        n for n, info in (template.nets or {}).items()
        if info.net_type == "power"
    }
    needs_metal = power_nets | port_nets

    def _terminal_at(dev: PlacedDevice, j: int) -> str:
        is_drain = (j % 2 == 1)
        if dev.spec.sd_flip:
            is_drain = not is_drain
        return "D" if is_drain else "S"

    for d in template.placement_directives:
        if d.relation != "abut_x" or not d.relative_to:
            continue
        dev    = placed.get(d.name)
        anchor = placed.get(d.relative_to)
        if dev is None or anchor is None:
            continue
        if dev.spec.device_type != anchor.spec.device_type:
            continue

        anchor_j   = anchor.geom.n_fingers              # rightmost S/D
        dev_j      = 0                                  # leftmost S/D
        anchor_net = template.devices[d.relative_to].terminals.get(
            _terminal_at(anchor, anchor_j), "")
        dev_net    = template.devices[d.name].terminals.get(
            _terminal_at(dev, dev_j), "")

        if (anchor_net and anchor_net == dev_net
                and anchor_net not in needs_metal):
            skip.setdefault(d.relative_to, set()).add(anchor_j)
            skip.setdefault(d.name,        set()).add(dev_j)
    return skip


def _merge_implants(
    comp:   Any,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> None:
    """Draw one merged implant box per cluster of same-type devices.

    Each PMOS / NMOS device's per-device bounding implant box is
    expanded by ``implant.enclosure_of_diff_um`` and clustered on Y
    when the inter-box gap is below ``implant.spacing_min_um``. Each
    cluster is emitted as a single rectangle on the device-type's
    ``implant_layer``.
    """
    from lithos_layout.cells.standard import _diff_y

    try:
        impl_enc = rules.section("implant").get("enclosure_of_diff_um", 0.0) or 0.0
        impl_sp  = rules.section("implant").get("spacing_min_um", 0.0)        or 0.0
    except Exception:                                # pragma: no cover
        impl_enc = 0.0
        impl_sp  = 0.0

    layer_devboxes: dict[str, list[tuple[PlacedDevice, tuple]]] = {}
    for dev in placed.values():
        try:
            impl_layer = rules.device(dev.spec.device_type)["implant_layer"]
        except (KeyError, AttributeError):
            continue
        dy0, dy1 = _diff_y(dev.geom, rules)
        bbox = (
            dev.x - impl_enc,
            dev.x + dev.geom.total_x_um + impl_enc,
            dy0 + dev.y - impl_enc,
            dy1 + dev.y + impl_enc,
        )
        layer_devboxes.setdefault(impl_layer, []).append((dev, bbox))

    for impl_layer_name, devboxes in layer_devboxes.items():
        if len(devboxes) < 2:
            continue
        lyr = rules.layer(impl_layer_name)
        devboxes.sort(key=lambda db: db[1][2])

        clusters: list[list[tuple]] = [[devboxes[0][1]]]
        for _, bbox in devboxes[1:]:
            prev_y1 = max(b[3] for b in clusters[-1])
            if bbox[2] - prev_y1 < impl_sp:
                clusters[-1].append(bbox)
            else:
                clusters.append([bbox])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            x0 = min(b[0] for b in cluster)
            x1 = max(b[1] for b in cluster)
            y0 = min(b[2] for b in cluster)
            y1 = max(b[3] for b in cluster)
            comp.add_polygon(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                layer = lyr,
            )


def _merge_nwells(
    comp:   Any,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> None:
    """Draw one merged N-well rectangle per cluster of nearby PMOS devices.

    Standalone PMOS transistors draw their own N-well; adjacent N-wells
    closer than ``nwell.spacing_min_um`` violate the spacing rule, so
    we fuse them into one rectangle wide enough to also satisfy
    ``nwell.width_min_um``.
    """
    from lithos_layout.cells.standard import _diff_y

    try:
        nw_enc   = rules.nwell.get("enclosure_of_pdiff_um", 0.0) or 0.0
        nw_sp    = rules.nwell.get("spacing_min_um", 0.0)        or 0.0
        nw_min_w = rules.nwell.get("width_min_um", 0.0)          or 0.0
    except Exception:                                # pragma: no cover
        nw_enc = nw_sp = nw_min_w = 0.0

    pmos_devs = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if not pmos_devs:
        return

    boxes: list[tuple[float, float, float, float]] = []
    for dev in pmos_devs:
        dy0, dy1 = _diff_y(dev.geom, rules)
        boxes.append((
            dev.x - nw_enc,
            dev.x + dev.geom.total_x_um + nw_enc,
            dy0 + dev.y - nw_enc,
            dy1 + dev.y + nw_enc,
        ))
    boxes.sort(key=lambda b: b[2])

    clusters: list[list[tuple]] = [[boxes[0]]]
    for box in boxes[1:]:
        prev_y1 = max(b[3] for b in clusters[-1])
        if box[2] - prev_y1 < nw_sp:
            clusters[-1].append(box)
        else:
            clusters.append([box])

    try:
        lyr = rules.layer("nwell")
    except KeyError:                                 # pragma: no cover
        return

    for cluster in clusters:
        x0 = min(b[0] for b in cluster)
        x1 = max(b[1] for b in cluster)
        y0 = min(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if x1 - x0 < nw_min_w:
            x0, x1 = cx - nw_min_w / 2, cx + nw_min_w / 2
        if y1 - y0 < nw_min_w:
            y0, y1 = cy - nw_min_w / 2, cy + nw_min_w / 2
        comp.add_polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            layer = lyr,
        )
