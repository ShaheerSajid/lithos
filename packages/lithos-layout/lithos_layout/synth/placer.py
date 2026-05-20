"""lithos_layout.synth.placer — floorplan constraint → placed devices.

Given a :class:`CellTemplate` and :class:`BootstrapRules`, the placer:

1. Computes :class:`TransistorGeom` for every device from the supplied
   params (with per-device template overrides).
2. Evaluates named constraints (e.g. ``inter_cell_gap``).
3. Resolves placement directives, stacked row pairs, or legacy
   pair-mode placement into global ``(x, y)`` offsets.
4. Snaps everything to the PDK manufacturing grid.

Global coordinate system (matches :func:`draw_transistor`):

- Origin ``(0, 0)`` = lower-left of the lowest device's poly bounding box.
- X axis = channel length direction.
- Y axis = channel width direction.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace as _replace
from typing import Any, Callable

from lithos_layout.cells.standard import _diff_y, _gate_x, _sd_x
from lithos_layout.rules          import BootstrapRules
from lithos_layout.transistor     import (
    TransistorGeom,
    _min_channel_width,
    sd_contact_columns,
    transistor_geom,
)
from lithos_layout.synth.constraints import eval_expr, resolve_named_constraints
from lithos_layout.synth.loader      import (
    CellTemplate,
    DeviceSpec,
)


# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_W: dict[str, float] = {"nmos": 0.52, "pmos": 0.52}


# ── Named spacing rules ─────────────────────────────────────────────────────
#
# Each callable takes BootstrapRules and returns a float in µm. The
# YAML author writes ``spacing_rule: cross_couple_wiring`` and the
# engine looks the value up automatically.

def _spacing_min_diff(r: BootstrapRules) -> float:
    """Minimum diffusion-to-diffusion spacing."""
    return r.diff["spacing_min_um"]


def _spacing_inter_cell_gap(r: BootstrapRules) -> float:
    """Gap between NMOS and PMOS tiers — the larger of:

    * the diff-to-diff spacing rule (after subtracting both rows' poly
      endcap overhangs);
    * the N-diff to N-well clearance (which depends on how far the well
      must extend past the P-diff).

    Falls back to ``diff.spacing_min_um`` when the well-related rules
    aren't in the bootstrap mapping.
    """
    endcap = r.poly["endcap_over_diff_um"]
    gap_diff = r.diff["spacing_min_um"] - 2 * endcap
    nw_enc   = r.nwell.get("enclosure_of_pdiff_um", 0.0) or 0.0
    nw_min_w = r.nwell.get("width_min_um", 0.0)         or 0.0
    ndiff_to_nwell = r.nwell.get("ndiff_to_nwell_um", 0.0) or 0.0
    gap_nwell = ndiff_to_nwell - 2 * endcap + max(nw_enc, nw_min_w / 2)
    return max(0.0, gap_diff, gap_nwell)


def _spacing_cross_couple_wiring(r: BootstrapRules) -> float:
    """Cross-couple gap: room for m0 wires AND nwell separation."""
    endcap   = r.poly["endcap_over_diff_um"]
    m0_room  = (2 * r.m0["spacing_min_um"] + r.m0["width_min_um"]
                - 2 * endcap)
    nw_room  = r.nwell["spacing_min_um"] - 2 * r.nwell["enclosure_of_pdiff_um"]
    return max(m0_room, nw_room)


def _spacing_min_well_separation(r: BootstrapRules) -> float:
    """Minimum nwell-to-nwell edge-to-edge separation."""
    return r.nwell["spacing_min_um"] - 2 * r.nwell["enclosure_of_pdiff_um"]


SPACING_RULES: dict[str, Callable[[BootstrapRules], float]] = {
    "min_diff_spacing":    _spacing_min_diff,
    "inter_cell_gap":      _spacing_inter_cell_gap,
    "cross_couple_wiring": _spacing_cross_couple_wiring,
    "min_well_separation": _spacing_min_well_separation,
}


def resolve_spacing_rule(name: str, rules: BootstrapRules) -> float:
    """Look up a named spacing rule and return the value in µm.

    Raises :class:`KeyError` if the name is not registered.
    """
    fn = SPACING_RULES.get(name)
    if fn is None:
        raise KeyError(
            f"Unknown spacing_rule {name!r}. "
            f"Available: {sorted(SPACING_RULES)}"
        )
    return fn(rules)


# ── PlacedDevice ────────────────────────────────────────────────────────────

@dataclass
class PlacedDevice:
    """A device placed at a specific origin in global cell coordinates.

    Attributes
    ----------
    name :
        Device instance name (e.g. ``"N"``, ``"P"``).
    spec :
        Original :class:`DeviceSpec` from the template.
    geom :
        Computed transistor geometry.
    x, y :
        Global origin offset. Pass to ``ref.move((x, y))`` when adding
        the drawn :class:`gdsfactory.Component` to the cell.
    component :
        The drawn :class:`gdsfactory.Component`. Filled in by the
        synthesizer after calling :func:`draw_transistor`; ``None``
        until then.
    """
    name:      str
    spec:      DeviceSpec
    geom:      TransistorGeom
    x:         float
    y:         float
    component: Any = None     # gf.Component (avoid gf import at module load)


# ── Terminal geometry ───────────────────────────────────────────────────────

@dataclass
class TerminalGeom:
    """Global bounding box of a device terminal (G, D, or S)."""
    dev_name: str
    terminal: str         # "G", "D", or "S"
    x0:       float
    x1:       float
    y0:       float
    y1:       float
    layer:    str         # logical layer name: ``"poly"`` or ``"m0"``


def resolve_terminal(
    ref:    str,
    placed: dict[str, PlacedDevice],
    rules:  BootstrapRules,
) -> TerminalGeom:
    """Resolve a ``"DeviceName.Terminal"`` reference to global geometry.

    Parameters
    ----------
    ref :
        Terminal reference string (``"N.G"``, ``"P.D"``, …).
    placed :
        Map of device name → :class:`PlacedDevice`.
    rules :
        Bootstrap rules (needed for diffusion Y extent).

    Returns
    -------
    TerminalGeom
    """
    parts = ref.split(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid terminal reference {ref!r} (expected 'Dev.Term')"
        )
    dev_name, term = parts
    if dev_name not in placed:
        raise KeyError(f"Device {dev_name!r} not found in placed devices")

    dev  = placed[dev_name]
    geom = dev.geom

    # S/D flip: when a device is flipped for diffusion sharing, swap the
    # physical positions of S and D (left ↔ right).
    phys_term = term
    if dev.spec.sd_flip and term in ("S", "D"):
        phys_term = "D" if term == "S" else "S"

    if phys_term == "G":
        lx0, lx1 = _gate_x(0, geom)
        return TerminalGeom(
            dev_name, term,
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=dev.y,       y1=dev.y + geom.total_y_um,
            layer="poly",
        )
    if phys_term == "D":
        j = geom.n_fingers              # rightmost S/D = drain for finger 0
        lx0, lx1 = _sd_x(j, geom, rules)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, term,
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=ly0 + dev.y, y1=ly1 + dev.y,
            layer="m0",
        )
    if phys_term == "S":
        lx0, lx1 = _sd_x(0, geom, rules)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, term,
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=ly0 + dev.y, y1=ly1 + dev.y,
            layer="m0",
        )
    raise ValueError(f"Unknown terminal {term!r} in reference {ref!r}")


# ── Global geometry helpers ─────────────────────────────────────────────────

def global_gate_x(dev: PlacedDevice, finger: int = 0) -> tuple[float, float]:
    """Global ``(x0, x1)`` of the ``finger``-th gate poly finger."""
    lx0, lx1 = _gate_x(finger, dev.geom)
    return lx0 + dev.x, lx1 + dev.x


def global_sd_x(
    dev:   PlacedDevice,
    j:     int,
    rules: BootstrapRules | None = None,
) -> tuple[float, float]:
    """Global ``(x0, x1)`` of the ``j``-th source/drain region.

    When ``rules`` is provided, contact-width m0 pullback is applied
    (matching the geometry produced by :func:`draw_transistor`).
    """
    lx0, lx1 = _sd_x(j, dev.geom, rules)
    return lx0 + dev.x, lx1 + dev.x


def global_diff_y(dev: PlacedDevice, rules: BootstrapRules) -> tuple[float, float]:
    """Global ``(y0, y1)`` of the diffusion region."""
    ly0, ly1 = _diff_y(dev.geom, rules)
    return ly0 + dev.y, ly1 + dev.y


def global_poly_top(dev: PlacedDevice) -> float:
    """Global Y of the poly top edge."""
    return dev.y + dev.geom.total_y_um


def global_poly_bottom(dev: PlacedDevice) -> float:
    """Global Y of the poly bottom edge (= device Y origin)."""
    return dev.y


# ── Placer ──────────────────────────────────────────────────────────────────

class Placer:
    """Resolves device placements from a :class:`CellTemplate`.

    Parameters
    ----------
    rules :
        Bootstrap rules.
    params :
        Device sizing. Keys:

        * ``"w_<DevName>"`` (µm) — per-device width override.
        * ``"l_<DevName>"`` (µm) — per-device length override.
        * ``"w"`` / ``"l"`` — fallback width / length for all devices.

        Example: ``{"w_n": 0.52, "w_p": 0.42, "l": 0.15}``.
    """

    def __init__(
        self,
        rules:  BootstrapRules,
        params: dict[str, Any] | None = None,
    ):
        self.rules  = rules
        self.params = {k.lower(): v for k, v in (params or {}).items()}
        self._device_params: dict[str, dict[str, Any]] = {}

    def place(self, template: CellTemplate) -> dict[str, PlacedDevice]:
        """Resolve all device positions and return the placement map."""
        self._device_params = template.device_params

        # Pass 1: transistor geometries.
        geoms = self._compute_geoms(template)

        # Pass 2: scalar named constraints.
        named = resolve_named_constraints(
            template.named_constraints, self.rules, geoms,
        )

        # Pass 3: place devices.
        if template.placement_directives:
            placed = self._place_from_directives(template, geoms, named)
        elif template.layout_mode == "stacked":
            placed = self._place_stacked(template, geoms, named)
        else:
            placed = self._place_devices(template, geoms, named)

        # Pass 4: centre devices within fixed cell width, if requested.
        if template.cell_dimensions.width > 0:
            _apply_cell_width(placed, template.cell_dimensions.width)

        # Pass 5: snap to manufacturing grid.
        grid = self.rules.mfg_grid
        if grid > 0:
            for dev in placed.values():
                dev.x = round(round(dev.x / grid) * grid, 6)
                dev.y = round(round(dev.y / grid) * grid, 6)

        return placed

    # ── Sizing ──────────────────────────────────────────────────────────

    def _w(self, dev_name: str, dev_type: str) -> float:
        dev_ovr = self._device_params.get(dev_name, {})
        if "w" in dev_ovr:
            return float(dev_ovr["w"])
        p = self.params
        return (
            p.get(f"w_{dev_name.lower()}")
            or p.get("w")
            or _DEFAULT_W.get(dev_type, 0.52)
        )

    def _l(self, dev_name: str) -> float:
        dev_ovr = self._device_params.get(dev_name, {})
        if "l" in dev_ovr:
            return float(dev_ovr["l"])
        default_l = self.rules.poly["width_min_um"]
        return (
            self.params.get(f"l_{dev_name.lower()}")
            or self.params.get("l")
            or default_l
        )

    def _compute_geoms(self, template: CellTemplate) -> dict[str, TransistorGeom]:
        result: dict[str, TransistorGeom] = {}
        for name, spec in template.devices.items():
            geom = transistor_geom(
                self._w(name, spec.device_type),
                self._l(name),
                spec.device_type,
                self.rules,
            )
            n = int(spec.fingers)
            if n > 0 and n != geom.n_fingers:
                w_f   = geom.w_um / n
                w_min = _min_channel_width(self.rules, spec.device_type)
                if w_min > 0 and w_f < w_min:
                    max_n = max(1, int(geom.w_um / w_min))
                    warnings.warn(
                        f"{name}: fingers={n} gives w_finger={w_f:.3f}µm "
                        f"< min channel width {w_min:.3f}µm; "
                        f"clamping to {max_n} fingers",
                        stacklevel=3,
                    )
                    n   = max_n
                    w_f = geom.w_um / n
                endcap = self.rules.poly["endcap_over_diff_um"]
                geom = _replace(
                    geom,
                    n_fingers    = n,
                    w_finger_um  = w_f,
                    total_x_um   = (n + 1) * geom.sd_length_um + n * geom.l_um,
                    total_y_um   = w_f + 2 * endcap,
                    n_contacts_y = sd_contact_columns(w_f, self.rules),
                )
            result[name] = geom
        return result

    # ── Placement modes ─────────────────────────────────────────────────

    def _place_devices(
        self,
        template: CellTemplate,
        geoms:    dict[str, TransistorGeom],
        named:    dict[str, float],
    ) -> dict[str, PlacedDevice]:
        placed: dict[str, PlacedDevice] = {}

        for dev_name in _topo_order(template.devices):
            spec = template.devices[dev_name]
            geom = geoms[dev_name]

            placed_offsets: dict[str, float] = {}
            for pname, pd in placed.items():
                placed_offsets[f"{pname}_x"] = pd.x
                placed_offsets[f"{pname}_y"] = pd.y

            full_named = {**named, **placed_offsets}
            x = _resolve_x(spec, placed, geom, self.rules, geoms, full_named)
            y = eval_expr(
                spec.y_offset_expr,
                self.rules,
                geoms,
                named = full_named,
            )

            placed[dev_name] = PlacedDevice(
                name = dev_name, spec = spec, geom = geom, x = x, y = y,
            )

        return placed

    def _place_from_directives(
        self,
        template: CellTemplate,
        geoms:    dict[str, TransistorGeom],
        named:    dict[str, float],
    ) -> dict[str, PlacedDevice]:
        """Place devices using explicit :class:`PlacementDirective` list.

        Each directive specifies exactly how to position a device
        relative to another (or at an absolute origin). The directive
        order defines the dependency-evaluation order.
        """
        placed: dict[str, PlacedDevice] = {}
        rules = self.rules

        for d in template.placement_directives:
            if d.name not in template.devices:
                warnings.warn(
                    f"PlacementDirective references unknown device "
                    f"{d.name!r}; skipped.",
                    stacklevel=3,
                )
                continue

            spec = template.devices[d.name]
            # ``MY`` and ``R180`` orientations imply S/D flip.
            if d.sd_flip or d.orientation in ("MY", "R180"):
                spec.sd_flip = True
            geom = geoms[d.name]

            # ── Absolute origin (first device) ─────────────────────────
            if d.origin is not None and not d.relative_to:
                x, y = d.origin
                placed[d.name] = PlacedDevice(
                    name=d.name, spec=spec, geom=geom, x=x, y=y,
                )
                continue

            # ── Relative placement ─────────────────────────────────────
            if d.relative_to not in placed:
                warnings.warn(
                    f"PlacementDirective for {d.name!r}: relative_to "
                    f"{d.relative_to!r} not yet placed; placing at origin.",
                    stacklevel=3,
                )
                placed[d.name] = PlacedDevice(
                    name=d.name, spec=spec, geom=geom, x=0.0, y=0.0,
                )
                continue

            anchor      = placed[d.relative_to]
            anchor_geom = geoms[d.relative_to]

            # ── X ──────────────────────────────────────────────────────
            if d.relation == "abut_x":
                # Shared-diffusion abutment: overlap one S/D region.
                x = anchor.x + anchor_geom.total_x_um - anchor_geom.sd_length_um
            elif d.relation == "space_x":
                gap = 0.0
                if d.spacing_rule:
                    gap = resolve_spacing_rule(d.spacing_rule, rules)
                elif "cross_gap" in named:
                    gap = named["cross_gap"]
                else:
                    gap = rules.diff["spacing_min_um"]
                x = anchor.x + anchor_geom.total_x_um + gap
            elif d.relation == "align_gate":
                x = anchor.x
            elif d.relation == "mirror_x":
                x = anchor.x + anchor_geom.total_x_um
            else:
                x = anchor.x + anchor_geom.total_x_um + rules.diff["spacing_min_um"]

            # ── Y ──────────────────────────────────────────────────────
            if d.alignment == "gate" or d.relation == "align_gate":
                if spec.device_type != template.devices[d.relative_to].device_type:
                    icg = named.get(
                        "inter_cell_gap",
                        _spacing_inter_cell_gap(rules),
                    )
                    y = anchor.y + anchor_geom.total_y_um + icg
                else:
                    y = anchor.y
            elif d.alignment == "top":
                y = anchor.y + anchor_geom.total_y_um - geom.total_y_um
            elif d.alignment == "center":
                y = anchor.y + (anchor_geom.total_y_um - geom.total_y_um) / 2
            else:                                # "bottom" (default)
                y = anchor.y

            placed[d.name] = PlacedDevice(
                name=d.name, spec=spec, geom=geom, x=x, y=y,
            )

        return placed

    def _place_stacked(
        self,
        template: CellTemplate,
        geoms:    dict[str, TransistorGeom],
        named:    dict[str, float],
    ) -> dict[str, PlacedDevice]:
        """Place devices in a vertically-stacked multi-row layout.

        Each :class:`RowPairSpec` produces an NMOS tier (bottom) and a
        PMOS tier (top), stacked vertically. Devices within a tier are
        placed left-to-right with shared-diffusion abutment.
        """
        placed: dict[str, PlacedDevice] = {}

        icg     = named.get("inter_cell_gap", _spacing_inter_cell_gap(self.rules))
        diff_sp = self.rules.diff["spacing_min_um"]
        inter_row_gap = named.get("inter_row_gap", diff_sp)

        current_y = 0.0

        for rp_idx, rp in enumerate(template.row_pairs):
            n_list = [(n, geoms[n]) for n in rp.nmos_devices if n in geoms]
            p_list = [(n, geoms[n]) for n in rp.pmos_devices if n in geoms]

            nmos_h = max((g.total_y_um for _, g in n_list), default=0.0)

            # ── NMOS tier ──────────────────────────────────────────────
            x = 0.0
            for i, (name, geom) in enumerate(n_list):
                if i > 0:
                    x -= geom.sd_length_um           # shared-diff abutment
                placed[name] = PlacedDevice(
                    name=name, spec=template.devices[name], geom=geom,
                    x=x, y=current_y,
                )
                x += geom.total_x_um

            # ── PMOS tier ──────────────────────────────────────────────
            pmos_y = current_y + nmos_h + icg if n_list else current_y

            x = 0.0
            for i, (name, geom) in enumerate(p_list):
                if i > 0:
                    x -= geom.sd_length_um
                placed[name] = PlacedDevice(
                    name=name, spec=template.devices[name], geom=geom,
                    x=x, y=pmos_y,
                )
                x += geom.total_x_um

            # ── Advance Y for next row pair ───────────────────────────
            pmos_h = max((g.total_y_um for _, g in p_list), default=0.0)
            if n_list and p_list:
                row_top = pmos_y + pmos_h
            elif n_list:
                row_top = current_y + nmos_h
            else:
                row_top = pmos_y + pmos_h

            if rp_idx < len(template.row_pairs) - 1:
                current_y = row_top + inter_row_gap

        return placed


# ── Helpers ─────────────────────────────────────────────────────────────────

def _topo_order(devices: dict[str, DeviceSpec]) -> list[str]:
    """Return device names sorted so each device's dependencies come first.

    Topologically sorts on the dependency graph induced by symbolic
    ``DEV_x`` / ``DEV_y`` / ``DEV.attr`` references in each device's
    :attr:`DeviceSpec.x_spec` or :attr:`y_offset_expr`. Within one
    dependency level, devices whose ``region`` contains ``"bottom"``
    come before ``"top"``.
    """
    all_names = list(devices)
    deps: dict[str, set[str]] = {n: set() for n in all_names}

    for name, spec in devices.items():
        for expr in (spec.x_spec, spec.y_offset_expr):
            if not isinstance(expr, str):
                continue
            for other in all_names:
                if other == name:
                    continue
                if re.search(rf"\b{re.escape(other)}[_.]", expr):
                    deps[name].add(other)

    # Kahn's algorithm.
    in_degree = {n: len(deps[n]) for n in all_names}
    queue     = [n for n in all_names if in_degree[n] == 0]

    def _tier(n: str) -> int:
        return 1 if "top" in devices[n].region.lower() else 0

    queue.sort(key=_tier)

    result: list[str] = []
    while queue:
        n = queue.pop(0)
        result.append(n)
        for other in all_names:
            if n in deps[other]:
                deps[other].discard(n)
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)
                    queue.sort(key=_tier)

    # Defensive: append anything left over (circular deps; shouldn't happen).
    for n in all_names:
        if n not in result:
            result.append(n)

    return result


def _resolve_x(
    spec:   DeviceSpec,
    placed: dict[str, PlacedDevice],
    geom:   TransistorGeom,
    rules:  BootstrapRules,
    geoms:  dict[str, TransistorGeom] | None = None,
    named:  dict[str, float]          | None = None,
) -> float:
    """Resolve the X origin for a device from its floorplan ``x_spec``.

    Forms accepted:

    * ``None`` or ``"left"`` → X = 0.
    * numeric → used directly.
    * ``"right_of: DEV"`` / ``"right_of(DEV)"`` → right edge of ``DEV``
      plus ``diff.spacing_min_um``.
    * ``"between(DEV_A, DEV_B)"`` → centred between ``DEV_A``'s right
      edge and ``DEV_B``'s left edge.
    * anything else → evaluated as a constraint expression via
      :func:`eval_expr` with the same namespace as ``y_offset_expr``
      (so ``rules.*``, device geom attributes, and placed-offset
      shortcuts like ``PD_L_x`` are all in scope).
    """
    x = spec.x_spec
    if x is None or x == "left":
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)

    xs = str(x).strip()

    m = re.match(r"right_of\s*[:(]\s*(\w+)\s*\)?", xs, re.IGNORECASE)
    if m:
        ref = m.group(1)
        if ref in placed:
            p  = placed[ref]
            sp = rules.diff["spacing_min_um"]
            return p.x + p.geom.total_x_um + sp

    m = re.match(r"between\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", xs, re.IGNORECASE)
    if m:
        ra, rb = m.group(1), m.group(2)
        if ra in placed and rb in placed:
            pa, pb = placed[ra], placed[rb]
            gap = pb.x - (pa.x + pa.geom.total_x_um)
            return pa.x + pa.geom.total_x_um + (gap - geom.total_x_um) / 2

    if geoms is not None:
        try:
            return eval_expr(xs, rules, geoms, named=named)
        except ValueError:
            pass

    warnings.warn(
        f"Cannot resolve x_spec {xs!r}; defaulting to 0.0", stacklevel=4,
    )
    return 0.0


def _apply_cell_width(
    placed:       dict[str, PlacedDevice],
    target_width: float,
) -> None:
    """Centre all placed devices within a fixed cell width.

    Shifts every device by the same X offset so the active area is
    centred within ``target_width``. Internal device-to-device spacing
    is unchanged.
    """
    if not placed:
        return
    x_min        = min(d.x for d in placed.values())
    x_max        = max(d.x + d.geom.total_x_um for d in placed.values())
    actual_width = x_max - x_min
    if actual_width >= target_width:
        return                                       # already wider — no shift
    offset = (target_width - actual_width) / 2 - x_min
    for dev in placed.values():
        dev.x += offset
