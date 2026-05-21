"""lithos_repair.actions — typed polygon-level repair verbs.

Each verb is a pure function with signature::

    apply(comp: gf.Component, ref: PolygonRef, params: <Params>)
        -> tuple[gf.Component, PolygonRef]

returning a **new** Component (gdsfactory components are not meant to be
mutated after they're added to the workspace) plus an updated
:class:`~lithos_repair.features.PolygonRef` pointing at where the
transformed polygon ended up. Verbs that delete the polygon return the
removed polygon's old ref.

The v1 vocabulary is axis-aligned-rectangle-friendly:

* :func:`widen` / :func:`narrow` — symmetric expand/contract along an axis.
* :func:`shift_n` / :func:`shift_s` / :func:`shift_e` / :func:`shift_w` —
  translate by a positive ``delta_um`` in one cardinal direction.
* :func:`extend` / :func:`shrink` — push a single edge outward / inward.
* :func:`snap_to_grid` — round every vertex to the nearest grid pitch.
* :func:`remove` — delete the polygon (inverse needs the polygon's data).
* :func:`redraw` — add a polygon (inverse is :func:`remove`).

Inverses are declared on :class:`~lithos_repair.registry.ActionDef` so
the registry can compose a round-trip without each verb knowing about
its sibling.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .features import LayerTuple, Polygon, PolygonRef, polygon_ref


# ── gdsfactory <-> polygon-list bridge ────────────────────────────────

def _activate_pdk() -> None:
    """Activate gdsfactory's generic PDK if no PDK is active.

    Repair actions need to add polygons to fresh Components, which
    requires an active PDK. We use the generic PDK so this module has
    no per-PDK coupling — the repair layer is PDK-agnostic.
    """
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
        return
    except ValueError:
        pass
    try:
        from gdsfactory.gpdk import get_generic_pdk
        get_generic_pdk().activate()
    except ImportError:                                  # pragma: no cover
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()


def extract_polygons(comp: Any) -> list[Polygon]:
    """Return every polygon in ``comp`` as a :class:`Polygon` (µm units).

    Crosses the gdsfactory / KLayout boundary: ``comp.get_polygons(by="tuple")``
    keys by ``(gds_layer, datatype)`` and yields KLayout polygons in
    integer database units. We map back to µm using the component's
    ``kcl.dbu``.
    """
    _activate_pdk()
    dbu = float(comp.kcl.dbu)
    out: list[Polygon] = []
    for layer_tuple, polys in comp.get_polygons(by="tuple").items():
        for kp in polys:
            dpoly = kp.to_dtype(dbu)
            pts = tuple((p.x, p.y) for p in dpoly.each_point_hull())
            out.append(Polygon(layer=layer_tuple, points=pts))
    return out


def rebuild_component(polygons: list[Polygon], *, name: Optional[str] = None) -> Any:
    """Build a fresh ``gf.Component`` containing exactly ``polygons``.

    Ports / sub-cell references are intentionally not preserved — the
    repair verbs operate on flattened geometry. Callers that need port
    metadata back can re-resolve after repair.
    """
    _activate_pdk()
    import gdsfactory as gf
    comp = gf.Component(name=name) if name else gf.Component()
    for p in polygons:
        comp.add_polygon(list(p.points), layer=p.layer)
    return comp


def find_polygon(
    polygons: list[Polygon],
    ref:      PolygonRef,
    *,
    tol_um:   float = 0.005,
) -> int:
    """Return the index of the polygon matching ``ref``.

    ``tol_um`` defaults to one mfg-grid pitch (0.005 µm), which absorbs
    the tiny round-trip drift that ``rebuild_component`` introduces
    (KLayout snaps to integer dbu on store).
    """
    for i, p in enumerate(polygons):
        if p.layer != ref.layer:
            continue
        cx, cy = p.centroid
        if abs(cx - ref.centroid_x) < tol_um and abs(cy - ref.centroid_y) < tol_um:
            return i
    raise ValueError(
        f"No polygon matches ref={ref!r} (tol={tol_um} µm). "
        f"Candidates on layer {ref.layer}: "
        f"{[p.centroid for p in polygons if p.layer == ref.layer]}"
    )


# ── helpers ─────────────────────────────────────────────────────────────

def _rect_points(
    x0: float, y0: float, x1: float, y1: float,
) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _replace(polygons: list[Polygon], i: int, new: Polygon) -> Polygon:
    polygons[i] = new
    return new


# ── param models ─────────────────────────────────────────────────────────

class WidenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis:     Literal["x", "y"]
    delta_um: float = Field(gt=0)


class NarrowParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis:     Literal["x", "y"]
    delta_um: float = Field(gt=0)


class ShiftParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    delta_um: float = Field(gt=0)


class EdgeParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    side:     Literal["n", "s", "e", "w"]
    delta_um: float = Field(gt=0)


class SnapParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    grid_um: float = Field(gt=0)


class RemoveParams(BaseModel):
    model_config = ConfigDict(frozen=True)


class RedrawParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    layer:  LayerTuple
    points: tuple[tuple[float, float], ...]


# ── verbs ────────────────────────────────────────────────────────────────

def widen(comp: Any, ref: PolygonRef, params: WidenParams) -> tuple[Any, PolygonRef]:
    """Expand the targeted polygon symmetrically by ``delta_um`` along ``axis``."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    x0, y0, x1, y1 = p.bbox
    h = params.delta_um / 2
    if params.axis == "x":
        new = Polygon(layer=p.layer, points=_rect_points(x0 - h, y0, x1 + h, y1))
    else:
        new = Polygon(layer=p.layer, points=_rect_points(x0, y0 - h, x1, y1 + h))
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def narrow(comp: Any, ref: PolygonRef, params: NarrowParams) -> tuple[Any, PolygonRef]:
    """Contract the polygon symmetrically by ``delta_um`` along ``axis``."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    x0, y0, x1, y1 = p.bbox
    h = params.delta_um / 2
    if params.axis == "x":
        if x1 - x0 < params.delta_um:
            raise ValueError(
                f"narrow Δ={params.delta_um} exceeds polygon width {x1 - x0:.4f} µm"
            )
        new = Polygon(layer=p.layer, points=_rect_points(x0 + h, y0, x1 - h, y1))
    else:
        if y1 - y0 < params.delta_um:
            raise ValueError(
                f"narrow Δ={params.delta_um} exceeds polygon height {y1 - y0:.4f} µm"
            )
        new = Polygon(layer=p.layer, points=_rect_points(x0, y0 + h, x1, y1 - h))
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def _shift(comp: Any, ref: PolygonRef, dx: float, dy: float) -> tuple[Any, PolygonRef]:
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    new = Polygon(
        layer=p.layer,
        points=tuple((x + dx, y + dy) for x, y in p.points),
    )
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def shift_n(comp: Any, ref: PolygonRef, params: ShiftParams) -> tuple[Any, PolygonRef]:
    """Translate the polygon by ``+delta_um`` in Y."""
    return _shift(comp, ref, 0.0,  params.delta_um)


def shift_s(comp: Any, ref: PolygonRef, params: ShiftParams) -> tuple[Any, PolygonRef]:
    """Translate the polygon by ``-delta_um`` in Y."""
    return _shift(comp, ref, 0.0, -params.delta_um)


def shift_e(comp: Any, ref: PolygonRef, params: ShiftParams) -> tuple[Any, PolygonRef]:
    """Translate the polygon by ``+delta_um`` in X."""
    return _shift(comp, ref,  params.delta_um, 0.0)


def shift_w(comp: Any, ref: PolygonRef, params: ShiftParams) -> tuple[Any, PolygonRef]:
    """Translate the polygon by ``-delta_um`` in X."""
    return _shift(comp, ref, -params.delta_um, 0.0)


def extend(comp: Any, ref: PolygonRef, params: EdgeParams) -> tuple[Any, PolygonRef]:
    """Push one edge outward by ``delta_um``."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    x0, y0, x1, y1 = p.bbox
    d = params.delta_um
    if   params.side == "n": y1 += d
    elif params.side == "s": y0 -= d
    elif params.side == "e": x1 += d
    else:                    x0 -= d
    new = Polygon(layer=p.layer, points=_rect_points(x0, y0, x1, y1))
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def shrink(comp: Any, ref: PolygonRef, params: EdgeParams) -> tuple[Any, PolygonRef]:
    """Push one edge inward by ``delta_um``."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    x0, y0, x1, y1 = p.bbox
    d = params.delta_um
    if params.side in ("n", "s"):
        if y1 - y0 < d:
            raise ValueError(
                f"shrink Δ={d} exceeds polygon height {y1 - y0:.4f} µm"
            )
    else:
        if x1 - x0 < d:
            raise ValueError(
                f"shrink Δ={d} exceeds polygon width {x1 - x0:.4f} µm"
            )
    if   params.side == "n": y1 -= d
    elif params.side == "s": y0 += d
    elif params.side == "e": x1 -= d
    else:                    x0 += d
    new = Polygon(layer=p.layer, points=_rect_points(x0, y0, x1, y1))
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def snap_to_grid(comp: Any, ref: PolygonRef, params: SnapParams) -> tuple[Any, PolygonRef]:
    """Round each vertex to the nearest ``grid_um`` multiple."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    p = polys[i]
    g = params.grid_um
    new_pts = tuple((round(x / g) * g, round(y / g) * g) for x, y in p.points)
    new = Polygon(layer=p.layer, points=new_pts)
    _replace(polys, i, new)
    return rebuild_component(polys), polygon_ref(new)


def remove(comp: Any, ref: PolygonRef, params: RemoveParams) -> tuple[Any, PolygonRef]:
    """Delete the polygon. Returns the ref of the removed polygon
    (useful only for diagnostics; the caller cannot re-target a deleted
    polygon)."""
    polys = extract_polygons(comp)
    i = find_polygon(polys, ref)
    removed = polys.pop(i)
    return rebuild_component(polys), polygon_ref(removed)


def redraw(comp: Any, ref: PolygonRef, params: RedrawParams) -> tuple[Any, PolygonRef]:
    """Add ``params.points`` as a new polygon on ``params.layer``.

    ``ref`` is ignored — :func:`redraw` is the inverse of :func:`remove`
    and provides the missing geometric data, so it doesn't need a target.
    """
    polys = extract_polygons(comp)
    new = Polygon(layer=params.layer, points=params.points)
    polys.append(new)
    return rebuild_component(polys), polygon_ref(new)


# ── inverse declarations ────────────────────────────────────────────────
#
# Each entry returns ``(inverse_verb_name, inverse_params_dict)``. The
# registry composes inverses via :func:`ActionRegistry.inverse_of`. For
# verbs that need the targeted polygon's data (notably :func:`remove`,
# whose inverse needs the deleted polygon's geometry), the inverse
# function takes an optional ``polygon`` argument.

def widen_inverse(params: WidenParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "narrow", {"axis": params.axis, "delta_um": params.delta_um}


def narrow_inverse(params: NarrowParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "widen", {"axis": params.axis, "delta_um": params.delta_um}


def shift_n_inverse(params: ShiftParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "shift_s", {"delta_um": params.delta_um}


def shift_s_inverse(params: ShiftParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "shift_n", {"delta_um": params.delta_um}


def shift_e_inverse(params: ShiftParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "shift_w", {"delta_um": params.delta_um}


def shift_w_inverse(params: ShiftParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "shift_e", {"delta_um": params.delta_um}


def extend_inverse(params: EdgeParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "shrink", {"side": params.side, "delta_um": params.delta_um}


def shrink_inverse(params: EdgeParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "extend", {"side": params.side, "delta_um": params.delta_um}


def snap_to_grid_inverse(params: SnapParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    # snap_to_grid is idempotent on an already-on-grid polygon. Its
    # inverse on an off-grid polygon is undefined (the original vertex
    # offsets are lost), so we return snap_to_grid itself — the round
    # trip is identity iff the polygon was on-grid to start with.
    return "snap_to_grid", {"grid_um": params.grid_um}


def remove_inverse(params: RemoveParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    if polygon is None:
        raise ValueError(
            "remove.inverse requires the removed polygon (pass `polygon=` to "
            "ActionRegistry.inverse_of). The deleted geometry is otherwise lost."
        )
    return "redraw", {"layer": polygon.layer, "points": polygon.points}


def redraw_inverse(params: RedrawParams, polygon: Optional[Polygon] = None) -> tuple[str, dict]:
    return "remove", {}
