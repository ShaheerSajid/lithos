"""lithos_repair.analyzer — DRC violation → ViolationContext.

Wraps the M2 feature schema with extraction logic. Given a single
:class:`~lithos_drc.DRCViolation`, the parent :class:`gdsfactory.Component`
the violation came from, and the :class:`~lithos_layout.BootstrapRules`
in use, :func:`analyze` returns a populated
:class:`~lithos_repair.features.ViolationContext` ready for the M5 agent
or M6 policy.

Pipeline:

1. Resolve the violation's layer to ``(gds, datatype)`` via the bootstrap
   metadata. ``violation.layer`` is the canonical layer name (when the
   backend supplies it); we look it up in
   :attr:`~lithos_core.PDKMetadata.layers` to get the GDS tuple.
2. Extract every polygon from the component (via
   :func:`~lithos_repair.actions.extract_polygons`).
3. Pick the primary polygon: the one that contains the violation's
   ``(x, y)`` centroid, or the nearest one on the violation's layer.
4. Compute the cardinal-direction free space from the primary's bbox
   to the nearest other polygon, and the per-side list of nearby
   neighbours (within ``search_radius_um``).
5. Look up the rule's :class:`~lithos_core.FixMetadata` in the rule DB
   if present, and surface the intent string as ``rule_hint``.

The analyzer doesn't walk the gdsfactory cell tree yet — the synthesizer
flattens before DRC runs, so ``device_path`` is left empty. That's fine
for the M5 acceptance criterion; if needed, the analyzer can be extended
to honour a pre-flatten device map later.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from .actions import extract_polygons
from .features import (
    FreeSpace,
    LayerTuple,
    Neighbor,
    Polygon,
    ViolationContext,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from lithos_core.db   import RuleDB
    from lithos_drc       import DRCViolation
    from lithos_layout    import BootstrapRules


# ── Public entry point ──────────────────────────────────────────────────

def analyze(
    violation:        "DRCViolation",
    comp:             Any,
    rules:            "BootstrapRules",
    *,
    search_radius_um: float = 2.0,
    db:               Optional["RuleDB"] = None,
) -> ViolationContext:
    """Build a :class:`ViolationContext` for ``violation`` against ``comp``.

    Parameters
    ----------
    violation
        One DRC violation from any :class:`~lithos_drc.DRCRunner` backend.
    comp
        The :class:`gdsfactory.Component` that produced the violation.
    rules
        Bootstrap rules — supplies the layer name ↔ GDS tuple map, the
        manufacturing grid (for the on-grid check), and the rule DB (for
        the fix-metadata hint, unless overridden via ``db``).
    search_radius_um
        How far (µm) to look for neighbour polygons. Anything outside
        this radius doesn't appear in :attr:`ViolationContext.neighbors`
        but free-space measurements always run on the full polygon list.
    db
        Optional override for the rule DB. Defaults to ``rules.db``.

    Returns
    -------
    ViolationContext
        Always returns a populated context. When the analyzer can't
        resolve a primary polygon (e.g. violation centroid is outside
        every polygon on the violation's layer and no polygons exist),
        falls back to a zero-area placeholder polygon at the violation
        centroid so downstream callers don't have to special-case None.
    """
    polygons = extract_polygons(comp)
    vx, vy   = float(violation.x), float(violation.y)

    layer_name, layer_tuple = _resolve_layer(violation, rules)

    primary = _find_primary(polygons, layer_tuple, vx, vy)
    if primary is None:
        # Degenerate fallback — no polygons at all, or the violation is
        # on a layer we don't know about. Emit a zero-area placeholder
        # so the schema is always populated.
        placeholder_layer: LayerTuple = layer_tuple or (-1, -1)
        primary = Polygon(
            layer  = placeholder_layer,
            points = ((vx, vy), (vx, vy), (vx, vy), (vx, vy)),
        )

    neighbors  = _collect_neighbors(polygons, primary, search_radius_um)
    free_space = _free_space(polygons, primary)
    on_grid    = _is_on_grid(primary, rules.mfg_grid)
    is_array   = _is_array_member(polygons, primary)
    rule_hint  = _fix_metadata_intent(violation, db or rules.db)

    cell_name = _safe_cell_name(comp)

    return ViolationContext(
        rule            = violation.rule,
        description     = violation.description,
        severity        = violation.severity,
        measured_um     = violation.value,
        layer_name      = layer_name,
        cell_name       = cell_name,
        primary         = primary,
        neighbors       = neighbors,
        free_space      = free_space,
        on_grid         = on_grid,
        is_array_member = is_array,
        device_path     = [],                   # cell-tree walk: deferred
        rule_hint       = rule_hint,
    )


# ── Layer resolution ────────────────────────────────────────────────────

def _resolve_layer(
    violation: "DRCViolation",
    rules:     "BootstrapRules",
) -> tuple[str, Optional[LayerTuple]]:
    """Resolve the violation's layer to ``(name, (gds, datatype))``.

    ``violation.layer`` is the human-readable layer name the backend
    reported. If that's set and the PDK knows it, we get a tuple. If
    not, we still return the string we have (or empty), and ``None``
    for the tuple — the primary-polygon search falls back to "any layer".
    """
    name = violation.layer or ""
    if not name:
        return "", None
    try:
        return name, rules.metadata.layer(name)
    except KeyError:
        return name, None


# ── Primary polygon ────────────────────────────────────────────────────

def _find_primary(
    polygons:    list[Polygon],
    layer:       Optional[LayerTuple],
    vx:          float,
    vy:          float,
) -> Optional[Polygon]:
    """Return the polygon best matching the violation's location.

    Selection rule:

    1. Candidates = polygons on the violation's layer (if known).
       If no candidates, fall back to all polygons.
    2. First preference: a polygon whose bbox **contains** ``(vx, vy)``.
       If multiple, pick the smallest by area (the most specific).
    3. Otherwise: the polygon with the smallest ``min`` distance from
       its bbox to ``(vx, vy)``.
    """
    if not polygons:
        return None

    if layer is not None:
        candidates = [p for p in polygons if p.layer == layer]
        if not candidates:
            candidates = polygons
    else:
        candidates = polygons

    containing = [p for p in candidates if _bbox_contains(p.bbox, vx, vy)]
    if containing:
        containing.sort(key=lambda p: _bbox_area(p.bbox))
        return containing[0]

    return min(candidates, key=lambda p: _bbox_distance(p.bbox, vx, vy))


def _bbox_contains(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_distance(bbox: tuple[float, float, float, float], x: float, y: float) -> float:
    """``(x, y)`` to bbox: 0 inside, Euclidean outside."""
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


# ── Free space ──────────────────────────────────────────────────────────

def _free_space(polygons: list[Polygon], primary: Polygon) -> FreeSpace:
    """N/S/E/W clearance from ``primary``'s bbox to the nearest other polygon.

    "Other" is defined as any polygon ≠ ``primary``. Layer is not
    considered: any geometry can block a same-axis move. The free-space
    measurement is taken edge-to-edge along each axis, with the
    requirement that the other polygon must overlap ``primary`` on the
    perpendicular axis (otherwise it's not actually blocking a straight
    push in that direction).
    """
    x0, y0, x1, y1 = primary.bbox
    n = s = e = w = float("inf")

    for p in polygons:
        if p is primary:
            continue
        px0, py0, px1, py1 = p.bbox

        # North / south require horizontal overlap.
        if px1 > x0 and px0 < x1:
            if py0 >= y1:                       # p is north of primary
                n = min(n, py0 - y1)
            elif py1 <= y0:                     # p is south of primary
                s = min(s, y0 - py1)
        # East / west require vertical overlap.
        if py1 > y0 and py0 < y1:
            if px0 >= x1:                       # p is east of primary
                e = min(e, px0 - x1)
            elif px1 <= x0:                     # p is west of primary
                w = min(w, x0 - px1)

    # Replace infinities with -1 so the JSON-serialised context stays
    # well-formed and finite ("we never measured anything in this
    # direction" → -1, easy for the agent to filter).
    def _finite(v: float) -> float:
        return v if math.isfinite(v) else -1.0

    return FreeSpace(n=_finite(n), s=_finite(s), e=_finite(e), w=_finite(w))


# ── Neighbors ───────────────────────────────────────────────────────────

def _collect_neighbors(
    polygons:         list[Polygon],
    primary:          Polygon,
    search_radius_um: float,
) -> list[Neighbor]:
    """Return polygons within ``search_radius_um`` of ``primary``'s bbox.

    Same-net detection is **not** wired in this M3 cut: ``same_net`` is
    left as ``None``. Net extraction is a downstream concern (LVS); for
    now the agent has to infer net identity from layer + spatial
    relationships.
    """
    out: list[Neighbor] = []
    for p in polygons:
        if p is primary:
            continue
        d = _bbox_to_bbox_distance(primary.bbox, p.bbox)
        if d <= search_radius_um:
            out.append(Neighbor(polygon=p, distance_um=d, same_net=None))
    # Sort by distance — agents typically care about the nearest few.
    out.sort(key=lambda nb: nb.distance_um)
    return out


def _bbox_to_bbox_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Minimum Euclidean distance between two bboxes. ``0`` when overlapping."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    return math.hypot(dx, dy)


# ── On-grid check ───────────────────────────────────────────────────────

def _is_on_grid(p: Polygon, grid_um: float, tol_um: float = 1e-9) -> bool:
    """True iff every vertex of ``p`` is on the manufacturing grid."""
    if grid_um <= 0:
        return True
    for x, y in p.points:
        if abs(round(x / grid_um) * grid_um - x) > tol_um:
            return False
        if abs(round(y / grid_um) * grid_um - y) > tol_um:
            return False
    return True


# ── Array-member heuristic ──────────────────────────────────────────────

def _is_array_member(polygons: list[Polygon], primary: Polygon) -> bool:
    """True if ``primary`` looks like one element of a regular array.

    Heuristic: count same-layer polygons whose bbox dimensions match
    ``primary`` (within 0.001 µm). A match of ≥3 (including ``primary``
    itself) is treated as an array — typical for contact rows, via
    matrices, m2 power straps. False positives are harmless here; this
    feature just flags a hint for the agent.
    """
    pw, ph = primary.width_um, primary.height_um
    matches = 0
    for p in polygons:
        if p.layer != primary.layer:
            continue
        if abs(p.width_um  - pw) > 0.001:
            continue
        if abs(p.height_um - ph) > 0.001:
            continue
        matches += 1
    return matches >= 3


# ── Fix-metadata lookup ─────────────────────────────────────────────────

def _fix_metadata_intent(
    violation: "DRCViolation",
    db:        Optional["RuleDB"],
) -> Optional[str]:
    """Return the rule's :class:`FixMetadata.intent` string if present.

    Looks up the violation's raw rule string in the alias table first,
    then fetches the canonical rule. Returns ``None`` when:

    * the DB is missing,
    * no alias matches,
    * the rule has no :class:`~lithos_core.FixMetadata`, or
    * the metadata's ``intent`` is empty.

    M3's acceptance criterion explicitly allows ``None`` here — sky130
    and TSMC180 don't have ``fix_metadata`` populated yet.
    """
    if db is None:
        return None
    try:
        code = db.resolve_alias(violation.rule) or violation.rule
        rule = db.get_rule(code)
    except Exception:                            # pragma: no cover
        return None
    if rule is None or rule.fix_metadata is None:
        return None
    intent = (rule.fix_metadata.intent or "").strip()
    return intent or None


# ── Cell name extraction ────────────────────────────────────────────────

def _safe_cell_name(comp: Any) -> str:
    try:
        return str(comp.name)
    except Exception:                            # pragma: no cover
        return ""
