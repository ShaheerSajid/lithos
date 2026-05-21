"""lithos_repair.features — schema for the local context an LLM/policy sees.

This module is the **data layer** of the repair package. It defines the
shape of what gets passed to the agent / policy for each violation:

* :class:`Polygon` — layer-tagged closed polygon in µm.
* :class:`PolygonRef` — lightweight reference (layer + centroid) used to
  identify a polygon across :mod:`actions` calls.
* :class:`FreeSpace` — N/S/E/W clearance to neighbouring geometry.
* :class:`Neighbor` — one polygon near the violating one, with same/diff
  net tagging when known.
* :class:`ViolationContext` — the bundle the agent receives per violation.

The actual extraction of :class:`ViolationContext` from a
``gdsfactory.Component`` lives in :mod:`lithos_repair.analyzer` (M3); this
module is just the schema.

All models are Pydantic so the agent's grammar is self-describing
(``ViolationContext.model_json_schema()`` → JSON schema).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


LayerTuple = tuple[int, int]
"""``(gds_layer, datatype)`` — the canonical layer identity in GDS."""


class Polygon(BaseModel):
    """Closed polygon (points in µm) tagged with its GDS layer.

    Points are stored as a tuple of ``(x, y)`` µm pairs. The polygon is
    implicitly closed (the loader does not require a duplicate final
    point). All :mod:`lithos_repair.actions` verbs assume axis-aligned
    rectangular polygons for v1; anything else round-trips correctly only
    if the verb commutes with the polygon's general shape.
    """
    model_config = ConfigDict(frozen=True)

    layer:  LayerTuple
    points: tuple[tuple[float, float], ...]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` µm bounding box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> tuple[float, float]:
        """Bounding-box centre in µm."""
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def width_um(self) -> float:
        x0, _, x1, _ = self.bbox
        return x1 - x0

    @property
    def height_um(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0


class PolygonRef(BaseModel):
    """Lightweight reference to a polygon: layer + centroid (µm).

    The repair actions return updated refs after each transformation so
    callers can chain operations without having to re-extract polygons.
    Matching tolerates a small tolerance (default one mfg-grid pitch)
    so that minor numerical drift from rebuilds doesn't break lookups.
    """
    model_config = ConfigDict(frozen=True)

    layer:      LayerTuple
    centroid_x: float
    centroid_y: float


def polygon_ref(p: Polygon) -> PolygonRef:
    """Build a :class:`PolygonRef` from a :class:`Polygon`."""
    cx, cy = p.centroid
    return PolygonRef(layer=p.layer, centroid_x=cx, centroid_y=cy)


class FreeSpace(BaseModel):
    """How much empty space (µm) lies in each cardinal direction.

    Zero or negative values mean another polygon is touching / overlapping
    on that side. ``None`` means the analyzer hasn't measured that side
    (e.g. cell-edge sentinel).
    """
    model_config = ConfigDict(frozen=True)

    n: float = 0.0
    s: float = 0.0
    e: float = 0.0
    w: float = 0.0


class Neighbor(BaseModel):
    """One polygon adjacent to the violating one.

    ``same_net`` is ``True`` if the analyzer determined the polygon
    belongs to the same electrical net (power, signal, etc.) as the
    violating polygon. ``None`` when net info isn't available — net
    extraction lives downstream of the repair agent.
    """
    model_config = ConfigDict(frozen=True)

    polygon:     Polygon
    distance_um: float
    same_net:    Optional[bool] = None


class ViolationContext(BaseModel):
    """All the local information the agent sees per violation.

    Populated by :func:`lithos_repair.analyzer.analyze` (M3) from a
    :class:`~lithos_drc.DRCViolation` + a ``gdsfactory.Component`` +
    rules.

    Attributes
    ----------
    rule
        Tool-emitted rule name (e.g. ``"NP.E.1"``).
    description
        Human-readable check description, when available.
    severity
        ``"error"`` or ``"warning"``.
    measured_um
        Measured value that caused the violation. ``None`` when the
        backend didn't report a numeric measurement.
    layer_name
        Logical (PDK-agnostic) layer name the violation is on, e.g.
        ``"poly"``. Empty when the analyzer couldn't determine it.
    cell_name
        Name of the cell the violation falls inside. Useful for the
        agent to disambiguate top-level vs sub-cell geometry.
    primary
        The polygon the analyzer believes is the principal subject of
        the violation. Always populated.
    neighbors
        Other polygons within the analyzer's search radius. Used by the
        agent to decide which direction is "safe" to move/widen.
    free_space
        N/S/E/W clearance from the primary polygon's bbox to the nearest
        other polygon.
    on_grid
        Whether every vertex of the primary polygon is on the mfg grid.
    is_array_member
        Heuristic: ``True`` if the primary polygon is part of an
        evenly-spaced array (e.g. one contact in a contact row). Useful
        for verbs that should propagate to siblings.
    device_path
        Cell-tree path from the top to the device the polygon belongs
        to, e.g. ``["INV", "M0"]``. Empty when the analyzer couldn't
        resolve a parent device.
    rule_hint
        Optional natural-language hint extracted by
        :class:`~lithos_ingest.FixMetadataExtractor`. Empty when the
        rule has no :class:`~lithos_core.FixMetadata` in the DB.
    """
    model_config = ConfigDict(frozen=True)

    rule:            str
    description:     str = ""
    severity:        str = "error"
    measured_um:     Optional[float] = None
    layer_name:      str = ""
    cell_name:       str = ""
    primary:         Polygon
    neighbors:       list[Neighbor] = Field(default_factory=list)
    free_space:      FreeSpace = Field(default_factory=FreeSpace)
    on_grid:         bool = True
    is_array_member: bool = False
    device_path:     list[str] = Field(default_factory=list)
    rule_hint:       Optional[str] = None

    @property
    def primary_ref(self) -> PolygonRef:
        """Convenience: ref to the primary polygon."""
        return polygon_ref(self.primary)
