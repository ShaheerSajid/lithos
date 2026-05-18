"""lithos_core.ir — typed constraint IR for DRC rules.

The schema both deck parsers (SVRF / PVS / KLayout-DRC / Magic) project into
and the runtime repair engine evaluates against geometry. It is the canonical
form of a parsed rule.

Three discriminated-union ASTs build on each other:

  1. ``LayerExpr`` — pure functions of primary layers producing derived layers
     (the deck's "layer algebra": booleans, sizing, selection by neighbour, …).
  2. ``MeasurementCondition`` — predicates that gate a check (parallel-run-length,
     width-band, edge orientation, …).
  3. ``CheckExpr`` — geometric measurement over layer expressions
     (`WidthCheck`, `SpacingCheck`, `EnclosureCheck`, `AreaCheck`,
     `DensityCheck`, `AntennaCheck`).

A ``Constraint`` ties these together: optional named ``derived_layers`` plus
one or more ``ConstraintBranch`` entries. The first branch whose predicate is
satisfied for a given geometry instance is the one that fires; an empty
predicate is the default branch.

Conventions
-----------
* Length-typed fields are in micrometres, suffixed ``_um``.
* Area fields are in µm², suffixed ``_um2``.
* Comparator operators use the strings ``"<"``, ``"<="``, ``">"``, ``">="``,
  ``"="``, ``"!="`` — these read naturally in branch predicates.
* Discriminated unions key on the ``kind`` field. To add a new variant,
  define the model with a unique ``kind`` literal and append it to the
  matching ``Annotated[Union[...], Field(discriminator="kind")]`` alias
  below.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


ComparatorOp = Literal["<", "<=", ">", ">=", "=", "!="]
"""Threshold comparator. Read naturally as `threshold {op} measured`."""

EdgeFilterAxis = Literal["horizontal", "vertical", "any"]
"""Edge orientation filter for `LayerEdges` and `EdgeOrientation` conditions."""

DeckDialect = Literal["svrf", "pvs", "klayout", "magic"]
"""Source dialect a `Constraint` was projected from."""


# ── Layer algebra ────────────────────────────────────────────────────────────
#
# Each `LayerExpr` node is a pure function of primary (or other derived) layers.
# Together they form an AST that mirrors the layer-derivation logic real decks
# perform before measuring anything.

class LayerRef(BaseModel):
    """A reference to a logical layer by name (e.g. ``"met2"``)."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["layer_ref"] = "layer_ref"
    name: str


class LayerBool(BaseModel):
    """Boolean combination of layer expressions.

    ``not`` requires exactly one operand; the others take two or more.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["bool"] = "bool"
    op: Literal["and", "or", "not", "xor"]
    operands: list["LayerExpr"]


class LayerSize(BaseModel):
    """SIZE BY — grow (positive) or shrink (negative) all polygons in ``operand``.

    Used to model bloat/oversize operations that show up before spacing or
    enclosure checks in many decks.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["size"] = "size"
    operand: "LayerExpr"
    by_um: float


class LayerSelect(BaseModel):
    """Select polygons of ``subject`` by their geometric relationship to ``reference``.

    Mirrors SVRF/PVS operators like ``INSIDE``, ``OUTSIDE``, ``INTERACT``,
    ``TOUCH``, ``ENCLOSE``, ``COVERS``.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["select"] = "select"
    op: Literal["inside", "outside", "interact", "touch", "enclose", "covers"]
    subject: "LayerExpr"
    reference: "LayerExpr"


class LayerEdges(BaseModel):
    """Extract polygon edges as a derived layer, optionally filtered by axis."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["edges"] = "edges"
    operand: "LayerExpr"
    axis: EdgeFilterAxis = "any"


class LayerHoles(BaseModel):
    """The holes (interior negative space) of ``operand``."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["holes"] = "holes"
    operand: "LayerExpr"


class LayerConnect(BaseModel):
    """A connectivity-derived layer: the union of polygons connected through
    ``layers`` (and optional ``via_layers``) into the same net.

    Used by short/open style checks. Most physical DRC rules don't need this,
    but it's needed for the small set that do.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["connect"] = "connect"
    layers: list["LayerExpr"]
    via_layers: list["LayerExpr"] = []


LayerExpr = Annotated[
    Union[
        LayerRef,
        LayerBool,
        LayerSize,
        LayerSelect,
        LayerEdges,
        LayerHoles,
        LayerConnect,
    ],
    Field(discriminator="kind"),
]


# ── Conditions that gate a measurement ───────────────────────────────────────
#
# Real spacing/width rules are rarely a single threshold. They typically take
# the form "if predicate P holds for the local geometry, the threshold is T".
# We model P as a list of `MeasurementCondition` nodes — all must hold.

class ParallelRunLength(BaseModel):
    """PRL filter: the parallel run length between two edges falls in [min, max).

    Either bound may be ``None`` for an open interval.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["prl"] = "prl"
    min_um: Optional[float] = None
    max_um: Optional[float] = None


class WidthBand(BaseModel):
    """Width filter on the participating polygon(s)."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["width_band"] = "width_band"
    min_um: Optional[float] = None
    max_um: Optional[float] = None


class LengthBand(BaseModel):
    """Length filter on the participating edge / polygon."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["length_band"] = "length_band"
    min_um: Optional[float] = None
    max_um: Optional[float] = None


class EdgeOrientation(BaseModel):
    """Restrict the check to edges of a given axis."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["edge_orientation"] = "edge_orientation"
    axis: EdgeFilterAxis


class LayerPresence(BaseModel):
    """Gate the check on whether the geometry sits inside / near / coincident
    with a reference derived-layer set.

    Models rules of the form "spacing only applies when both shapes are over
    a particular implant / well / mask".
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["layer_presence"] = "layer_presence"
    reference: LayerExpr
    mode: Literal["inside", "near", "coincident"] = "inside"


MeasurementCondition = Annotated[
    Union[
        ParallelRunLength,
        WidthBand,
        LengthBand,
        EdgeOrientation,
        LayerPresence,
    ],
    Field(discriminator="kind"),
]


# ── Check algebra ────────────────────────────────────────────────────────────
#
# Each `CheckExpr` node corresponds to one geometric measurement that produces
# violations when the comparison fails.

SpacingModifier = str
"""Edge-pair selector modifier names, as emitted by the source dialect.

Modifiers are dialect-specific decorations on spacing checks (``projecting``
edge pairs only, ``square`` end-of-line geometry, …). Rather than force
every parser into one canonical vocabulary, the IR keeps them as plain
strings and documents the common values:

    SVRF / PVS              : opposite, projecting, parallel, square
    KLayout DRC (Ruby)      : projection, square, euclidian, opposite,
                              transparent, intra_polygon, shielded,
                              whole_edges

The repair engine maps these back to behavioural intent at runtime."""

EnclosureSides = Literal["all", "two_adjacent", "two_opposite", "one"]
"""Which subset of edges of ``outer`` must satisfy the enclosure rule.

`two_adjacent` and `two_opposite` model asymmetric enclosure (a sky130 idiom
on contacts, vias).
"""


class WidthCheck(BaseModel):
    """Minimum (or other comparator) width of ``target``."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["width"] = "width"
    target: LayerExpr
    op: ComparatorOp
    threshold_um: float
    conditions: list[MeasurementCondition] = []


class SpacingCheck(BaseModel):
    """Spacing between ``layer_a`` and ``layer_b``.

    If ``layer_b`` is ``None`` the check is internal (same-layer) spacing.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["spacing"] = "spacing"
    layer_a: LayerExpr
    layer_b: Optional[LayerExpr] = None
    op: ComparatorOp
    threshold_um: float
    conditions: list[MeasurementCondition] = []
    modifiers: list[SpacingModifier] = []


class EnclosureCheck(BaseModel):
    """``inner`` must be enclosed by ``outer`` by at least ``threshold_um``."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["enclosure"] = "enclosure"
    inner: LayerExpr
    outer: LayerExpr
    op: ComparatorOp
    threshold_um: float
    on_sides: EnclosureSides = "all"
    conditions: list[MeasurementCondition] = []


class AreaCheck(BaseModel):
    """Polygon area comparison on ``target``."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["area"] = "area"
    target: LayerExpr
    op: ComparatorOp
    threshold_um2: float


class DensityCheck(BaseModel):
    """Layer density over a window of size ``window_um``.

    At least one of ``min_ratio`` / ``max_ratio`` should be set (both for
    band rules).
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["density"] = "density"
    target: LayerExpr
    window_um: float
    min_ratio: Optional[float] = None
    max_ratio: Optional[float] = None


class AntennaCheck(BaseModel):
    """Antenna ratio: metal area attached to a gate divided by gate area.

    ``via_dependent=True`` indicates the deck applies cumulative-area logic
    that depends on which via layer is in play (common in nodes ≤ 65nm).
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["antenna"] = "antenna"
    metal_area: LayerExpr
    gate_area: LayerExpr
    ratio_limit: float
    via_dependent: bool = False


class ExistenceCheck(BaseModel):
    """A boolean "this set must be (non-)empty" check.

    Many foundry rules don't have a numeric threshold — they express a
    forbidden pattern as a layer expression that must evaluate to no
    polygons. Examples from real Calibre decks::

        # implant exclusivity
        PP AND NP                 → ExistenceCheck(target=PP AND NP, must_be_empty=True)

        # NW resistor not doped
        RWDMY AND NPOD            → must_be_empty=True

        # NT_N must not interact DNW
        NTN AND DNW               → must_be_empty=True

    When ``must_be_empty=True`` (the common case), the rule fires when the
    target layer set is non-empty. ``False`` is the rarer "this set must
    exist" form (e.g. a mandatory guard ring).
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["existence"] = "existence"
    target: LayerExpr
    must_be_empty: bool = True
    conditions: list[MeasurementCondition] = []


CheckExpr = Annotated[
    Union[
        WidthCheck,
        SpacingCheck,
        EnclosureCheck,
        AreaCheck,
        DensityCheck,
        AntennaCheck,
        ExistenceCheck,
    ],
    Field(discriminator="kind"),
]


# ── Top-level Constraint ─────────────────────────────────────────────────────

class ConstraintBranch(BaseModel):
    """One arm of a multi-clause rule.

    The branch fires when every entry in ``predicate`` is satisfied.
    An empty predicate is the default branch and always fires.
    """
    model_config = ConfigDict(frozen=True)

    predicate: list[MeasurementCondition] = []
    check: CheckExpr


class Constraint(BaseModel):
    """A parsed DRC rule constraint.

    ``derived_layers`` are named layer expressions referenceable by name from
    inside ``branches[*].check`` (useful for rules that share intermediates
    across multiple branches).
    """
    model_config = ConfigDict(frozen=True)

    derived_layers: dict[str, LayerExpr] = {}
    branches: list[ConstraintBranch]
    deck_dialect: Optional[DeckDialect] = None
    raw_deck_text: Optional[str] = None


# Pydantic v2 needs explicit rebuild for recursive forward references.
LayerBool.model_rebuild()
LayerSize.model_rebuild()
LayerSelect.model_rebuild()
LayerEdges.model_rebuild()
LayerHoles.model_rebuild()
LayerConnect.model_rebuild()
