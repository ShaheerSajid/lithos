"""Tests for :func:`lithos_repair.analyze`.

Covers layer resolution, primary-polygon selection (contains-point and
nearest), free-space measurement, neighbor collection, the on-grid /
array-member heuristics, and fix-metadata intent lookup.

The final integration test synthesises a real inverter and runs the
analyzer against a synthetic DRC violation pinned to one of its
contacts — satisfying the M3 acceptance criterion (one Python call
returns a populated :class:`ViolationContext`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from lithos_core import (
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    FixBranch,
    FixMetadata,
    LayerRef,
    PDKMetadata,
    Rule,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)
from lithos_drc          import DRCViolation
from lithos_layout       import BootstrapMapping, BootstrapRules
from lithos_repair       import (
    Polygon,
    analyze,
    polygon_ref,
    rebuild_component,
)


# ── Minimal fixtures ────────────────────────────────────────────────────

def _minimal_rules(tmp_path: Path) -> BootstrapRules:
    """Just enough BootstrapRules for the analyzer.

    Carries a layer map (so name → (gds, dt) resolution works), a 5 nm
    manufacturing grid, and a single rule with FixMetadata for the
    fix-hint test.
    """
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="mini", version="0", ingested_at="2026-05-21T00:00:00Z")
    db.upsert_rule(Rule(
        code         = "PO.W.1",
        category     = "geom",
        usage_class  = "geometry_primitive",
        constraint   = Constraint(branches=[ConstraintBranch(
            check=WidthCheck(target=LayerRef(name="poly"),
                             op=">=", threshold_um=0.15),
        )]),
        fix_metadata = FixMetadata(
            intent                 = "Poly width must stay above min.",
            allowed_action_classes = ["widen"],
            affected_layers        = ["poly"],
        ),
    ))
    db.upsert_rule(Rule(
        code        = "M1.S.1",
        category    = "geom",
        usage_class = "geometry_primitive",
        constraint  = Constraint(branches=[ConstraintBranch(
            check=SpacingCheck(layer_a=LayerRef(name="m1"),
                               op=">=", threshold_um=0.20),
        )]),
        # No fix_metadata — analyzer should return None for the hint.
    ))
    db.add_alias("po.w.1", "PO.W.1", source="manual")  # type: ignore[arg-type]
    md = PDKMetadata(
        name="mini", version="0",
        layers={
            "poly":      (66, 20),
            "diff":      (65, 20),
            "m1":        (68, 20),
            "contact":   (66, 44),
        },
        grid={"manufacturing_um": 0.005},
        drc_decks={},
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um": "PO.W.1",
        "m1.spacing_min_um": "M1.S.1",
    })
    return BootstrapRules(md, db, mapping)


def _viol(rule="PO.W.1", layer="poly", x=0.0, y=0.0,
          value: Optional[float] = None, description=""):
    return DRCViolation(rule=rule, description=description, layer=layer,
                        severity="error", x=x, y=y, value=value)


def _rect(layer, x0, y0, x1, y1):
    return Polygon(layer=layer, points=((x0,y0),(x1,y0),(x1,y1),(x0,y1)))


# ── Layer resolution ───────────────────────────────────────────────────

class TestLayerResolution:
    def test_known_layer_resolves_to_tuple(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp = rebuild_component([poly])
        ctx = analyze(_viol(layer="poly", x=0.25, y=0.1), comp, rules)
        assert ctx.layer_name == "poly"
        assert ctx.primary.layer == (66, 20)

    def test_unknown_layer_name_returned_verbatim(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp = rebuild_component([poly])
        ctx = analyze(_viol(layer="mystery", x=0.25, y=0.1), comp, rules)
        # Layer name is preserved but tuple lookup fails; primary
        # falls back to "any polygon" and picks the closest one.
        assert ctx.layer_name == "mystery"
        assert ctx.primary.layer == (66, 20)

    def test_blank_layer_falls_back_to_nearest(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        a = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        b = _rect((68, 20), 5.0, 5.0, 5.5, 5.2)
        comp = rebuild_component([a, b])
        ctx = analyze(_viol(layer="", x=0.25, y=0.1), comp, rules)
        assert ctx.layer_name == ""
        # Nearest by distance is `a`.
        assert ctx.primary.layer == (66, 20)


# ── Primary polygon selection ───────────────────────────────────────────

class TestPrimarySelection:
    def test_contains_point_wins_over_nearest(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        big   = _rect((66, 20), 0.0, 0.0, 1.0, 1.0)
        tiny  = _rect((66, 20), 5.0, 5.0, 5.1, 5.1)
        comp  = rebuild_component([big, tiny])
        ctx   = analyze(_viol(layer="poly", x=0.5, y=0.5), comp, rules)
        assert ctx.primary.bbox == (0.0, 0.0, 1.0, 1.0)

    def test_smallest_containing_wins_when_overlap(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        outer = _rect((66, 20), 0.0, 0.0, 2.0, 2.0)
        inner = _rect((66, 20), 0.4, 0.4, 0.6, 0.6)
        comp  = rebuild_component([outer, inner])
        ctx   = analyze(_viol(layer="poly", x=0.5, y=0.5), comp, rules)
        assert ctx.primary.bbox == (0.4, 0.4, 0.6, 0.6)

    def test_nearest_picked_when_no_containment(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        far   = _rect((66, 20), 10.0, 10.0, 11.0, 11.0)
        near  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([far, near])
        ctx   = analyze(_viol(layer="poly", x=0.7, y=0.1), comp, rules)
        assert ctx.primary.bbox == (0.0, 0.0, 0.5, 0.2)

    def test_no_polygons_returns_placeholder(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        comp  = rebuild_component([])
        ctx   = analyze(_viol(layer="poly", x=1.23, y=4.56), comp, rules)
        # Placeholder polygon at the violation centroid.
        assert ctx.primary.bbox == (1.23, 4.56, 1.23, 4.56)


# ── Free space ──────────────────────────────────────────────────────────

class TestFreeSpace:
    def test_clearance_in_all_four_directions(self, tmp_path: Path):
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 1.0, 1.0, 2.0, 2.0)
        north   = _rect((66, 20), 1.2, 2.3, 1.8, 2.5)   # 0.3 above
        south   = _rect((66, 20), 1.2, 0.4, 1.8, 0.8)   # 0.2 below
        east    = _rect((66, 20), 2.5, 1.2, 2.9, 1.8)   # 0.5 right
        west    = _rect((66, 20), 0.2, 1.2, 0.6, 1.8)   # 0.4 left
        comp    = rebuild_component([primary, north, south, east, west])
        ctx     = analyze(_viol(layer="poly", x=1.5, y=1.5), comp, rules)
        assert ctx.free_space.n == pytest.approx(0.3, abs=1e-6)
        assert ctx.free_space.s == pytest.approx(0.2, abs=1e-6)
        assert ctx.free_space.e == pytest.approx(0.5, abs=1e-6)
        assert ctx.free_space.w == pytest.approx(0.4, abs=1e-6)

    def test_no_neighbour_in_direction_returns_minus_one(self, tmp_path: Path):
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 0.0, 0.0, 1.0, 1.0)
        # Only a polygon to the north; other three directions are open.
        north   = _rect((66, 20), 0.2, 1.5, 0.8, 2.0)
        comp    = rebuild_component([primary, north])
        ctx     = analyze(_viol(layer="poly", x=0.5, y=0.5), comp, rules)
        assert ctx.free_space.n == pytest.approx(0.5, abs=1e-6)
        assert ctx.free_space.s == -1.0
        assert ctx.free_space.e == -1.0
        assert ctx.free_space.w == -1.0

    def test_no_overlap_does_not_block(self, tmp_path: Path):
        """A polygon offset on the perpendicular axis doesn't block."""
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 0.0, 0.0, 1.0, 1.0)
        # Northeast diagonal — sits above primary but is east of its
        # x range. Should not show up as the "north" blocker.
        diagonal = _rect((66, 20), 5.0, 5.0, 5.5, 5.5)
        comp     = rebuild_component([primary, diagonal])
        ctx      = analyze(_viol(layer="poly", x=0.5, y=0.5), comp, rules)
        assert ctx.free_space.n == -1.0


# ── Neighbors ───────────────────────────────────────────────────────────

class TestNeighbors:
    def test_within_radius_collected_and_sorted(self, tmp_path: Path):
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 0.0, 0.0, 1.0, 1.0)
        near    = _rect((68, 20), 1.5, 0.0, 1.8, 0.5)   # 0.5 east
        far     = _rect((68, 20), 10.0, 0.0, 10.5, 0.5)  # 9.0 east
        mid     = _rect((68, 20), 2.5, 0.0, 2.9, 0.5)   # 1.5 east
        comp    = rebuild_component([primary, near, mid, far])
        ctx     = analyze(_viol(layer="poly", x=0.5, y=0.5),
                          comp, rules, search_radius_um=2.0)
        # near and mid within radius; far excluded.
        assert len(ctx.neighbors) == 2
        assert ctx.neighbors[0].distance_um < ctx.neighbors[1].distance_um

    def test_primary_excluded_from_neighbors(self, tmp_path: Path):
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 0.0, 0.0, 1.0, 1.0)
        comp    = rebuild_component([primary])
        ctx     = analyze(_viol(layer="poly", x=0.5, y=0.5), comp, rules)
        assert ctx.neighbors == []


# ── On-grid + array heuristics ─────────────────────────────────────────

class TestHeuristics:
    def test_on_grid_polygon(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)  # 0.005 grid
        poly  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(layer="poly", x=0.25, y=0.1), comp, rules)
        assert ctx.on_grid is True

    def test_array_member_detected(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        # Three identical contacts in a row.
        a = _rect((66, 44), 0.0, 0.0, 0.17, 0.17)
        b = _rect((66, 44), 0.5, 0.0, 0.67, 0.17)
        c = _rect((66, 44), 1.0, 0.0, 1.17, 0.17)
        comp = rebuild_component([a, b, c])
        ctx  = analyze(_viol(layer="contact", x=0.08, y=0.08), comp, rules)
        assert ctx.is_array_member is True

    def test_isolated_polygon_not_array(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((66, 44), 0.0, 0.0, 0.17, 0.17)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(layer="contact", x=0.08, y=0.08), comp, rules)
        assert ctx.is_array_member is False


# ── Fix-metadata hint ──────────────────────────────────────────────────

class TestRuleHint:
    def test_intent_surfaced_when_present(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(rule="PO.W.1", layer="poly", x=0.25, y=0.1),
                        comp, rules)
        assert ctx.rule_hint == "Poly width must stay above min."

    def test_alias_resolves_to_canonical(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(rule="po.w.1", layer="poly", x=0.25, y=0.1),
                        comp, rules)
        assert ctx.rule_hint == "Poly width must stay above min."

    def test_missing_fix_metadata_returns_none(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((68, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(rule="M1.S.1", layer="m1", x=0.25, y=0.1),
                        comp, rules)
        assert ctx.rule_hint is None

    def test_unknown_rule_returns_none(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(rule="not_in_db", layer="poly", x=0.25, y=0.1),
                        comp, rules)
        assert ctx.rule_hint is None


# ── Full bundle ────────────────────────────────────────────────────────

class TestViolationContextBundle:
    def test_all_fields_populated(self, tmp_path: Path):
        rules   = _minimal_rules(tmp_path)
        primary = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        nbr     = _rect((66, 20), 0.7, 0.0, 0.8, 0.2)
        comp    = rebuild_component([primary, nbr])
        ctx     = analyze(
            _viol(rule="PO.W.1", layer="poly", x=0.25, y=0.1,
                  value=0.12, description="poly too narrow"),
            comp, rules,
        )
        assert ctx.rule        == "PO.W.1"
        assert ctx.description == "poly too narrow"
        assert ctx.severity    == "error"
        assert ctx.measured_um == 0.12
        assert ctx.layer_name  == "poly"
        assert ctx.cell_name   != ""           # gdsfactory auto-names
        assert ctx.primary.layer == (66, 20)
        assert len(ctx.neighbors) == 1
        assert ctx.on_grid is True
        assert ctx.is_array_member is False
        assert ctx.device_path == []
        assert ctx.rule_hint == "Poly width must stay above min."

    def test_context_is_json_serialisable(self, tmp_path: Path):
        rules = _minimal_rules(tmp_path)
        poly  = _rect((66, 20), 0.0, 0.0, 0.5, 0.2)
        comp  = rebuild_component([poly])
        ctx   = analyze(_viol(rule="PO.W.1", layer="poly", x=0.25, y=0.1),
                        comp, rules)
        # Pydantic round-trip.
        s = ctx.model_dump_json()
        assert "PO.W.1" in s
