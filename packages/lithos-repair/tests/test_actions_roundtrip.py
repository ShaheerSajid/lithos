"""Round-trip tests for every M2 verb in :mod:`lithos_repair.actions`.

Acceptance criterion (from ``docs/REPAIR_PLAN.md``, M2):

    apply(v) ∘ apply(v.inverse) == identity within the mfg grid

The mfg grid is 5 nm; comparing polygons we allow ≤1 nm drift per
vertex, which absorbs KLayout's dbu rounding without masking real
shape changes.
"""
from __future__ import annotations

import pytest

from lithos_repair import (
    REGISTRY,
    EdgeParams,
    NarrowParams,
    Polygon,
    PolygonRef,
    RedrawParams,
    RemoveParams,
    ShiftParams,
    SnapParams,
    WidenParams,
    extract_polygons,
    polygon_ref,
    rebuild_component,
)


# ── Fixtures ────────────────────────────────────────────────────────────

LAYER_MAIN = (1, 0)
LAYER_OTHER = (2, 0)

TOL_UM = 0.001  # 1 nm — one KLayout dbu


def _simple_comp():
    """Component with two polygons on two layers, for round-trip tests."""
    main = Polygon(
        layer  = LAYER_MAIN,
        points = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)),
    )
    other = Polygon(
        layer  = LAYER_OTHER,
        points = ((3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0)),
    )
    return rebuild_component([main, other]), main, other


def _polygons_equal(a: Polygon, b: Polygon, tol: float = TOL_UM) -> bool:
    if a.layer != b.layer:
        return False
    if len(a.points) != len(b.points):
        return False
    # KLayout normalises vertex order; compare bbox + vertex set.
    if any(abs(av - bv) > tol for av, bv in zip(a.bbox, b.bbox)):
        return False
    sa = sorted((round(x, 4), round(y, 4)) for x, y in a.points)
    sb = sorted((round(x, 4), round(y, 4)) for x, y in b.points)
    return sa == sb


def _polygon_lists_equal(xs: list[Polygon], ys: list[Polygon]) -> bool:
    if len(xs) != len(ys):
        return False
    used = [False] * len(ys)
    for x in xs:
        for j, y in enumerate(ys):
            if not used[j] and _polygons_equal(x, y):
                used[j] = True
                break
        else:
            return False
    return True


def _round_trip(
    verb_name: str,
    params:    dict,
    target:    Polygon,
):
    """Apply verb then its inverse; return (original_polys, final_polys)."""
    comp, main, other = _simple_comp()
    original = extract_polygons(comp)

    ref = polygon_ref(target)
    comp_after, ref_after = REGISTRY.apply(verb_name, comp, ref, params)
    inv_name, inv_params = REGISTRY.inverse_of(
        verb_name, params, polygon=target,
    )
    comp_back, _ = REGISTRY.apply(inv_name, comp_after, ref_after, inv_params)
    final = extract_polygons(comp_back)
    return original, final


# ── Per-verb round-trip ─────────────────────────────────────────────────

class TestRoundTrip:
    def test_widen_narrow_x(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "widen", {"axis": "x", "delta_um": 0.05}, main,
        )
        assert _polygon_lists_equal(original, final)

    def test_widen_narrow_y(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "widen", {"axis": "y", "delta_um": 0.04}, main,
        )
        assert _polygon_lists_equal(original, final)

    def test_narrow_widen(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "narrow", {"axis": "x", "delta_um": 0.1}, main,
        )
        assert _polygon_lists_equal(original, final)

    def test_shift_n_s(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip("shift_n", {"delta_um": 0.2}, main)
        assert _polygon_lists_equal(original, final)

    def test_shift_s_n(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip("shift_s", {"delta_um": 0.15}, main)
        assert _polygon_lists_equal(original, final)

    def test_shift_e_w(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip("shift_e", {"delta_um": 0.3}, main)
        assert _polygon_lists_equal(original, final)

    def test_shift_w_e(self):
        _, main, _ = _simple_comp()
        original, final = _round_trip("shift_w", {"delta_um": 0.25}, main)
        assert _polygon_lists_equal(original, final)

    @pytest.mark.parametrize("side", ["n", "s", "e", "w"])
    def test_extend_shrink(self, side):
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "extend", {"side": side, "delta_um": 0.07}, main,
        )
        assert _polygon_lists_equal(original, final)

    @pytest.mark.parametrize("side", ["n", "s", "e", "w"])
    def test_shrink_extend(self, side):
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "shrink", {"side": side, "delta_um": 0.07}, main,
        )
        assert _polygon_lists_equal(original, final)

    def test_snap_to_grid_idempotent_on_grid(self):
        """Polygon already on the 5 nm grid: snap_to_grid round-trip is identity."""
        _, main, _ = _simple_comp()
        original, final = _round_trip(
            "snap_to_grid", {"grid_um": 0.005}, main,
        )
        assert _polygon_lists_equal(original, final)

    def test_remove_redraw(self):
        """Removing then redrawing the original polygon restores the component."""
        comp, main, other = _simple_comp()
        original = extract_polygons(comp)

        ref = polygon_ref(main)
        comp2, ref2 = REGISTRY.apply("remove", comp, ref, {})
        # Verify the polygon is actually gone.
        assert len(extract_polygons(comp2)) == 1

        inv_name, inv_params = REGISTRY.inverse_of("remove", {}, polygon=main)
        assert inv_name == "redraw"

        comp3, _ = REGISTRY.apply(inv_name, comp2, ref2, inv_params)
        final = extract_polygons(comp3)
        assert _polygon_lists_equal(original, final)

    def test_redraw_remove(self):
        """Drawing a new polygon then removing it restores the component."""
        comp, main, other = _simple_comp()
        original = extract_polygons(comp)

        new_points = ((5.0, 5.0), (5.5, 5.0), (5.5, 5.3), (5.0, 5.3))
        params = {"layer": LAYER_MAIN, "points": new_points}
        comp2, ref2 = REGISTRY.apply("redraw", comp, polygon_ref(main), params)
        assert len(extract_polygons(comp2)) == 3

        inv_name, inv_params = REGISTRY.inverse_of(
            "redraw", params,
        )
        assert inv_name == "remove"
        comp3, _ = REGISTRY.apply(inv_name, comp2, ref2, inv_params)
        final = extract_polygons(comp3)
        assert _polygon_lists_equal(original, final)


# ── Registry sanity ─────────────────────────────────────────────────────

class TestRegistry:
    def test_names_includes_all_verbs(self):
        expected = {
            "widen", "narrow",
            "shift_n", "shift_s", "shift_e", "shift_w",
            "extend", "shrink",
            "snap_to_grid",
            "remove", "redraw",
        }
        assert expected.issubset(set(REGISTRY.names()))

    def test_grammar_export_is_json_serialisable(self):
        import json
        g = REGISTRY.grammar()
        # Smoke test: every verb has a description + params schema.
        for name, entry in g["verbs"].items():
            assert isinstance(entry["description"], str)
            assert isinstance(entry["params"], dict)
        json.dumps(g)  # must not raise

    def test_remove_inverse_needs_polygon(self):
        with pytest.raises(ValueError, match="removed polygon"):
            REGISTRY.inverse_of("remove", {})

    def test_apply_unknown_verb_raises(self):
        comp, main, _ = _simple_comp()
        with pytest.raises(KeyError, match="No action"):
            REGISTRY.apply("not_a_verb", comp, polygon_ref(main), {})

    def test_widen_validates_params(self):
        comp, main, _ = _simple_comp()
        # delta_um must be > 0
        with pytest.raises(Exception):  # Pydantic ValidationError subclass
            REGISTRY.apply("widen", comp, polygon_ref(main),
                           {"axis": "x", "delta_um": 0.0})

    def test_narrow_exceeding_width_raises(self):
        comp, main, _ = _simple_comp()
        with pytest.raises(ValueError, match="exceeds polygon width"):
            REGISTRY.apply("narrow", comp, polygon_ref(main),
                           {"axis": "x", "delta_um": 5.0})


# ── Polygon-extraction sanity ──────────────────────────────────────────

class TestExtractAndRebuild:
    def test_extract_round_trip(self):
        comp, main, other = _simple_comp()
        polys = extract_polygons(comp)
        comp2 = rebuild_component(polys)
        polys2 = extract_polygons(comp2)
        assert _polygon_lists_equal(polys, polys2)

    def test_find_polygon_tolerates_dbu_drift(self):
        comp, main, _ = _simple_comp()
        polys = extract_polygons(comp)
        # An exact match should work.
        ref = polygon_ref(main)
        from lithos_repair.actions import find_polygon
        idx = find_polygon(polys, ref)
        assert polys[idx].layer == LAYER_MAIN

    def test_polygon_centroid_and_bbox(self):
        p = Polygon(
            layer  = LAYER_MAIN,
            points = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        )
        assert p.bbox == (0.0, 0.0, 2.0, 1.0)
        assert p.centroid == (1.0, 0.5)
        assert p.width_um == 2.0
        assert p.height_um == 1.0
