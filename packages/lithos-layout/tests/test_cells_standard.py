"""lithos_layout.cells.standard — pure geometry helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_core import (
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    LayerRef,
    PDKMetadata,
    Rule,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)

from lithos_layout import (
    BootstrapMapping,
    BootstrapRules,
    transistor_geom,
)
from lithos_layout.cells.standard import (
    _diff_y,
    _gate_x,
    _inter_cell_gap,
    _rect,
    _routing_gap,
    _sd_x,
    _snap,
)


# ── Fixture ────────────────────────────────────────────────────────────────

def _rules(tmp_path: Path) -> BootstrapRules:
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-19T00:00:00Z")
    for code, check in [
        ("PO.W.1",  WidthCheck(target=LayerRef(name="poly"), op=">=", threshold_um=0.15)),
        ("PO.E.1",  EnclosureCheck(inner=LayerRef(name="diff"), outer=LayerRef(name="poly"),
                                   op=">=", threshold_um=0.13)),
        ("DI.W.1",  WidthCheck(target=LayerRef(name="diff"), op=">=", threshold_um=0.15)),
        ("DI.S.1",  SpacingCheck(layer_a=LayerRef(name="diff"), op=">=", threshold_um=0.27)),
        ("DI.E.1",  EnclosureCheck(inner=LayerRef(name="poly"), outer=LayerRef(name="diff"),
                                   op=">=", threshold_um=0.25)),
        ("CO.W.1",  WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.S.1",  SpacingCheck(layer_a=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.E.D.1", EnclosureCheck(inner=LayerRef(name="contact"), outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.04)),
        ("M0.W.1",  WidthCheck(target=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M0.S.1",  SpacingCheck(layer_a=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))

    metadata = PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20),
                "contact": (66, 44), "m0": (67, 20)},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={
            "nmos": {"w_finger_max_um": 5.0, "sd_length_min_um": 0.29},
        },
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "diff.width_min_um":            "DI.W.1",
        "diff.spacing_min_um":          "DI.S.1",
        "diff.extension_past_poly_um":  "DI.E.1",
        "contact.size_um":              "CO.W.1",
        "contact.spacing_um":           "CO.S.1",
        "contact.enclosure_in_diff_um": "CO.E.D.1",
        "m0.width_min_um":              "M0.W.1",
        "m0.spacing_min_um":            "M0.S.1",
    })
    return BootstrapRules(metadata, db, mapping)


# ── _snap ──────────────────────────────────────────────────────────────────

def test_snap_rounds_to_grid():
    assert _snap(0.0072, 0.005) == pytest.approx(0.005)
    assert _snap(0.0078, 0.005) == pytest.approx(0.010)
    # Exact grid points stay put.
    assert _snap(0.250, 0.005) == pytest.approx(0.250)


def test_snap_zero_grid_passthrough():
    """A zero or negative grid disables snapping."""
    assert _snap(0.123456, 0.0) == 0.123456


# ── _sd_x / _gate_x ────────────────────────────────────────────────────────

def test_sd_x_without_rules_returns_full_region(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        g = transistor_geom(0.6, 0.15, "nmos", r)
        x0, x1 = _sd_x(0, g, rules=None)
        # Full S/D region: [0, sd_length_um].
        assert x0 == pytest.approx(0.0)
        assert x1 == pytest.approx(g.sd_length_um)
    finally:
        r.db.close()


def test_sd_x_with_rules_centres_on_contact(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        g = transistor_geom(0.6, 0.15, "nmos", r)
        x0, x1 = _sd_x(0, g, rules=r)
        # Centre is sd_length_um / 2; half-width is contact_size / 2 = 0.085.
        c_half = r.contact["size_um"] / 2
        cx     = g.sd_length_um / 2
        assert x0 == pytest.approx(cx - c_half)
        assert x1 == pytest.approx(cx + c_half)
    finally:
        r.db.close()


def test_gate_x_strides_by_finger_pitch(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        g = transistor_geom(0.6, 0.15, "nmos", r, )
        # finger i sits after (i+1) S/D regions and i previous fingers.
        x0_0, x1_0 = _gate_x(0, g)
        assert x0_0 == pytest.approx(g.sd_length_um)
        assert x1_0 == pytest.approx(g.sd_length_um + g.l_um)
    finally:
        r.db.close()


# ── _diff_y ────────────────────────────────────────────────────────────────

def test_diff_y_brackets_with_endcap(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        g = transistor_geom(0.52, 0.15, "nmos", r)
        endcap = r.get("poly.endcap_over_diff_um")
        y0, y1 = _diff_y(g, r)
        assert y0 == pytest.approx(endcap)
        assert y1 == pytest.approx(endcap + g.w_finger_um)
    finally:
        r.db.close()


# ── _inter_cell_gap ────────────────────────────────────────────────────────

def test_inter_cell_gap_nonneg(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        gap = _inter_cell_gap(r)
        # With endcap=0.13 and diff_min_sep=0.27:
        # gap = max(0, 0.27 - 2*0.13) = max(0, 0.01) = 0.01
        assert gap == pytest.approx(0.01)
        assert gap >= 0.0
    finally:
        r.db.close()


# ── _routing_gap ───────────────────────────────────────────────────────────

def test_routing_gap_fits_one_m0_track(tmp_path: Path):
    """Returned gap must accommodate 2*spacing + 1*width of an m0 track."""
    r = _rules(tmp_path)
    try:
        gap     = _routing_gap(r)
        endcap  = r.get("poly.endcap_over_diff_um")
        ext     = r.get("diff.extension_past_poly_um")
        m0_sp   = r.get("m0.spacing_min_um")
        m0_w    = r.get("m0.width_min_um")
        needed  = 2 * m0_sp + m0_w
        # pmos_y - nmos_y_top after subtracting 2*ext - 2*endcap >= needed.
        assert gap >= needed - 2 * endcap + 2 * ext
    finally:
        r.db.close()


# ── _rect ──────────────────────────────────────────────────────────────────

class _FakeComp:
    """Captures add_polygon calls for tests that don't want a real component."""
    def __init__(self) -> None:
        self.polys: list[tuple] = []
    def add_polygon(self, pts: list, layer) -> None:
        self.polys.append((tuple(pts), layer))


def test_rect_snaps_and_emits_polygon():
    c = _FakeComp()
    _rect(c, 0.0021, 0.4972, 0.0001, 0.1003, layer=(1, 0), snap_grid=0.005)
    assert len(c.polys) == 1
    pts, layer = c.polys[0]
    assert layer == (1, 0)
    # All corner coords land on the 5 nm grid.
    for x, y in pts:
        assert round(x / 0.005) * 0.005 == pytest.approx(x)
        assert round(y / 0.005) * 0.005 == pytest.approx(y)
