"""lithos_layout.cells.vias — via and via-stack cell factories."""
from __future__ import annotations

from pathlib import Path

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
    via_diff_m0,
    via_m0_m1,
    via_m0_m2,
    via_m1_m2,
    via_poly_m0,
    via_poly_m1,
    via_poly_m2,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

# Layer assignments — distinct (gds_layer, datatype) per abstract name so
# we can confidently assert which layers a cell drew on.
_LAYERS = {
    "poly":      (66, 20),
    "diff":      (65, 20),
    "contact":   (66, 44),
    "m0":        (67, 20),
    "m1":        (68, 20),
    "m2":        (69, 20),
    "via_m0_m1": (67, 44),
    "via_m1_m2": (68, 44),
}


def _seeded_db(path: Path) -> RuleDB:
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-19T00:00:00Z")
    for code, check in [
        ("CO.W.1",       WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.S.1",       SpacingCheck(layer_a=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.E.P.1",     EnclosureCheck(inner=LayerRef(name="contact"),
                                        outer=LayerRef(name="poly"),
                                        op=">=", threshold_um=0.05)),
        ("CO.E.P.2ADJ",  EnclosureCheck(inner=LayerRef(name="contact"),
                                        outer=LayerRef(name="poly"),
                                        op=">=", threshold_um=0.08,
                                        on_sides="two_adjacent")),
        ("CO.E.M0.1",    EnclosureCheck(inner=LayerRef(name="contact"),
                                        outer=LayerRef(name="m0"),
                                        op=">=", threshold_um=0.0)),
        ("CO.E.M0.2ADJ", EnclosureCheck(inner=LayerRef(name="contact"),
                                        outer=LayerRef(name="m0"),
                                        op=">=", threshold_um=0.08,
                                        on_sides="two_adjacent")),
        ("M0.W.1",       WidthCheck(target=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M1.W.1",       WidthCheck(target=LayerRef(name="m1"), op=">=", threshold_um=0.14)),
        ("M2.W.1",       WidthCheck(target=LayerRef(name="m2"), op=">=", threshold_um=0.14)),
        ("V01.W.1",      WidthCheck(target=LayerRef(name="via_m0_m1"), op=">=", threshold_um=0.17)),
        ("V01.E.M1.2A",  EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                        outer=LayerRef(name="m1"),
                                        op=">=", threshold_um=0.06,
                                        on_sides="two_adjacent")),
        ("V12.W.1",      WidthCheck(target=LayerRef(name="via_m1_m2"), op=">=", threshold_um=0.15)),
        ("V12.E.M1.2A",  EnclosureCheck(inner=LayerRef(name="via_m1_m2"),
                                        outer=LayerRef(name="m1"),
                                        op=">=", threshold_um=0.055,
                                        on_sides="two_adjacent")),
        ("V12.E.M2.2A",  EnclosureCheck(inner=LayerRef(name="via_m1_m2"),
                                        outer=LayerRef(name="m2"),
                                        op=">=", threshold_um=0.055,
                                        on_sides="two_adjacent")),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    return db


def _metadata(layers: dict | None = None) -> PDKMetadata:
    return PDKMetadata(
        name="t", version="0",
        layers=layers or _LAYERS,
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={},
    )


def _mapping() -> BootstrapMapping:
    return BootstrapMapping(mapping={
        "contact.size_um":                       "CO.W.1",
        "contact.spacing_um":                    "CO.S.1",
        "contact.poly_enclosure_um":             "CO.E.P.1",
        "contact.poly_enclosure_2adj_um":        "CO.E.P.2ADJ",
        "contact.enclosure_in_m0_um":            "CO.E.M0.1",
        "contact.enclosure_in_m0_2adj_um":       "CO.E.M0.2ADJ",
        "m0.width_min_um":                       "M0.W.1",
        "m1.width_min_um":                       "M1.W.1",
        "m2.width_min_um":                       "M2.W.1",
        "via_m0_m1.size_um":                     "V01.W.1",
        "m1.enclosure_of_via_m0_m1_2adj_um":     "V01.E.M1.2A",
        "via_m1_m2.size_um":                     "V12.W.1",
        "via_m1_m2.enclosure_in_m1_2adj_um":     "V12.E.M1.2A",
        "m2.enclosure_of_via_m1_m2_2adj_um":     "V12.E.M2.2A",
    })


def _rules(tmp_path: Path, *, layers: dict | None = None) -> BootstrapRules:
    db = _seeded_db(tmp_path / "rules.db")
    return BootstrapRules(_metadata(layers), db, _mapping())


def _layers_present(component) -> set[tuple[int, int]]:
    """All (gds_layer, datatype) pairs that carry at least one shape, after
    flattening any sub-cell references. ``Component.flatten`` is in-place
    (returns None) in this gdsfactory build.
    """
    component.flatten()
    kc = component.kdb_cell
    layout = kc.layout()
    present: set[tuple[int, int]] = set()
    for layer_idx in range(layout.layers()):
        if any(True for _ in kc.each_shape(layer_idx)):
            info = layout.get_info(layer_idx)
            present.add((info.layer, info.datatype))
    return present


# ── Single-cut cells ───────────────────────────────────────────────────────

def test_via_poly_m0_draws_poly_contact_m0(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = via_poly_m0(r)
        present = _layers_present(c)
        assert _LAYERS["poly"]    in present
        assert _LAYERS["contact"] in present
        assert _LAYERS["m0"]      in present
    finally:
        r.db.close()


def test_via_diff_m0_omits_poly(tmp_path: Path):
    """Diff contact draws the contact cut + m0 pad, but NOT a poly pad."""
    r = _rules(tmp_path)
    try:
        c = via_diff_m0(r)
        present = _layers_present(c)
        assert _LAYERS["contact"] in present
        assert _LAYERS["m0"]      in present
        assert _LAYERS["poly"] not in present
    finally:
        r.db.close()


def test_via_m0_m1_draws_cut_and_m1(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = via_m0_m1(r)
        present = _layers_present(c)
        assert _LAYERS["via_m0_m1"] in present
        assert _LAYERS["m1"]        in present
        # Doesn't touch m0 or contact.
        assert _LAYERS["m0"]      not in present
        assert _LAYERS["contact"] not in present
    finally:
        r.db.close()


def test_via_m0_m1_collapsed_skips_cut(tmp_path: Path):
    """When the PDK collapses m0 onto m1 (same gds tuple), the cut is omitted."""
    collapsed_layers = dict(_LAYERS)
    # Force m0 and m1 to the same GDS coordinates.
    collapsed_layers["m0"] = collapsed_layers["m1"]
    r = _rules(tmp_path, layers=collapsed_layers)
    try:
        assert r.m0_is_m1 is True
        c = via_m0_m1(r)
        present = _layers_present(c)
        # Only the m1 pad — no cut layer.
        assert collapsed_layers["m1"] in present
        # The cut layer (via_m0_m1) should not appear.
        assert _LAYERS["via_m0_m1"] not in present
    finally:
        r.db.close()


def test_via_m1_m2_draws_cut_m1_m2(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = via_m1_m2(r)
        present = _layers_present(c)
        assert _LAYERS["via_m1_m2"] in present
        assert _LAYERS["m1"]        in present
        assert _LAYERS["m2"]        in present
    finally:
        r.db.close()


# ── Composite stacks ──────────────────────────────────────────────────────

def test_via_poly_m1_includes_full_stack(tmp_path: Path):
    """Poly → m1 stack should pull in poly+contact+m0+via_m0_m1+m1 layers."""
    r = _rules(tmp_path)
    try:
        c = via_poly_m1(r)
        present = _layers_present(c)
        for k in ("poly", "contact", "m0", "via_m0_m1", "m1"):
            assert _LAYERS[k] in present, f"{k} missing from via_poly_m1 stack"
    finally:
        r.db.close()


def test_via_poly_m2_reaches_m2(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = via_poly_m2(r)
        present = _layers_present(c)
        for k in ("poly", "contact", "m0", "via_m0_m1", "m1", "via_m1_m2", "m2"):
            assert _LAYERS[k] in present, f"{k} missing from via_poly_m2 stack"
    finally:
        r.db.close()


def test_via_m0_m2_skips_poly_and_contact(tmp_path: Path):
    """m0 → m2 stack does NOT include a poly/diff contact."""
    r = _rules(tmp_path)
    try:
        c = via_m0_m2(r)
        present = _layers_present(c)
        for k in ("via_m0_m1", "m1", "via_m1_m2", "m2"):
            assert _LAYERS[k] in present, f"{k} missing from via_m0_m2 stack"
        assert _LAYERS["poly"]    not in present
        assert _LAYERS["contact"] not in present
    finally:
        r.db.close()
