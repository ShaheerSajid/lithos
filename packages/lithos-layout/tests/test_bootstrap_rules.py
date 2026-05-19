"""BootstrapRules: bridge between rule DB and cell generation."""
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
    load_bootstrap_mapping,
)


def _seeded_db(path: Path) -> RuleDB:
    """A small DB with poly/contact/diff rules + asymmetric enclosure values."""
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
    rules = [
        Rule(
            code="PO.W.1", category="poly", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=WidthCheck(target=LayerRef(name="poly"), op=">=", threshold_um=0.15),
            )]),
        ),
        Rule(
            code="PO.S.1", category="poly", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=SpacingCheck(layer_a=LayerRef(name="poly"), op=">=", threshold_um=0.21),
            )]),
        ),
        Rule(
            code="PO.E.1", category="poly", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=EnclosureCheck(
                    inner=LayerRef(name="diff"), outer=LayerRef(name="poly"),
                    op=">=", threshold_um=0.13,
                ),
            )]),
        ),
        Rule(
            code="CO.W.1", category="contact", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17),
            )]),
        ),
        Rule(
            code="CO.S.1", category="contact", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=SpacingCheck(layer_a=LayerRef(name="contact"), op=">=", threshold_um=0.17),
            )]),
        ),
        Rule(
            code="CO.E.D.1", category="contact", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=EnclosureCheck(
                    inner=LayerRef(name="contact"), outer=LayerRef(name="diff"),
                    op=">=", threshold_um=0.04,
                ),
            )]),
        ),
        # Asymmetric: 2-adjacent-edge vs all-sides enclosure of contact by m0.
        Rule(
            code="CO.E.M0.2ADJ", category="contact", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=EnclosureCheck(
                    inner=LayerRef(name="contact"), outer=LayerRef(name="m0"),
                    op=">=", threshold_um=0.08, on_sides="two_adjacent",
                ),
            )]),
        ),
        Rule(
            code="CO.E.M0.ALL", category="contact", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=EnclosureCheck(
                    inner=LayerRef(name="contact"), outer=LayerRef(name="m0"),
                    op=">=", threshold_um=0.0,
                ),
            )]),
        ),
        Rule(
            code="DI.W.1", category="diff", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(
                check=WidthCheck(target=LayerRef(name="diff"), op=">=", threshold_um=0.15),
            )]),
        ),
    ]
    for r in rules:
        db.upsert_rule(r)
    return db


def _metadata(devices: dict | None = None) -> PDKMetadata:
    return PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20), "contact": (66, 44), "m0": (67, 20)},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices=devices or {
            "nmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "nimplant", "bulk_layer": "pwell",
                "nwell": False, "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
            "pmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "pimplant", "bulk_layer": "nwell",
                "nwell": True, "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
        },
    )


def _mapping() -> BootstrapMapping:
    return BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.spacing_min_um":          "PO.S.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "contact.size_um":              "CO.W.1",
        "contact.spacing_um":           "CO.S.1",
        "contact.enclosure_in_diff_um": "CO.E.D.1",
        "contact.enclosure_in_m0_2adj_um": "CO.E.M0.2ADJ",
        "contact.enclosure_in_m0_um":      "CO.E.M0.ALL",
        "diff.width_min_um":            "DI.W.1",
    })


# ── Flat semantic API ──────────────────────────────────────────────────────

def test_get_resolves_through_mapping(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        assert r.get("poly.width_min_um")     == 0.15
        assert r.get("poly.spacing_min_um")   == 0.21
        assert r.get("contact.size_um")       == 0.17
        assert r.get("diff.width_min_um")     == 0.15
    finally:
        db.close()


def test_get_caches_repeated_lookups(tmp_path: Path, monkeypatch):
    """get() should consult the DB once per semantic name."""
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        calls = {"n": 0}
        real = db.get_rule
        def counted(code):
            calls["n"] += 1
            return real(code)
        monkeypatch.setattr(db, "get_rule", counted)
        for _ in range(5):
            r.get("poly.width_min_um")
        assert calls["n"] == 1
    finally:
        db.close()


def test_unmapped_semantic_name_raises(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        with pytest.raises(KeyError, match="No bootstrap mapping"):
            r.get("m2.width_min_um")
    finally:
        db.close()


def test_mapped_but_missing_rule_raises(tmp_path: Path):
    """Mapping points at a rule code that's not in the DB."""
    db = _seeded_db(tmp_path / "rules.db")
    try:
        m = BootstrapMapping(mapping={"phantom.x": "ZZ.99"})
        r = BootstrapRules(_metadata(), db, m)
        with pytest.raises(KeyError, match="no such rule"):
            r.get("phantom.x")
    finally:
        db.close()


# ── Dict-section compatibility for ported code ─────────────────────────────

def test_dict_section_works_via_attr(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        assert r.poly["width_min_um"]    == 0.15
        assert r.contact["size_um"]      == 0.17
        assert r.diff["width_min_um"]    == 0.15
    finally:
        db.close()


def test_dict_section_get_returns_default(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        assert r.poly.get("nonexistent_key", 42) == 42
    finally:
        db.close()


def test_dict_section_contains(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        assert "width_min_um" in r.poly
        assert "nonexistent" not in r.poly
    finally:
        db.close()


# ── Asymmetric enclosure ───────────────────────────────────────────────────

def test_enclosure_returns_adj2_and_all(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        adj, opp = r.enclosure("contact", "enclosure_in_m0")
        assert adj == 0.08
        assert opp == 0.0
    finally:
        db.close()


def test_enclosure_symmetric_fallback(tmp_path: Path):
    """Mapping with only the ``_um`` key (no ``_2adj_um``) → symmetric."""
    db = _seeded_db(tmp_path / "rules.db")
    try:
        m = BootstrapMapping(mapping={
            "contact.enclosure_in_diff_um": "CO.E.D.1",
        })
        r = BootstrapRules(_metadata(), db, m)
        adj, opp = r.enclosure("contact", "enclosure_in_diff")
        assert adj == 0.04 and opp == 0.04
    finally:
        db.close()


# ── Forwards to metadata ──────────────────────────────────────────────────

def test_layer_and_device_forward(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        r = BootstrapRules(_metadata(), db, _mapping())
        assert r.layer("poly") == (66, 20)
        nmos = r.device("nmos")
        assert nmos["gate_layer"] == "poly"
        assert nmos["nwell"] is False
        assert r.name == "t"
        assert r.mfg_grid == 0.005
    finally:
        db.close()


# ── YAML loader ────────────────────────────────────────────────────────────

def test_load_bootstrap_mapping_flat(tmp_path: Path):
    yaml_text = """\
mapping:
  poly.width_min_um: PO.W.1
  diff.width_min_um: DI.W.1
"""
    p = tmp_path / "bootstrap.yaml"
    p.write_text(yaml_text)
    m = load_bootstrap_mapping(p)
    assert m.mapping == {"poly.width_min_um": "PO.W.1", "diff.width_min_um": "DI.W.1"}


def test_load_bootstrap_mapping_nested_flattens(tmp_path: Path):
    yaml_text = """\
mapping:
  poly:
    width_min_um: PO.W.1
    spacing_min_um: PO.S.1
  contact:
    size_um: CO.W.1
"""
    p = tmp_path / "bootstrap.yaml"
    p.write_text(yaml_text)
    m = load_bootstrap_mapping(p)
    assert m.mapping == {
        "poly.width_min_um":   "PO.W.1",
        "poly.spacing_min_um": "PO.S.1",
        "contact.size_um":     "CO.W.1",
    }
