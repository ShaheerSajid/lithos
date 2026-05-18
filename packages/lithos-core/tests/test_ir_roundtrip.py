"""End-to-end round-trip for the lithos-core schema.

Build a Constraint + FixMetadata in Python → save to a fresh SQLite DB →
read back → assert equality. Also exercises alias collision, the category
config matcher, and category-filtered queries.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lithos_core import (
    CategoryConfig,
    CategoryDef,
    Constraint,
    ConstraintBranch,
    FixBranch,
    FixMetadata,
    LayerRef,
    ParallelRunLength,
    Rule,
    RuleDB,
    SpacingCheck,
    WidthBand,
    WidthCheck,
)


def _wide_metal_spacing() -> Constraint:
    """The canonical wide-metal spacing rule used through the design doc:

    M2 spacing ≥ 0.14 µm by default; if both adjacent edges have width
    ≥ 0.3 µm AND parallel-run-length ≥ 1.0 µm, ≥ 0.30 µm.
    """
    met2 = LayerRef(name="met2")
    return Constraint(
        branches=[
            ConstraintBranch(
                predicate=[],
                check=SpacingCheck(
                    layer_a=met2, layer_b=None, op=">=", threshold_um=0.14,
                ),
            ),
            ConstraintBranch(
                predicate=[
                    WidthBand(min_um=0.3),
                    ParallelRunLength(min_um=1.0),
                ],
                check=SpacingCheck(
                    layer_a=met2, layer_b=None, op=">=", threshold_um=0.30,
                ),
            ),
        ],
        deck_dialect="svrf",
    )


def _poly_min_width() -> Constraint:
    return Constraint(
        branches=[
            ConstraintBranch(
                predicate=[],
                check=WidthCheck(
                    target=LayerRef(name="poly"),
                    op=">=",
                    threshold_um=0.15,
                ),
            ),
        ],
        deck_dialect="svrf",
    )


# ── Pydantic IR round-trip ──────────────────────────────────────────────────

def test_constraint_pydantic_json_roundtrip():
    c = _wide_metal_spacing()
    c2 = Constraint.model_validate_json(c.model_dump_json())
    assert c == c2


def test_constraint_pydantic_dict_roundtrip():
    c = _wide_metal_spacing()
    c2 = Constraint.model_validate(c.model_dump())
    assert c == c2


# ── DB round-trip ───────────────────────────────────────────────────────────

def test_pdk_identity(tmp_path: Path):
    with RuleDB(tmp_path / "rules.db") as db:
        db.set_pdk(
            name="test_pdk",
            version="0.0.1",
            ingested_at="2026-05-18T00:00:00Z",
            ingest_tool_versions={"llm": "qwen2.5-coder-3b", "ingest": "0.1.0"},
            deck_files=["decks/svrf.drc"],
            pdf_files=["docs/rule_manual.pdf"],
        )
        assert db.pdk_identity() == ("test_pdk", "0.0.1")


def test_rule_roundtrip(tmp_path: Path):
    db_path = tmp_path / "rules.db"
    rule = Rule(
        code="M2.S.1",
        category="metal_low",
        usage_class="geometry_primitive",
        short_desc="metal2 minimum spacing",
        constraint=_wide_metal_spacing(),
        fix_metadata=FixMetadata(
            intent="prevents litho bridging between adjacent metal2 lines",
            allowed_action_classes=["widen", "shift_orthogonal"],
            forbidden_action_classes=["add_fill"],
            affected_layers=["met2"],
            branches=[
                FixBranch(
                    condition="violating shape is a power rail",
                    forbidden_action_classes=["widen"],
                    notes="shift the neighbour instead",
                ),
            ],
            notes="see also wide-metal rule M2.W.1",
        ),
        provenance={"constraint": "deck", "fix_metadata": "pdf"},
        confidence={"constraint": 1.0, "fix_metadata": 0.85},
        needs_review=False,
    )

    with RuleDB(db_path) as db:
        db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
        db.upsert_rule(rule)
        db.add_alias("M2.S.1",   code="M2.S.1", source="foundry_code")
        db.add_alias("m2_sp_70", code="M2.S.1", source="deck_rulecheck")
        db.add_relation("M2.S.1", "M2.W.1", relation="fix_may_trigger")
        db.set_source(
            code="M2.S.1",
            deck_block="m2_sp = EXTERNAL met2 < 0.14",
            deck_title="M2.S.1: metal2 spacing < 70nm",
            pdf_chunk="M2.S.1 The minimum spacing between adjacent metal2 ...",
            pdf_page=142,
        )

    with RuleDB(db_path) as db:
        loaded = db.get_rule("M2.S.1")
        assert loaded == rule
        assert db.resolve_alias("m2_sp_70") == "M2.S.1"
        assert db.resolve_alias("M2.S.1") == "M2.S.1"
        assert db.resolve_alias("unknown") is None
        assert db.aliases_for("M2.S.1") == [
            ("M2.S.1", "foundry_code"),
            ("m2_sp_70", "deck_rulecheck"),
        ]
        assert db.relations_from("M2.S.1") == [("M2.W.1", "fix_may_trigger")]
        assert db.relations_from("M2.S.1", relation="fix_may_trigger") == [
            ("M2.W.1", "fix_may_trigger"),
        ]
        assert db.relations_from("M2.S.1", relation="see_also") == []
        src = db.get_source("M2.S.1")
        assert src is not None
        assert src["pdf_page"] == 142


def test_alias_collision_raises(tmp_path: Path):
    """Strict alias PK: registering the same alias under two rules raises."""
    with RuleDB(tmp_path / "rules.db") as db:
        db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
        for code in ("A.1", "A.2"):
            db.upsert_rule(Rule(
                code=code,
                category="unknown",
                usage_class="geometry_primitive",
            ))
        db.add_alias("shared_alias", code="A.1", source="manual")
        with pytest.raises(sqlite3.IntegrityError):
            db.add_alias("shared_alias", code="A.2", source="manual")


# ── Category filtering ──────────────────────────────────────────────────────

def test_category_filter_and_coverage(tmp_path: Path):
    with RuleDB(tmp_path / "rules.db") as db:
        db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
        db.upsert_rule(Rule(
            code="P.1",
            category="poly",
            usage_class="geometry_primitive",
            constraint=_poly_min_width(),
        ))
        db.upsert_rule(Rule(
            code="M2.S.1",
            category="metal_low",
            usage_class="geometry_primitive",
            constraint=_wide_metal_spacing(),
        ))
        db.upsert_rule(Rule(
            code="DENS.1",
            category="density",
            usage_class="density",
        ))

        assert db.count_rules() == 3
        assert db.count_rules(category="poly") == 1
        assert db.count_rules(category="metal_low") == 1
        assert db.count_rules(usage_class="geometry_primitive") == 2
        assert db.count_rules(
            usage_class="geometry_primitive", category="poly",
        ) == 1

        poly_rules = list(db.all_rules(category="poly"))
        assert len(poly_rules) == 1 and poly_rules[0].code == "P.1"

        coverage = dict(db.categories())
        assert coverage == {"density": 1, "metal_low": 1, "poly": 1}


# ── Categories config ───────────────────────────────────────────────────────

def test_category_config_matching():
    cfg = CategoryConfig(
        categories=[
            CategoryDef(name="poly", code_prefixes=["P.", "poly."], priority=10),
            CategoryDef(name="metal_low",
                        code_prefixes=["M1.", "M2.", "li.", "li1."],
                        priority=20),
            CategoryDef(name="antenna",
                        code_prefixes=["A.", "antenna."],
                        enabled=False, priority=90),
            CategoryDef(name="catch_all",
                        code_pattern=r"^.*$", priority=999),
        ],
    )

    assert cfg.category_for("P.1")     == "poly"
    assert cfg.category_for("M2.S.1")  == "metal_low"
    assert cfg.category_for("A.1")     == "catch_all"   # antenna disabled
    assert cfg.category_for("ZZ.99")   == "catch_all"
    # Disabling catch_all leaves ZZ.99 → unknown.
    cfg2 = cfg.model_copy(update={
        "categories": [c.model_copy(update={"enabled": False})
                       if c.name == "catch_all" else c
                       for c in cfg.categories],
    })
    assert cfg2.category_for("ZZ.99") == "unknown"


def test_category_config_priority_order():
    """Lower priority wins. Two enabled categories that both claim a code:
    the lower-priority one is returned."""
    cfg = CategoryConfig(
        categories=[
            CategoryDef(name="broad",
                        code_prefixes=["M"], priority=100),
            CategoryDef(name="narrow",
                        code_prefixes=["M2."], priority=10),
        ],
    )
    assert cfg.category_for("M2.S.1") == "narrow"
    assert cfg.category_for("M3.X.1") == "broad"


def test_category_config_yaml_load(tmp_path: Path):
    yaml_text = """\
default_category: unknown
categories:
  - name: poly
    code_prefixes: ["P.", "poly."]
    priority: 10
    description: Poly width, spacing, endcap.
  - name: antenna
    code_prefixes: ["A.", "antenna."]
    enabled: false
    priority: 90
"""
    path = tmp_path / "categories.yaml"
    path.write_text(yaml_text)

    from lithos_core import load_categories
    cfg = load_categories(path)
    assert cfg.default_category == "unknown"
    assert len(cfg.categories) == 2
    assert cfg.by_name("poly").description == "Poly width, spacing, endcap."
    assert cfg.by_name("antenna").enabled is False
    assert cfg.category_for("P.1") == "poly"
    assert cfg.category_for("A.1") == "unknown"        # antenna disabled
