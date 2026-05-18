"""Joiner: merge ParsedRule + FixMetadata + Chunks into Rule rows."""
from __future__ import annotations

from lithos_core import (
    CategoryConfig,
    CategoryDef,
    Constraint,
    ConstraintBranch,
    FixMetadata,
    LayerRef,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.chunker import Chunk
from lithos_ingest.joiner import (
    cross_validate,
    join_rule,
    join_rules,
    layers_in_constraint,
    usage_class_from_constraint,
)
from lithos_ingest.parsers.types import ParsedRule


def _met2_spacing_rule() -> ParsedRule:
    return ParsedRule(
        code="M2.S.1",
        title="M2.S.1: metal2 minimum spacing",
        aliases=[("M2.S.1", "foundry_code"), ("M2.S.1: metal2 minimum spacing", "deck_rulecheck")],
        constraint=Constraint(
            branches=[ConstraintBranch(check=SpacingCheck(
                layer_a=LayerRef(name="met2"), layer_b=None,
                op=">=", threshold_um=0.14,
            ))],
            deck_dialect="svrf",
        ),
        deck_block="RULECHECK \"M2.S.1...\" { EXTERNAL met2 < 0.14 }",
    )


def _poly_width_rule() -> ParsedRule:
    return ParsedRule(
        code="PO.W.1",
        title="PO.W.1: poly minimum width",
        aliases=[("PO.W.1", "foundry_code")],
        constraint=Constraint(
            branches=[ConstraintBranch(check=WidthCheck(
                target=LayerRef(name="poly"), op=">=", threshold_um=0.15,
            ))],
            deck_dialect="svrf",
        ),
        deck_block="RULECHECK \"PO.W.1...\" { WIDTH poly < 0.15 }",
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_layers_in_constraint_simple():
    parsed = _met2_spacing_rule()
    assert layers_in_constraint(parsed.constraint) == {"met2"}


def test_layers_in_constraint_none():
    assert layers_in_constraint(None) == set()


def test_usage_class_geometry_primitive():
    parsed = _met2_spacing_rule()
    assert usage_class_from_constraint(parsed.constraint) == "geometry_primitive"


def test_usage_class_unknown_when_empty():
    assert usage_class_from_constraint(None) == "unknown"


# ── Cross-validation ────────────────────────────────────────────────────────

def test_cross_validate_no_fix_metadata_is_clean():
    assert cross_validate(_met2_spacing_rule(), None) == []


def test_cross_validate_matching_layers_clean():
    fix = FixMetadata(
        intent="prevents litho bridging",
        allowed_action_classes=["widen"],
        affected_layers=["met2"],
    )
    assert cross_validate(_met2_spacing_rule(), fix) == []


def test_cross_validate_layer_mismatch_flagged():
    fix = FixMetadata(
        intent="prevents litho bridging",
        allowed_action_classes=["widen"],
        affected_layers=["poly"],          # wrong layer for an M2 rule
    )
    issues = cross_validate(_met2_spacing_rule(), fix)
    assert any("layer mismatch" in m for m in issues)


def test_cross_validate_empty_intent_flagged():
    fix = FixMetadata(
        intent="   ",                       # whitespace-only
        allowed_action_classes=["widen"],
        affected_layers=["met2"],
    )
    issues = cross_validate(_met2_spacing_rule(), fix)
    assert any("intent is empty" in m for m in issues)


def test_cross_validate_no_actions_flagged():
    fix = FixMetadata(
        intent="prevents bridging",
        allowed_action_classes=[],
        forbidden_action_classes=[],
        affected_layers=["met2"],
    )
    issues = cross_validate(_met2_spacing_rule(), fix)
    assert any("no allowed nor forbidden action" in m
               or "neither allowed nor forbidden" in m
               for m in issues)


def test_cross_validate_collects_all_issues():
    fix = FixMetadata(
        intent="",
        allowed_action_classes=[],
        forbidden_action_classes=[],
        affected_layers=["poly"],
    )
    issues = cross_validate(_met2_spacing_rule(), fix)
    # All three checks fire.
    assert len(issues) == 3


# ── join_rule ───────────────────────────────────────────────────────────────

def test_join_rule_deck_only():
    parsed = _met2_spacing_rule()
    result = join_rule(parsed)
    assert result.rule.code == "M2.S.1"
    assert result.rule.category == "unknown"
    assert result.rule.usage_class == "geometry_primitive"
    assert result.rule.constraint == parsed.constraint
    assert result.rule.fix_metadata is None
    assert result.rule.needs_review is False
    assert result.mismatches == []
    assert result.rule.provenance == {"constraint": "deck"}
    assert result.rule.confidence == {"constraint": 1.0}


def test_join_rule_with_fix_metadata():
    parsed = _met2_spacing_rule()
    fix = FixMetadata(
        intent="prevents litho bridging",
        allowed_action_classes=["widen", "shift_orthogonal"],
        affected_layers=["met2"],
    )
    result = join_rule(parsed, fix, category="metal_low")
    assert result.rule.category == "metal_low"
    assert result.rule.fix_metadata == fix
    assert result.rule.needs_review is False
    assert result.rule.provenance["fix_metadata"] == "llm"
    assert result.rule.confidence["fix_metadata"] == 0.85


def test_join_rule_with_mismatch_halves_confidence():
    parsed = _met2_spacing_rule()
    fix = FixMetadata(
        intent="x",
        allowed_action_classes=["widen"],
        affected_layers=["poly"],          # mismatched layer
    )
    result = join_rule(parsed, fix, fix_confidence=0.9)
    assert result.rule.needs_review is True
    assert len(result.mismatches) == 1
    assert result.rule.confidence["fix_metadata"] == 0.45  # halved


def test_join_rule_attaches_chunk():
    parsed = _met2_spacing_rule()
    chunk = Chunk(
        code="M2.S.1", text="M2.S.1 the minimum spacing...", page=42,
        span=(0, 100), anchor=0,
    )
    result = join_rule(parsed, None, chunk=chunk)
    assert result.rule_source_chunk is chunk


# ── join_rules bulk ─────────────────────────────────────────────────────────

def test_join_rules_iterates_and_resolves_categories():
    cfg = CategoryConfig(categories=[
        CategoryDef(name="metal_low", code_prefixes=["M2."], priority=10),
        CategoryDef(name="poly",      code_prefixes=["PO."], priority=20),
    ])
    parsed = [_met2_spacing_rule(), _poly_width_rule()]
    results = list(join_rules(parsed, categories=cfg))
    assert [r.rule.code for r in results] == ["M2.S.1", "PO.W.1"]
    assert results[0].rule.category == "metal_low"
    assert results[1].rule.category == "poly"


def test_join_rules_per_code_fix_metadata_and_chunks():
    parsed = [_met2_spacing_rule(), _poly_width_rule()]
    fix_meta = {
        "M2.S.1": FixMetadata(
            intent="prevents litho bridging",
            allowed_action_classes=["widen"],
            affected_layers=["met2"],
        ),
        # PO.W.1: deliberately no entry → falls through to deck-only.
    }
    chunks = {
        "M2.S.1": [Chunk(code="M2.S.1", text="excerpt", page=5,
                         span=(0, 10), anchor=0)],
    }
    results = list(join_rules(parsed, fix_meta, chunks=chunks))
    assert results[0].rule.fix_metadata is not None
    assert results[0].rule_source_chunk is not None
    assert results[0].rule_source_chunk.page == 5
    assert results[1].rule.fix_metadata is None
    assert results[1].rule_source_chunk is None
