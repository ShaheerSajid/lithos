"""Resolver: tool-emitted check names → canonical foundry codes via RuleDB."""
from __future__ import annotations

from pathlib import Path

from lithos_core import (
    Constraint,
    ConstraintBranch,
    LayerRef,
    Rule,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)

from lithos_drc.base import DRCViolation
from lithos_drc.resolver import partition_unresolved, resolve_violations


def _seeded_db(path: Path) -> RuleDB:
    """Create a small DB with two rules and a handful of aliases."""
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
    db.upsert_rule(Rule(
        code="M2.S.1",
        category="metal_low",
        usage_class="geometry_primitive",
        short_desc="metal2 minimum spacing",
        constraint=Constraint(
            branches=[ConstraintBranch(
                check=SpacingCheck(
                    layer_a=LayerRef(name="met2"),
                    layer_b=None, op=">=", threshold_um=0.14,
                ),
            )],
            deck_dialect="svrf",
        ),
    ))
    db.upsert_rule(Rule(
        code="M2.W.1",
        category="metal_low",
        usage_class="geometry_primitive",
        short_desc="metal2 minimum width",
        constraint=Constraint(
            branches=[ConstraintBranch(
                check=WidthCheck(
                    target=LayerRef(name="met2"), op=">=", threshold_um=0.14,
                ),
            )],
            deck_dialect="svrf",
        ),
    ))
    db.add_alias("M2.S.1",   code="M2.S.1", source="foundry_code")
    db.add_alias("m2_sp_70", code="M2.S.1", source="deck_rulecheck")
    db.add_alias("M2.W.1",   code="M2.W.1", source="foundry_code")
    return db


def test_resolves_known_aliases(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        violations = [
            DRCViolation(rule="m2_sp_70", layer="met2", value=0.12),
            DRCViolation(rule="M2.W.1",   layer="met2", value=0.13),
        ]
        resolved = resolve_violations(violations, db)
        assert len(resolved) == 2
        assert resolved[0].code        == "M2.S.1"
        assert resolved[0].category    == "metal_low"
        assert resolved[0].usage_class == "geometry_primitive"
        assert resolved[0].unresolved is False
        assert resolved[1].code == "M2.W.1"
    finally:
        db.close()


def test_unresolved_alias_carries_violation(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        v = DRCViolation(rule="unknown_check", layer="met2")
        [r] = resolve_violations([v], db)
        assert r.unresolved is True
        assert r.code is None
        assert r.rule is None
        assert r.violation is v               # original payload preserved
        assert r.category is None
    finally:
        db.close()


def test_partition_unresolved_splits_stream(tmp_path: Path):
    db = _seeded_db(tmp_path / "rules.db")
    try:
        violations = [
            DRCViolation(rule="m2_sp_70"),
            DRCViolation(rule="unknown_a"),
            DRCViolation(rule="M2.W.1"),
            DRCViolation(rule="unknown_b"),
        ]
        resolved = resolve_violations(violations, db)
        known, unknown = partition_unresolved(resolved)
        assert [r.code for r in known] == ["M2.S.1", "M2.W.1"]
        assert [r.violation.rule for r in unknown] == ["unknown_a", "unknown_b"]
    finally:
        db.close()


def test_caching_avoids_duplicate_lookups(tmp_path: Path, monkeypatch):
    """Repeated aliases should resolve via the per-call cache."""
    db = _seeded_db(tmp_path / "rules.db")
    try:
        calls = {"resolve_alias": 0, "get_rule": 0}
        real_resolve = db.resolve_alias
        real_get_rule = db.get_rule

        def _counted_resolve(alias):
            calls["resolve_alias"] += 1
            return real_resolve(alias)

        def _counted_get(code):
            calls["get_rule"] += 1
            return real_get_rule(code)

        monkeypatch.setattr(db, "resolve_alias", _counted_resolve)
        monkeypatch.setattr(db, "get_rule", _counted_get)

        violations = [DRCViolation(rule="m2_sp_70") for _ in range(5)]
        resolve_violations(violations, db)
        assert calls["resolve_alias"] == 1
        assert calls["get_rule"]      == 1
    finally:
        db.close()
