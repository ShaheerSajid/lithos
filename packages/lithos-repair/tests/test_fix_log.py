"""Tests for :mod:`lithos_repair.fix_log`.

Covers schema setup, ``record``, query filters, and the ``JOIN
rule_alias`` acceptance criterion (M4 acceptance: "rows are queryable
via ``JOIN rule_alias``").
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lithos_core import (
    Constraint,
    ConstraintBranch,
    LayerRef,
    Rule,
    RuleDB,
    WidthCheck,
)
from lithos_repair import (
    FixLog,
    FixOutcome,
    FixSource,
    Polygon,
    ViolationContext,
    WidenParams,
    polygon_ref,
)


# ── Fixtures ────────────────────────────────────────────────────────────

def _shared_db(tmp_path: Path) -> Path:
    """Build a tiny RuleDB with one rule + aliases the JOIN tests use."""
    path = tmp_path / "shared.db"
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="sky130A", version="1.0.5", ingested_at="2026-05-21T00:00:00Z")
    db.upsert_rule(Rule(
        code        = "poly.1a",
        category    = "geom",
        usage_class = "geometry_primitive",
        constraint  = Constraint(branches=[ConstraintBranch(
            check=WidthCheck(target=LayerRef(name="poly"),
                             op=">=", threshold_um=0.15),
        )]),
    ))
    # Aliases the DRC backend might emit for that canonical code.
    db.add_alias("poly.1a",  "poly.1a", source="manual")       # type: ignore[arg-type]
    db.add_alias("po.w.1",   "poly.1a", source="manual")       # type: ignore[arg-type]
    db.add_alias("PO_WIDTH", "poly.1a", source="manual")       # type: ignore[arg-type]
    db.close()
    return path


def _ctx() -> ViolationContext:
    poly = Polygon(layer=(66, 20),
                   points=((0.0, 0.0), (0.5, 0.0), (0.5, 0.2), (0.0, 0.2)))
    return ViolationContext(
        rule="poly.1a", layer_name="poly", primary=poly, on_grid=True,
    )


# ── Lifecycle + schema ─────────────────────────────────────────────────

class TestLifecycle:
    def test_open_creates_schema(self, tmp_path: Path):
        path = tmp_path / "log.db"
        with FixLog(path):
            pass
        # Inspect the file directly.
        with sqlite3.connect(path) as raw:
            tables = [r[0] for r in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            assert "fix_log" in tables

    def test_double_open_is_idempotent(self, tmp_path: Path):
        path = tmp_path / "log.db"
        log = FixLog(path)
        log.open()
        log.open()  # opening twice should not crash
        log.close()


# ── record + query ──────────────────────────────────────────────────────

class TestRecord:
    def test_single_row_inserted(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            row_id = log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="poly.1a", rule_code="poly.1a",
                violation_x=0.25, violation_y=0.1,
                context=_ctx(),
                action_verb="widen",
                params=WidenParams(axis="x", delta_um=0.02),
                outcome=FixOutcome.APPLIED,
                pre_count=10, post_count=9,
                source=FixSource.MANUAL,
            )
            assert row_id == 1
            assert log.count() == 1
            rows = log.all_rows()
            assert rows[0].rule_code  == "poly.1a"
            assert rows[0].action_verb == "widen"
            assert rows[0].outcome     == "applied"
            assert rows[0].source      == "manual"
            assert rows[0].delta_violations == -1
            # context_json + params_json round-trip through JSON.
            assert "poly.1a" in rows[0].context_json
            params = json.loads(rows[0].params_json)
            assert params["axis"] == "x"
            assert params["delta_um"] == pytest.approx(0.02)

    def test_rule_code_defaults_to_raw_when_unresolved(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            log.record(
                pdk="x", cell="c",
                rule_raw="MYSTERY", rule_code=None,
                violation_x=0.0, violation_y=0.0,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.01},
                outcome=FixOutcome.APPLIED,
                pre_count=1, post_count=0,
                source="manual",
            )
            row = log.all_rows()[0]
            assert row.rule_raw  == "MYSTERY"
            assert row.rule_code == "MYSTERY"     # fallback

    def test_string_enums_accepted(self, tmp_path: Path):
        """``outcome`` / ``source`` accept raw strings as well as Enums."""
        with FixLog(tmp_path / "log.db") as log:
            log.record(
                pdk="x", cell="c",
                rule_raw="r", rule_code="r",
                violation_x=0.0, violation_y=0.0,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.01},
                outcome="applied", source="llm",
                pre_count=2, post_count=1,
            )
            row = log.all_rows()[0]
            assert row.source == "llm"

    def test_timestamp_override_for_determinism(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            log.record(
                pdk="x", cell="c",
                rule_raw="r", rule_code="r",
                violation_x=0.0, violation_y=0.0,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.01},
                outcome=FixOutcome.APPLIED,
                pre_count=1, post_count=0,
                source=FixSource.MANUAL,
                timestamp="2022-09-01T00:00:00Z",
            )
            assert log.all_rows()[0].timestamp == "2022-09-01T00:00:00Z"


class TestQuery:
    def _seed(self, log: FixLog) -> None:
        for verb, source, outcome, rule in [
            ("widen",   FixSource.MANUAL, FixOutcome.APPLIED, "poly.1a"),
            ("shift_n", FixSource.LLM,    FixOutcome.APPLIED, "poly.1a"),
            ("widen",   FixSource.LLM,    FixOutcome.FAILED,  "m1.2"),
        ]:
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw=rule, rule_code=rule,
                violation_x=0.0, violation_y=0.0,
                context=_ctx(),
                action_verb=verb,
                params={"axis": "x", "delta_um": 0.01}
                if verb == "widen" else {"delta_um": 0.02},
                outcome=outcome,
                pre_count=5, post_count=4,
                source=source,
            )

    def test_filter_by_source(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            self._seed(log)
            llm_rows = log.query(source="llm")
            assert len(llm_rows) == 2
            assert all(r.source == "llm" for r in llm_rows)

    def test_filter_by_action_verb_and_outcome(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            self._seed(log)
            rows = log.query(action_verb="widen", outcome="applied")
            assert len(rows) == 1
            assert rows[0].source == "manual"

    def test_filter_by_rule_code(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            self._seed(log)
            rows = log.query(rule_code="poly.1a")
            assert len(rows) == 2

    def test_clear(self, tmp_path: Path):
        with FixLog(tmp_path / "log.db") as log:
            self._seed(log)
            assert log.count() == 3
            assert log.clear() == 3
            assert log.count() == 0


# ── JOIN rule_alias (M4 acceptance) ────────────────────────────────────

class TestJoinRuleAlias:
    """M4 acceptance: rows in ``fix_log`` are queryable via
    ``JOIN rule_alias`` to recover every tool-emitted name for the
    canonical rule code stored on the log row."""

    def test_aliases_for_known_code(self, tmp_path: Path):
        path = _shared_db(tmp_path)
        with FixLog(path) as log:
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="po.w.1", rule_code="poly.1a",
                violation_x=0.25, violation_y=0.1,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.02},
                outcome=FixOutcome.APPLIED,
                pre_count=2, post_count=1,
                source=FixSource.LLM,
            )
            aliases = log.aliases_for("poly.1a")
            assert set(aliases) == {"poly.1a", "po.w.1", "PO_WIDTH"}

    def test_left_join_returns_one_row_per_alias(self, tmp_path: Path):
        path = _shared_db(tmp_path)
        with FixLog(path) as log:
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="PO_WIDTH", rule_code="poly.1a",
                violation_x=0.0, violation_y=0.0,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.01},
                outcome=FixOutcome.APPLIED,
                pre_count=1, post_count=0,
                source=FixSource.MANUAL,
            )
            rows = log.join_rule_alias(pdk="sky130A")
            # 3 aliases × 1 fix_log row = 3 joined rows.
            assert len(rows) == 3
            assert {r["alias"] for r in rows} == {"poly.1a", "po.w.1", "PO_WIDTH"}
            assert all(r["rule_code"] == "poly.1a" for r in rows)
            assert all(r["rule_raw"]  == "PO_WIDTH" for r in rows)

    def test_join_when_code_has_no_aliases(self, tmp_path: Path):
        path = _shared_db(tmp_path)
        with FixLog(path) as log:
            log.record(
                pdk="sky130A", cell="x",
                rule_raw="ORPHAN", rule_code="ORPHAN",
                violation_x=0.0, violation_y=0.0,
                context=_ctx(), action_verb="widen",
                params={"axis": "x", "delta_um": 0.01},
                outcome=FixOutcome.APPLIED,
                pre_count=1, post_count=0,
                source=FixSource.MANUAL,
            )
            rows = log.join_rule_alias(pdk="sky130A")
            assert len(rows) == 1
            assert rows[0]["alias"] is None       # LEFT JOIN → NULL


# ── Hand-driven session (M4 acceptance, integration form) ──────────────

class TestHandDrivenSession:
    """Walks a tiny scripted "manual repair" session through FixLog,
    then asserts the populated table reads back correctly. Demonstrates
    the M4 acceptance criterion end-to-end."""

    def test_three_step_session(self, tmp_path: Path):
        db_path = _shared_db(tmp_path)
        with FixLog(db_path) as log:
            # Step 1: agent applies widen, violation count drops.
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="po.w.1", rule_code="poly.1a",
                violation_x=0.25, violation_y=0.1,
                context=_ctx(),
                action_verb="widen",
                params=WidenParams(axis="x", delta_um=0.02),
                outcome=FixOutcome.APPLIED,
                pre_count=3, post_count=2,
                source=FixSource.LLM,
            )
            # Step 2: agent tries shift_n, runs into a neighbour.
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="m1.2", rule_code=None,           # unresolved
                violation_x=0.5, violation_y=0.5,
                context=_ctx(), action_verb="shift_n",
                params={"delta_um": 0.05},
                outcome=FixOutcome.FAILED,
                pre_count=2, post_count=3,                  # made it worse
                source=FixSource.LLM,
            )
            # Step 3: human takes over.
            log.record(
                pdk="sky130A", cell="inverter",
                rule_raw="poly.1a", rule_code="poly.1a",
                violation_x=0.25, violation_y=0.1,
                context=_ctx(), action_verb="extend",
                params={"side": "e", "delta_um": 0.03},
                outcome=FixOutcome.APPLIED,
                pre_count=3, post_count=1,
                source=FixSource.MANUAL,
            )

            # All three rows are queryable.
            assert log.count() == 3

            # JOIN rule_alias: rows whose rule_code resolves should appear
            # 3 × number-of-aliases times (3 aliases for poly.1a → 6 rows).
            joined = log.join_rule_alias(pdk="sky130A")
            poly_rows = [r for r in joined if r["rule_code"] == "poly.1a"]
            assert len(poly_rows) == 6     # 2 fix_log rows × 3 aliases
            # The orphan row (m1.2, unresolved) still surfaces — alias is NULL.
            orphan = [r for r in joined if r["rule_code"] == "m1.2"]
            assert len(orphan) == 1
            assert orphan[0]["alias"] is None

            # Per-rule outcome rollup the M6 trainer would compute.
            applied_widens = log.query(action_verb="widen", outcome="applied")
            assert len(applied_widens) == 1
            assert applied_widens[0].delta_violations == -1
