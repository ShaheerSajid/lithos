"""End-to-end ingestion: SVRF deck + mock LLM extraction + chunks → RuleDB.

Exercises the full pipeline minus a real LLM (the extractor is mocked).
Verifies the joined rows in the DB carry deck-derived constraints,
LLM-derived fix metadata, category assignment, source text, and the
needs_review flag for cross-validation failures.
"""
from __future__ import annotations

from pathlib import Path

from lithos_core import (
    CategoryConfig,
    CategoryDef,
    FixMetadata,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest import svrf_to_db
from lithos_ingest.chunker import Chunk


_DECK = """\
RULECHECK "PO.W.1: poly minimum width" {
    WIDTH poly < 0.15
}
RULECHECK "M2.S.1: metal2 spacing default" {
    EXTERNAL met2 < 0.14
}
RULECHECK "LI.E.1: licon enclosure" {
    ENCLOSURE licon1 BY li1 < 0.04
}
"""


def _categories() -> CategoryConfig:
    return CategoryConfig(categories=[
        CategoryDef(name="poly",       code_prefixes=["PO."],                priority=10),
        CategoryDef(name="metal_low",  code_prefixes=["M1.", "M2.", "LI."],  priority=20),
    ])


def _mock_chunks() -> dict[str, list[Chunk]]:
    return {
        "PO.W.1": [Chunk(code="PO.W.1", text="PO.W.1: poly minimum width is 0.15um.",
                         page=10, span=(0, 40), anchor=0, section="poly")],
        "M2.S.1": [Chunk(code="M2.S.1", text="M2.S.1: metal2 spacing 0.14um for litho.",
                         page=22, span=(0, 40), anchor=0, section="metal_low")],
    }


def _mock_fix_metadata() -> dict[str, FixMetadata]:
    return {
        "PO.W.1": FixMetadata(
            intent="ensures the gate channel is uniformly formed at minimum geometry",
            allowed_action_classes=["widen"],
            affected_layers=["poly"],
            notes="critical primitive: never violated for unstacked transistors.",
        ),
        "M2.S.1": FixMetadata(
            intent="prevents litho bridging between adjacent metal2 lines",
            allowed_action_classes=["widen", "shift_orthogonal"],
            affected_layers=["met2"],
        ),
        # LI.E.1 deliberately omitted — joiner should treat it as deck-only.
    }


def test_full_ingestion_writes_joined_rows(tmp_path: Path):
    deck_path = tmp_path / "deck.drc"
    db_path   = tmp_path / "rules.db"
    deck_path.write_text(_DECK)

    n = svrf_to_db(
        deck_path, db_path,
        pdk_name="test_pdk", pdk_version="0.0.1",
        categories=_categories(),
        fix_metadata=_mock_fix_metadata(),
        chunks=_mock_chunks(),
    )
    assert n == 3

    with RuleDB(db_path) as db:
        # Deck-only rule (LI.E.1) — no FixMetadata, no PDF chunk, not flagged.
        li = db.get_rule("LI.E.1")
        assert li is not None
        assert li.fix_metadata is None
        assert li.needs_review is False
        assert li.category == "metal_low"
        assert li.provenance == {"constraint": "deck"}

        # Joined rule (PO.W.1) — has FixMetadata from the mock extractor.
        po = db.get_rule("PO.W.1")
        assert po is not None
        assert po.fix_metadata is not None
        assert "uniformly formed" in po.fix_metadata.intent
        assert po.fix_metadata.allowed_action_classes == ["widen"]
        assert po.category == "poly"
        assert po.provenance == {"constraint": "deck", "fix_metadata": "llm"}
        assert po.confidence["fix_metadata"] == 0.85
        assert po.needs_review is False

        # PDF chunk text was recorded in rule_source.
        src = db.get_source("PO.W.1")
        assert src is not None
        assert src["pdf_chunk"] == "PO.W.1: poly minimum width is 0.15um."
        assert src["pdf_page"] == 10
        # Deck text is also there.
        assert "WIDTH poly" in src["deck_block"]

        # M2.S.1 — same shape.
        m2 = db.get_rule("M2.S.1")
        assert isinstance(m2.constraint.branches[0].check, SpacingCheck)
        assert m2.fix_metadata.intent.startswith("prevents litho bridging")

        # Coverage by category.
        assert dict(db.categories()) == {"poly": 1, "metal_low": 2}


def test_full_ingestion_flags_mismatch(tmp_path: Path):
    deck_path = tmp_path / "deck.drc"
    db_path   = tmp_path / "rules.db"
    deck_path.write_text(_DECK)

    bad_fix = {
        "M2.S.1": FixMetadata(
            intent="...",
            allowed_action_classes=["widen"],
            affected_layers=["poly"],      # wrong layer for an M2 rule
        ),
    }

    svrf_to_db(
        deck_path, db_path,
        pdk_name="t", pdk_version="0",
        fix_metadata=bad_fix,
    )

    with RuleDB(db_path) as db:
        m2 = db.get_rule("M2.S.1")
        assert m2.needs_review is True
        assert "review_mismatches" in m2.provenance
        assert any("layer mismatch" in m for m in m2.provenance["review_mismatches"])
        # Confidence halved because of the mismatch.
        assert m2.confidence["fix_metadata"] == 0.425
