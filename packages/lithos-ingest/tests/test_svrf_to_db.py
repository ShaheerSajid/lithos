"""End-to-end test: SVRF file on disk → RuleDB → query back."""
from __future__ import annotations

from pathlib import Path

from lithos_core import (
    CategoryConfig,
    CategoryDef,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest import svrf_to_db


DECK = """\
// Sky130-flavoured snippet — small but representative.
RULECHECK "PO.W.1: poly minimum width" {
    WIDTH poly < 0.15
}
RULECHECK "M2.W.1: metal2 minimum width" {
    WIDTH met2 < 0.14
}
RULECHECK "M2.S.1: metal2 spacing default" {
    EXTERNAL met2 < 0.14
}
RULECHECK "LI.E.1: licon enclosure" {
    ENCLOSURE licon1 BY li1 < 0.04
}
"""


def _categories() -> CategoryConfig:
    return CategoryConfig(
        categories=[
            CategoryDef(name="poly",       code_prefixes=["PO.", "P.", "poly."], priority=10),
            CategoryDef(name="metal_low",  code_prefixes=["M1.", "M2.", "li.", "LI."], priority=20),
            CategoryDef(name="antenna",    code_prefixes=["A.", "ANT."],
                        enabled=False, priority=90),
        ],
    )


def test_svrf_to_db_full_path(tmp_path: Path):
    svrf = tmp_path / "deck.drc"
    db   = tmp_path / "rules.db"
    svrf.write_text(DECK)

    n = svrf_to_db(
        svrf, db,
        pdk_name="test_pdk", pdk_version="0.0.1",
        categories=_categories(),
    )
    assert n == 4

    with RuleDB(db) as conn:
        assert conn.pdk_identity() == ("test_pdk", "0.0.1")

        # Category assignment from the user-configured matcher.
        assert conn.count_rules(category="poly") == 1
        assert conn.count_rules(category="metal_low") == 3
        coverage = dict(conn.categories())
        assert coverage == {"poly": 1, "metal_low": 3}

        # Every rule is geometry_primitive — width/spacing/enclosure only.
        assert conn.count_rules(usage_class="geometry_primitive") == 4

        # Spot-check the metal2 spacing rule round-trips through SQLite.
        r = conn.get_rule("M2.S.1")
        assert r is not None
        assert r.category    == "metal_low"
        assert r.usage_class == "geometry_primitive"
        chk = r.constraint.branches[0].check
        assert isinstance(chk, SpacingCheck)
        assert chk.threshold_um == 0.14
        assert chk.op == ">="

        # Aliases: both the foundry code and the full title resolve back.
        assert conn.resolve_alias("M2.S.1") == "M2.S.1"
        assert conn.resolve_alias("M2.S.1: metal2 spacing default") == "M2.S.1"
        assert conn.resolve_alias("totally unknown") is None

        # Raw deck source is preserved verbatim for human review.
        src = conn.get_source("M2.S.1")
        assert src is not None
        assert "EXTERNAL met2" in src["deck_block"]
        assert src["deck_title"] == "M2.S.1: metal2 spacing default"


def test_svrf_to_db_without_category_config(tmp_path: Path):
    svrf = tmp_path / "deck.drc"
    db   = tmp_path / "rules.db"
    svrf.write_text(DECK)

    n = svrf_to_db(svrf, db, pdk_name="t", pdk_version="0")
    assert n == 4

    with RuleDB(db) as conn:
        # Every rule lands in "unknown" without a category config.
        assert conn.count_rules(category="unknown") == 4
