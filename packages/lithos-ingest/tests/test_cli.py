"""CLI tests — invoke main() directly with synthetic inputs.

We bypass the installed entry point and call the parser/dispatcher in
process. Output goes to stderr / stdout via the normal print path; we
capture those with capsys.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_core import RuleDB

from lithos_ingest.cli import main


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

_CATEGORIES_YAML = """\
default_category: unknown
categories:
  - name: poly
    code_prefixes: ["PO."]
    priority: 10
  - name: metal_low
    code_prefixes: ["M1.", "M2.", "LI."]
    priority: 20
"""

_DOC_TEXT = """\
Chapter 5. Metal Layers
========================

PO.W.1 Minimum poly width
--------------------------

Minimum drawn width of polysilicon shall be 0.15 micrometres.

M2.S.1 Minimum metal2 spacing
------------------------------

Minimum drawn spacing between adjacent met2 polygons shall be 0.14 micrometres.

LI.E.1 Licon to li1 enclosure
------------------------------

Minimum enclosure of licon1 by li1 shall be 0.04 micrometres on all sides.
"""


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    deck = tmp_path / "deck.drc"
    cats = tmp_path / "categories.yaml"
    doc  = tmp_path / "rules.rst"
    deck.write_text(_DECK)
    cats.write_text(_CATEGORIES_YAML)
    doc.write_text(_DOC_TEXT)
    return deck, cats, doc


# ── svrf subcommand ────────────────────────────────────────────────────────

def test_svrf_writes_deck_only_db(tmp_path: Path, capsys):
    deck, cats, _ = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    code = main([
        "svrf", str(deck),
        "--db", str(db),
        "--pdk-name", "test_pdk",
        "--pdk-version", "0.0.1",
        "--categories", str(cats),
    ])
    assert code == 0
    assert "Wrote 3 rules" in capsys.readouterr().err

    with RuleDB(db) as conn:
        assert conn.pdk_identity() == ("test_pdk", "0.0.1")
        assert conn.count_rules() == 3
        # Category resolved from the YAML.
        assert conn.count_rules(category="poly") == 1
        assert conn.count_rules(category="metal_low") == 2
        # Deck-only, no LLM yet.
        r = conn.get_rule("M2.S.1")
        assert r.fix_metadata is None


def test_svrf_without_categories_defaults_to_unknown(tmp_path: Path, capsys):
    deck, _, _ = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    code = main([
        "svrf", str(deck),
        "--db", str(db),
        "--pdk-name", "t", "--pdk-version", "0",
    ])
    assert code == 0
    with RuleDB(db) as conn:
        assert conn.count_rules(category="unknown") == 3


# ── full subcommand ────────────────────────────────────────────────────────

def test_full_with_rst_doc_and_no_llm(tmp_path: Path, capsys):
    deck, cats, doc = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    code = main([
        "full",
        "--svrf",        str(deck),
        "--doc",         str(doc),
        "--db",          str(db),
        "--pdk-name",    "test_pdk",
        "--pdk-version", "0.0.1",
        "--categories",  str(cats),
        "--no-llm",
    ])
    assert code == 0
    out = capsys.readouterr().err
    assert "Wrote 3 rules" in out
    assert "llm_extracted=no" in out

    with RuleDB(db) as conn:
        # Each rule's source row now carries a pdf_chunk extracted from the RST.
        po = conn.get_source("PO.W.1")
        assert po is not None
        assert "polysilicon" in po["pdf_chunk"].lower()
        assert po["pdf_chunk"] is not None
        # No LLM ⇒ no fix_metadata on the rules.
        r = conn.get_rule("PO.W.1")
        assert r.fix_metadata is None


def test_full_warns_when_no_model_and_no_no_llm(tmp_path: Path, capsys):
    """Forgetting both --model and --no-llm produces a warning, not an error."""
    deck, _, doc = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    code = main([
        "full",
        "--svrf",        str(deck),
        "--doc",         str(doc),
        "--db",          str(db),
        "--pdk-name",    "t",
        "--pdk-version", "0",
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "--model not provided" in err


def test_full_with_csv_doc(tmp_path: Path):
    deck, _, _ = _write_fixtures(tmp_path)
    csv = tmp_path / "rules.csv"
    csv.write_text(
        "rule_code,layer,value_um,description\n"
        "PO.W.1,poly,0.15,poly min width\n"
        "M2.S.1,met2,0.14,metal2 spacing\n"
    )
    db = tmp_path / "rules.db"
    code = main([
        "full",
        "--svrf",            str(deck),
        "--doc",             str(csv),
        "--db",              str(db),
        "--pdk-name",        "t", "--pdk-version", "0",
        "--no-llm",
        "--csv-code-column", "rule_code",
    ])
    assert code == 0
    with RuleDB(db) as conn:
        src = conn.get_source("PO.W.1")
        assert "poly min width" in src["pdf_chunk"]
        # LI.E.1 has no CSV row → no source.
        li_src = conn.get_source("LI.E.1")
        # LI.E.1 still has a deck_block from the SVRF, so source isn't None,
        # but pdf_chunk should be missing.
        assert li_src is None or li_src["pdf_chunk"] is None


def test_full_csv_requires_code_column(tmp_path: Path):
    deck, _, _ = _write_fixtures(tmp_path)
    csv = tmp_path / "rules.csv"
    csv.write_text("rule_code,layer\nPO.W.1,poly\n")
    db = tmp_path / "rules.db"
    with pytest.raises(ValueError, match="--csv-code-column"):
        main([
            "full",
            "--svrf", str(deck),
            "--doc",  str(csv),
            "--db",   str(db),
            "--pdk-name", "t", "--pdk-version", "0",
            "--no-llm",
        ])


def test_full_unknown_doc_extension_errors(tmp_path: Path):
    deck, _, _ = _write_fixtures(tmp_path)
    weird = tmp_path / "rules.xyz"
    weird.write_text("doesn't matter")
    db = tmp_path / "rules.db"
    with pytest.raises(ValueError, match="Cannot detect loader"):
        main([
            "full",
            "--svrf", str(deck),
            "--doc",  str(weird),
            "--db",   str(db),
            "--pdk-name", "t", "--pdk-version", "0",
            "--no-llm",
        ])


# ── stats subcommand ───────────────────────────────────────────────────────

def test_stats_reports_pdk_and_categories(tmp_path: Path, capsys):
    deck, cats, _ = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    main(["svrf", str(deck), "--db", str(db),
          "--pdk-name", "test_pdk", "--pdk-version", "0.0.1",
          "--categories", str(cats)])
    capsys.readouterr()                            # drain svrf's output

    code = main(["stats", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "PDK:" in out and "test_pdk" in out
    assert "Rules:" in out
    assert "metal_low" in out
    assert "poly" in out


# ── review subcommand ─────────────────────────────────────────────────────

def test_review_shows_flagged_rules(tmp_path: Path, capsys):
    """Manually inject a flagged rule, then verify the review subcommand finds it."""
    from lithos_core import (
        Constraint, ConstraintBranch, FixMetadata, LayerRef, Rule,
        WidthCheck,
    )

    db = tmp_path / "rules.db"
    with RuleDB(db) as conn:
        conn.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
        conn.upsert_rule(Rule(
            code="X.1",
            category="unknown",
            usage_class="geometry_primitive",
            short_desc="some test rule",
            constraint=Constraint(branches=[ConstraintBranch(check=WidthCheck(
                target=LayerRef(name="met2"), op=">=", threshold_um=0.1,
            ))]),
            fix_metadata=FixMetadata(intent="x", affected_layers=["poly"]),
            provenance={"constraint": "deck", "review_mismatches": [
                "layer mismatch: deck constraint references ['met2'] but FixMetadata.affected_layers is ['poly']",
            ]},
            needs_review=True,
        ))

    code = main(["review", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "X.1" in out
    assert "layer mismatch" in out


def test_review_clean_db_message(tmp_path: Path, capsys):
    deck, _, _ = _write_fixtures(tmp_path)
    db = tmp_path / "rules.db"
    main(["svrf", str(deck), "--db", str(db),
          "--pdk-name", "t", "--pdk-version", "0"])
    capsys.readouterr()
    code = main(["review", str(db)])
    assert code == 0
    assert "No rules flagged" in capsys.readouterr().err


# ── Help & misuse ─────────────────────────────────────────────────────────

def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "lithos-ingest" in out
    assert "svrf" in out
    assert "full" in out


def test_missing_subcommand_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
