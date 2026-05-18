"""Tests for the SVRF parser.

Focuses on the patterns that actually appear in foundry decks:
WIDTH, EXTERNAL spacing (single and dual-layer), ENCLOSURE, SIZE BY,
boolean layer derivations, and title-based code extraction.
"""
from __future__ import annotations

import pytest

from lithos_core.ir import (
    EnclosureCheck,
    LayerBool,
    LayerRef,
    LayerSize,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.parsers.svrf import SVRFParseError, parse_svrf


SAMPLE_DECK = """\
// Sky130-flavoured SVRF snippet for parser tests.

RULECHECK "M2.W.1: metal2 minimum width" {
    WIDTH met2 < 0.14
}

# alternative comment style
RULECHECK "M2.S.1: metal2 spacing default" {
    EXTERNAL met2 < 0.14
}

RULECHECK "M2.S.1.W: metal2 wide spacing" {
    wide_met2 = SIZE met2 BY 0.05
    EXTERNAL wide_met2 < 0.30
}

RULECHECK "LI.E.1: licon to li1 enclosure" {
    ENCLOSURE licon1 BY li1 < 0.04
}

RULECHECK "PO.W.1: poly clean of npc" {
    poly_clean = poly NOT npc
    WIDTH poly_clean < 0.15
}

RULECHECK "M2.M3.S.1: met2 to met3 spacing" {
    EXTERNAL met2 met3 < 0.12
}
"""


def test_parse_returns_six_rules():
    rules = parse_svrf(SAMPLE_DECK)
    assert len(rules) == 6
    codes = [r.code for r in rules]
    assert codes == [
        "M2.W.1", "M2.S.1", "M2.S.1.W", "LI.E.1", "PO.W.1", "M2.M3.S.1",
    ]


def test_width_check():
    rules = parse_svrf(SAMPLE_DECK)
    r = rules[0]
    assert r.code == "M2.W.1"
    assert r.title == "M2.W.1: metal2 minimum width"
    assert r.constraint is not None
    assert len(r.constraint.branches) == 1
    chk = r.constraint.branches[0].check
    assert isinstance(chk, WidthCheck)
    assert isinstance(chk.target, LayerRef) and chk.target.name == "met2"
    # SVRF measurement "< 0.14" is a violation comparator; the IR stores
    # the *rule* comparator: minimum width >= 0.14.
    assert chk.op == ">="
    assert chk.threshold_um == 0.14


def test_spacing_default_single_layer():
    r = parse_svrf(SAMPLE_DECK)[1]
    chk = r.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert isinstance(chk.layer_a, LayerRef) and chk.layer_a.name == "met2"
    assert chk.layer_b is None              # internal (same-layer) spacing
    assert chk.op == ">="
    assert chk.threshold_um == 0.14


def test_spacing_dual_layer():
    r = parse_svrf(SAMPLE_DECK)[5]
    chk = r.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert chk.layer_a == LayerRef(name="met2")
    assert chk.layer_b == LayerRef(name="met3")
    assert chk.op == ">="
    assert chk.threshold_um == 0.12


def test_size_by_derived_layer_in_rulecheck():
    r = parse_svrf(SAMPLE_DECK)[2]
    assert r.code == "M2.S.1.W"
    # Inside-RULECHECK assignment lives in derived_layers.
    assert "wide_met2" in r.constraint.derived_layers
    derived = r.constraint.derived_layers["wide_met2"]
    assert isinstance(derived, LayerSize)
    assert derived.operand == LayerRef(name="met2")
    assert derived.by_um == 0.05
    # The check still references the underlying layer name by string (LayerRef);
    # joining derived names is the joiner's job, not the parser's.
    chk = r.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert chk.layer_a == LayerRef(name="wide_met2")
    assert chk.threshold_um == 0.30


def test_enclosure_check():
    r = parse_svrf(SAMPLE_DECK)[3]
    chk = r.constraint.branches[0].check
    assert isinstance(chk, EnclosureCheck)
    assert chk.inner == LayerRef(name="licon1")
    assert chk.outer == LayerRef(name="li1")
    assert chk.op == ">="
    assert chk.threshold_um == 0.04


def test_not_layer_in_derived():
    r = parse_svrf(SAMPLE_DECK)[4]
    derived = r.constraint.derived_layers["poly_clean"]
    # `poly NOT npc` is SVRF set-subtraction → normalised to a AND (NOT b).
    assert isinstance(derived, LayerBool)
    assert derived.op == "and"
    assert derived.operands[0] == LayerRef(name="poly")
    inner = derived.operands[1]
    assert isinstance(inner, LayerBool)
    assert inner.op == "not"
    assert inner.operands == [LayerRef(name="npc")]


def test_aliases_contain_code_and_title():
    rules = parse_svrf(SAMPLE_DECK)
    r = rules[0]
    alias_strings = {a for a, _src in r.aliases}
    assert "M2.W.1" in alias_strings
    assert "M2.W.1: metal2 minimum width" in alias_strings
    sources = {src for _a, src in r.aliases}
    assert "foundry_code" in sources
    assert "deck_rulecheck" in sources


def test_comments_stripped():
    src = """\
// header comment
RULECHECK "X.1: trivial" {
    // an inline comment
    WIDTH a < 0.1   # trailing comment
}
"""
    rules = parse_svrf(src)
    assert len(rules) == 1
    assert rules[0].code == "X.1"
    assert rules[0].constraint.branches[0].check.threshold_um == 0.1


def test_empty_input():
    assert parse_svrf("") == []
    assert parse_svrf("// only comments\n# nothing else\n") == []


def test_tsmc_style_bare_name_block():
    """Real Calibre decks use ``NAME { @ desc body }`` (no quoted name).

    Sample patterns drawn from the TSMC180 deck:
      * ``INT <layer> < t``      → WidthCheck (single-layer INT = width)
      * ``EXT <a> <b> < t``      → SpacingCheck cross-layer
      * ``ENC <inner> <outer> < t`` → EnclosureCheck
      * ``A AND B``              → ExistenceCheck "this set must be empty"
      * trailing modifiers (``ABUT < 90 SINGULAR REGION``) are absorbed
    """
    src = """\
NW.W.1 { @ Min. NWEL width < 0.86
  INT NWEL < 0.86 ABUT < 90 SINGULAR REGION
}
OD.S.1 { @ Min. OD space < 0.28
  EXT OD < 0.28 ABUT < 90 SINGULAR REGION
}
NWR.E.1 { @ Min. OD enclose NWEL resistor < 1.0
  ENC ODWR NWEL < 1.0 ABUT < 90 SINGULAR REGION
}
PP.R.1_NP.R.1 { @ PP and NP not allowed to overlap
  PP AND NP
}
"""
    rules = parse_svrf(src)
    by_code = {r.code: r for r in rules}

    from lithos_core.ir import ExistenceCheck
    assert by_code["NW.W.1"].constraint.branches[0].check.threshold_um == 0.86
    assert by_code["NW.W.1"].constraint.branches[0].check.op == ">="
    assert by_code["OD.S.1"].constraint.branches[0].check.threshold_um == 0.28
    assert by_code["NWR.E.1"].constraint.branches[0].check.threshold_um == 1.0
    pp_chk = by_code["PP.R.1_NP.R.1"].constraint.branches[0].check
    assert isinstance(pp_chk, ExistenceCheck)
    assert pp_chk.must_be_empty is True


def test_existence_check_for_bare_layer_body():
    """A rule whose only body statement is a layer expression with no
    comparator gets an ExistenceCheck (the LLM-on-demand pathway picks up
    cases this can't structure)."""
    src = """\
RES.9 { @ DMN2V overlap DMP2V not allowed
  DMP2V AND DMN2V
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    from lithos_core.ir import ExistenceCheck, LayerBool
    assert isinstance(chk, ExistenceCheck)
    assert isinstance(chk.target, LayerBool)
    assert chk.target.op == "and"


def test_postfix_area_check():
    """``OD AREA < 0.202`` — Calibre's layer-first form."""
    src = """\
OD.A.1 { @ Min. OD area < 0.202
  OD AREA < 0.202
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import AreaCheck
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, AreaCheck)
    assert chk.threshold_um2 == 0.202
    assert chk.op == ">="


def test_preprocessor_directives_are_skipped():
    """`#IFDEF` blocks don't disrupt rule parsing."""
    src = """\
#IFDEF FOO
#DEFINE BAR
RULE.X { @ A rule
  INT POLY < 0.15
}
#ENDIF
"""
    rules = parse_svrf(src)
    assert [r.code for r in rules] == ["RULE.X"]


def test_parser_is_tolerant_of_unknown_constructs():
    """Garbage inside a rule body is skipped to the closing brace; the
    rule is still emitted with at least code + title. Tolerance is
    deliberate so real foundry decks (which mix in many constructs the
    parser doesn't model) yield a useful rule catalogue."""
    src = """\
RULECHECK "X.1: bad body" {
   ??? some garbage we don't model
   COIN NONSENSE
}
"""
    rules = parse_svrf(src)
    assert len(rules) == 1
    assert rules[0].code == "X.1"
    # Constraint exists but has no branches because no check parsed.
    assert rules[0].constraint.branches == []


def test_deck_block_text_preserved():
    rules = parse_svrf(SAMPLE_DECK)
    block = rules[0].deck_block
    assert "RULECHECK" in block
    assert "metal2 minimum width" in block
    assert "WIDTH met2" in block
