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


def test_bare_name_block_real_deck_shapes():
    """Real Calibre decks use ``NAME { @ desc body }`` (no quoted name).

    Representative patterns:
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


def test_colon_suffixed_bare_name_block():
    """``NAME:SUFFIX { ... }`` keeps the full ``NAME:SUFFIX`` as the code.

    Some foundry decks scope a rule with a per-layer or per-severity
    suffix joined by ``:`` (e.g. ``VIA1.R.4:M2``, ``MIM.X:ERROR``). The
    lexer splits the colon out, so without explicit glue every
    ``…:M2`` rule would collapse onto the bare code ``M2``. This test
    pins the fix.
    """
    src = """\
VIA1.R.4:M1 { @ Two-via rule for M1
  EXT M1 < 0.1
}
VIA1.R.4:M2 { @ Two-via rule for M2
  EXT M2 < 0.12
}
OPT.X:ERROR { @ Mutually exclusive options
  PP AND NP
}
"""
    rules = parse_svrf(src)
    codes = [r.code for r in rules]
    assert codes == ["VIA1.R.4:M1", "VIA1.R.4:M2", "OPT.X:ERROR"], codes
    # Each rule keeps its own constraint (no collapse onto a shared code).
    by_code = {r.code: r for r in rules}
    assert by_code["VIA1.R.4:M1"].constraint.branches[0].check.threshold_um == 0.1
    assert by_code["VIA1.R.4:M2"].constraint.branches[0].check.threshold_um == 0.12


def test_multi_colon_bare_name_block():
    """Chains like ``A:B:C { ... }`` survive (rare but real)."""
    src = """\
RR:AR:SP:PO.S.2 { @ Recommended gate space in same OD
  EXT POS2_GATE_W < 0.2
}
"""
    [rule] = parse_svrf(src)
    assert rule.code == "RR:AR:SP:PO.S.2"


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


# ── VARIABLE-resolved thresholds + new check verbs ──────────────────────────

def test_variable_resolved_threshold():
    """Real foundry decks declare numeric thresholds via VARIABLE NAME VALUE
    and reference them in rule bodies (``EXT a b < NAME ABUT < 90``).
    The parser must pre-scan VARIABLE declarations into a symbol table
    so these rules still structure with the right number.
    """
    src = """\
VARIABLE  NW_S_5  0.16
NW.S.5 { @ Space to PW STRAP >= ^NW_S_5 um
  EXT NWi PPOD < NW_S_5 ABUT < 90 SINGULAR REGION
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import SpacingCheck
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert chk.threshold_um == 0.16
    # Comparator is inverted: deck says "violation if < 0.16", rule says ">= 0.16".
    assert chk.op == ">="


def test_define_resolved_threshold():
    """``#DEFINE NAME <num>`` (rare but legal) also resolves."""
    src = """\
#DEFINE PO_W_1 0.09
PO.W.1 { @ Minimum poly width
  WIDTH PO < PO_W_1
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == 0.09


def test_unresolvable_identifier_threshold_falls_back():
    """If the threshold IDENT isn't in the symbol table, the rule still
    parses but ends up no-branch (parser declined the check shape)."""
    src = """\
X.W.1 { @ Width
  WIDTH X < UNDECLARED_VAR
}
"""
    [rule] = parse_svrf(src)
    assert rule.code == "X.W.1"
    assert rule.constraint.branches == []


def test_angle_check():
    """``ANGLE <layer> <cmp> <num> <cmp> <num>`` — angle-bounds check."""
    src = """\
G.3:DNW { @ Shapes must be orthogonal or on a 45 degree angle.
  ANGLE DNW >0 <45
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import ExistenceCheck, LayerRef
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, ExistenceCheck)
    assert isinstance(chk.target, LayerRef)
    assert chk.target.name == "DNW"


def test_offgrid_check():
    """``OFFGRID <layer> <grid>`` — off-grid violation."""
    src = """\
G.1:DNWi { @ grid must be integer multiple
  OFFGRID DNWi 5
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import ExistenceCheck, LayerRef
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, ExistenceCheck)
    assert chk.target.name == "DNWi"


def test_density_check_with_threshold():
    """``DENSITY <layer> <region> <cmp> <num>`` — area-fraction check."""
    src = """\
OD.DN.1 { @ Min OD density
  DENSITY OD CHIP < 0.2
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import DensityCheck
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, DensityCheck)
    # Violation if density < 0.2 → rule wants >= 0.2 → min_ratio = 0.2.
    assert chk.min_ratio == 0.2
    assert chk.max_ratio is None


def test_density_check_upper_bound():
    src = """\
OD.DN.2 { @ Max OD density
  DENSITY OD CHIP > 0.9
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.max_ratio == 0.9
    assert chk.min_ratio is None


def test_enclose_rectangle_check():
    """``ENCLOSE RECTANGLE <layer> <args>`` — shape-enclosure check."""
    src = """\
VARIABLE OD_S_1 0.12
VARIABLE OD_S_3_L 0.2
OD.S.3 { @ Space of two ODs
  OD_SPACE = EXT Wide_OD < 0.2 OPPOSITE REGION
  ENCLOSE RECTANGLE OD_SPACE OD_S_1 OD_S_3_L
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import ExistenceCheck, LayerRef
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, ExistenceCheck)
    assert isinstance(chk.target, LayerRef)
    assert chk.target.name == "OD_SPACE"


def test_copy_check_promotes_layer_to_violation_set():
    """``COPY <layer>`` rule body emits an ExistenceCheck on the layer."""
    src = """\
LPG.OPTION:ERROR { @ Mutually exclusive options
  COPY CHIPx
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import ExistenceCheck, LayerRef
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, ExistenceCheck)
    assert isinstance(chk.target, LayerRef)
    assert chk.target.name == "CHIPx"


def test_recovery_does_not_eat_following_check():
    """Regression: when an assignment leaves dangling tokens on its line
    (e.g. unmodelled SVRF modifiers like GOOD), the parser must still
    pick up the check on the *next* line instead of bailing out of the
    whole rule body.
    """
    src = """\
VARIABLE CO_EN_4 0.020
CO.EN.3__CO.EN.4 { @ Enclosure
  X = RECTANGLE ENCLOSURE CO POLYs ABUT >0 < 90 SINGULAR GOOD CO_EN_4 OPPOSITE
  ENC X POLYs < CO_EN_4 ABUT < 90 SINGULAR REGION
}
"""
    [rule] = parse_svrf(src)
    from lithos_core.ir import EnclosureCheck
    assert rule.code == "CO.EN.3__CO.EN.4"
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, EnclosureCheck)
    assert chk.threshold_um == 0.020


# ── Numeric-expression thresholds (Phase 2) ─────────────────────────────────

def test_numeric_expression_threshold_multiplication():
    """Threshold can be a math expression: `var * var`."""
    src = """\
VARIABLE A 0.05
VARIABLE B 0.10
TEST.W { @ test
  EXT M1 < A * B
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == pytest.approx(0.005, rel=1e-9)


def test_numeric_expression_threshold_addition():
    src = """\
VARIABLE A 0.05
TEST.W { @ test
  EXT M1 < A + 0.002
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == pytest.approx(0.052)


def test_numeric_expression_with_parens():
    """The manual's own example (page 72): `3 * (GRID + 6)`."""
    src = """\
VARIABLE GRID 0.005
TEST.W { @ test
  EXT M1 < 3 * (GRID + 6)
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == pytest.approx(18.015)


def test_math_function_in_threshold():
    """Math functions from Table 2-4 — MAX/MIN with two args."""
    src = """\
VARIABLE A 0.05
VARIABLE B 0.10
TEST.W { @ test
  EXT M1 < MAX(A, B)
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == pytest.approx(0.10)


def test_undefined_identifier_in_expression_is_recoverable():
    """If a VARIABLE reference can't resolve, the rule falls back to
    no-branch (not a hard crash)."""
    src = """\
TEST.W { @ test
  EXT M1 < UNDEFINED_VAR
}
"""
    [rule] = parse_svrf(src)
    assert rule.code == "TEST.W"
    assert rule.constraint.branches == []


# ── Digit-prefixed layer names (manual page 67) ──────────────────────────────

def test_digit_prefixed_layer_name():
    """Real foundry decks use layer names like `25_18V_GATE_W`."""
    src = """\
VARIABLE WIDTH_LIMIT 0.18
RULE.X { @ test
  INT 25_18V_GATE_W < WIDTH_LIMIT
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    # Single-layer INT → WidthCheck on the digit-prefixed layer.
    from lithos_core.ir import WidthCheck, LayerRef
    assert isinstance(chk, WidthCheck)
    assert isinstance(chk.target, LayerRef)
    assert chk.target.name == "25_18V_GATE_W"


def test_digit_prefixed_rule_code():
    """Real foundry decks use rule codes like `3DMIM.S.1`."""
    src = """\
VARIABLE 3DMIM_S_1 0.4
3DMIM.S.1 { @ test
  EXT CMM CTM < 3DMIM_S_1
}
"""
    [rule] = parse_svrf(src)
    assert rule.code == "3DMIM.S.1"
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == pytest.approx(0.4)


# ── Last-assigned-layer fallback ─────────────────────────────────────────────

def test_last_assigned_layer_promoted_to_existence_check():
    """When a rule body is a chain of layer assignments with no explicit
    check, the parser promotes the *last* assigned layer to an
    :class:`ExistenceCheck`. Real foundry decks use this idiom for
    via-stack analysis (``Branch1`` / ``GoodBranch`` / ``BAD_REGION``).
    """
    src = """\
VIA1.X { @ assignment-chain rule
  Branch1 = M1 AND VIA1
  GoodBranch = Branch1 AND M2
  BAD_REGION = Branch1 NOT GoodBranch
}
"""
    [rule] = parse_svrf(src)
    chk = rule.constraint.branches[0].check
    from lithos_core.ir import ExistenceCheck, LayerRef
    assert isinstance(chk, ExistenceCheck)
    assert isinstance(chk.target, LayerRef)
    assert chk.target.name == "BAD_REGION"
    assert set(rule.constraint.derived_layers) == {"Branch1", "GoodBranch", "BAD_REGION"}
