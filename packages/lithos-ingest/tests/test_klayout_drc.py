"""KLayout-DRC parser tests.

Covers the declarative subset: assignments, layer algebra (and/or/not/xor),
selection (inside/outside/interact), sizing, and the geometric checks
(width / space / separation / enclosing / enclosed_by / with_area). Each
parsed deck is a tiny KLayout DRC snippet — we don't drive klayout.
"""
from __future__ import annotations

import pytest

from lithos_core.ir import (
    AreaCheck,
    EnclosureCheck,
    LayerBool,
    LayerRef,
    LayerSelect,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.parsers.klayout_drc import (
    KLayoutDRCParseError,
    parse_klayout_drc,
)


def _only(parsed):
    assert len(parsed) == 1, f"expected 1 rule, got {len(parsed)}"
    return parsed[0]


# ── Width / space ──────────────────────────────────────────────────────────

def test_width_rule_basic():
    src = """\
met2 = input(69, 20)
met2.width(0.14).output("M2.W.1", "metal2 minimum width")
"""
    rule = _only(parse_klayout_drc(src))
    assert rule.code == "M2.W.1"
    assert rule.title == "metal2 minimum width"
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, WidthCheck)
    assert chk.target == LayerRef(name="met2")
    assert chk.op == ">="
    assert chk.threshold_um == 0.14
    assert rule.constraint.deck_dialect == "klayout"


def test_same_layer_spacing_with_modifier():
    src = """\
met2 = input(69, 20)
met2.space(0.14, projection).output("M2.S.1", "metal2 spacing")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert chk.layer_a == LayerRef(name="met2")
    assert chk.layer_b is None
    assert chk.threshold_um == 0.14
    assert "projection" in chk.modifiers


def test_cross_layer_separation():
    src = """\
met2 = input(69, 20)
met3 = input(70, 20)
met2.separation(met3, 0.12, projection).output("M2.M3.S.1", "met2 to met3 spacing")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, SpacingCheck)
    assert chk.layer_a == LayerRef(name="met2")
    assert chk.layer_b == LayerRef(name="met3")
    assert chk.threshold_um == 0.12
    assert "projection" in chk.modifiers


# ── Enclosure ──────────────────────────────────────────────────────────────

def test_enclosing_check_outer_is_receiver():
    """`outer.enclosing(inner, t)` means outer must enclose inner by t."""
    src = """\
licon = input(66, 44)
li1 = input(67, 20)
li1.enclosing(licon, 0.04).output("LI.E.1", "li1 must enclose licon")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, EnclosureCheck)
    assert chk.outer == LayerRef(name="li1")
    assert chk.inner == LayerRef(name="licon")
    assert chk.threshold_um == 0.04


def test_enclosed_by_inverts_inner_outer():
    """`inner.enclosed_by(outer, t)` should produce the same shape with
    inner == receiver."""
    src = """\
licon = input(66, 44)
li1 = input(67, 20)
licon.enclosed_by(li1, 0.04).output("LI.E.1", "licon enclosed by li1")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, EnclosureCheck)
    assert chk.inner == LayerRef(name="licon")
    assert chk.outer == LayerRef(name="li1")


# ── Area ───────────────────────────────────────────────────────────────────

def test_with_area_min():
    src = """\
li1 = input(67, 20)
li1.with_area(0.0561).output("LI.A.1", "li1 minimum area")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, AreaCheck)
    assert chk.threshold_um2 == 0.0561


# ── Layer algebra ──────────────────────────────────────────────────────────

def test_and_combination_in_check():
    src = """\
poly = input(66, 20)
diff = input(65, 20)
gate = poly.and(diff)
gate.width(0.15).output("PO.W.1", "gate width")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert isinstance(chk, WidthCheck)
    # target is the LayerRef "gate" (the bound name)
    assert chk.target == LayerRef(name="gate")


def test_not_is_set_subtraction():
    """`a.not(b)` is set subtraction → a AND (NOT b). Verified by ensuring
    the resulting bound layer parses through to a check without issue."""
    src = """\
poly = input(66, 20)
npc  = input(95, 20)
poly_clean = poly.not(npc)
poly_clean.width(0.15).output("PO.W.1", "clean poly width")
"""
    rules = parse_klayout_drc(src)
    assert len(rules) == 1
    # The bound `poly_clean` is the result of a Bool(and, [poly, Bool(not, [npc])]),
    # but the rule's target is the LayerRef bound to that name.


def test_inside_select():
    src = """\
poly = input(66, 20)
diff = input(65, 20)
gate_poly = poly.inside(diff)
gate_poly.width(0.15).output("PO.W.2", "gate poly inside diff width")
"""
    rules = parse_klayout_drc(src)
    assert len(rules) == 1


# ── Comments + multiple rules ──────────────────────────────────────────────

def test_comments_and_multiple_rules():
    src = """\
# A small KLayout DRC deck for tests.
poly = input(66, 20)   # gate poly
met2 = input(69, 20)
poly.width(0.15).output("PO.W.1", "poly minimum width")
# Spacing rule next.
met2.space(0.14).output("M2.S.1", "metal2 spacing")
"""
    rules = parse_klayout_drc(src)
    assert [r.code for r in rules] == ["PO.W.1", "M2.S.1"]


# ── .output behaviour ─────────────────────────────────────────────────────

def test_single_arg_output_uses_code_as_title():
    src = """\
met2 = input(69, 20)
met2.width(0.14).output("M2.W.1")
"""
    rule = _only(parse_klayout_drc(src))
    assert rule.code == "M2.W.1"
    assert rule.title == "M2.W.1"
    # No second alias because title == code.
    sources = {src_ for _, src_ in rule.aliases}
    assert "deck_rulecheck" not in sources


def test_output_aliases_include_both_when_distinct():
    src = """\
met2 = input(69, 20)
met2.width(0.14).output("M2.W.1", "metal2 min width")
"""
    rule = _only(parse_klayout_drc(src))
    sources = {src_ for _, src_ in rule.aliases}
    assert "foundry_code" in sources
    assert "deck_rulecheck" in sources


# ── No-op chains ───────────────────────────────────────────────────────────

def test_um_suffix_is_skipped_on_numbers():
    """`0.14.um` is a Ruby float-method call; we treat it as a no-op µm cast."""
    src = """\
met2 = input(69, 20)
met2.width(0.14.um).output("M2.W.1", "metal2 width")
"""
    rule = _only(parse_klayout_drc(src))
    chk = rule.constraint.branches[0].check
    assert chk.threshold_um == 0.14


def test_unknown_method_is_passthrough():
    """Tolerated: deck calls .count, .info, etc.; we ignore them."""
    src = """\
met2 = input(69, 20)
met2.width(0.14).output("M2.W.1", "metal2 width")
met2.width(0.14).count
"""
    rules = parse_klayout_drc(src)
    assert len(rules) == 1
    assert rules[0].code == "M2.W.1"


# ── Error path ─────────────────────────────────────────────────────────────

def test_parse_error_reports_position():
    src = """\
met2 = input(69, 20)
met2.width(0.14).output("M.1"  ???
"""
    with pytest.raises(KLayoutDRCParseError) as exc:
        parse_klayout_drc(src)
    assert "line 2" in str(exc.value)


def test_empty_input():
    assert parse_klayout_drc("") == []
    assert parse_klayout_drc("# just comments\n# more comments\n") == []
