"""Tests for ``lithos_layout.stack`` — canonical metal-stack + via lookup."""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_core import (
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    LayerRef,
    PDKMetadata,
    Rule,
    RuleDB,
    WidthCheck,
)
from lithos_layout       import BootstrapMapping, BootstrapRules
from lithos_layout.stack import ViaTransition, _stack_index, via_stack_between


# ── Shared fixture ──────────────────────────────────────────────────────────

def _rules(
    tmp_path: Path,
    *,
    m0_collapsed: bool = False,
    contact_in_diff:    float = 0.04,
    contact_size:       float = 0.17,
    m0_enc_contact_2adj: float = 0.06,
    m0_enc_contact_all:  float = 0.04,
    via01_size:         float = 0.17,
    m0_enc_via01_2adj:  float = 0.085,
    m0_enc_via01_all:   float = 0.06,
    m1_enc_via01_2adj:  float = 0.085,
    m1_enc_via01_all:   float = 0.06,
    via12_size:         float = 0.20,
    m1_enc_via12_2adj:  float = 0.085,
    m2_enc_via12_2adj:  float = 0.085,
) -> BootstrapRules:
    """Build a minimal BootstrapRules with the keys ``via_stack_between`` reads."""
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-20T00:00:00Z")

    rules_to_seed = [
        ("CO.W.1",            WidthCheck(target=LayerRef(name="contact"),
                                          op=">=", threshold_um=contact_size)),
        ("CO.E.D.1",          EnclosureCheck(inner=LayerRef(name="contact"),
                                              outer=LayerRef(name="diff"),
                                              op=">=", threshold_um=contact_in_diff)),
        ("M0.E.CO.2ADJ",      EnclosureCheck(inner=LayerRef(name="contact"),
                                              outer=LayerRef(name="m0"),
                                              op=">=", threshold_um=m0_enc_contact_2adj)),
        ("M0.E.CO.ALL",       EnclosureCheck(inner=LayerRef(name="contact"),
                                              outer=LayerRef(name="m0"),
                                              op=">=", threshold_um=m0_enc_contact_all)),
        ("V01.W.1",           WidthCheck(target=LayerRef(name="via_m0_m1"),
                                          op=">=", threshold_um=via01_size)),
        ("M0.E.V01.2ADJ",     EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                              outer=LayerRef(name="m0"),
                                              op=">=", threshold_um=m0_enc_via01_2adj)),
        ("M0.E.V01.ALL",      EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                              outer=LayerRef(name="m0"),
                                              op=">=", threshold_um=m0_enc_via01_all)),
        ("M1.E.V01.2ADJ",     EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                              outer=LayerRef(name="m1"),
                                              op=">=", threshold_um=m1_enc_via01_2adj)),
        ("M1.E.V01.ALL",      EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                              outer=LayerRef(name="m1"),
                                              op=">=", threshold_um=m1_enc_via01_all)),
        ("V12.W.1",           WidthCheck(target=LayerRef(name="via_m1_m2"),
                                          op=">=", threshold_um=via12_size)),
        ("M1.E.V12.2ADJ",     EnclosureCheck(inner=LayerRef(name="via_m1_m2"),
                                              outer=LayerRef(name="m1"),
                                              op=">=", threshold_um=m1_enc_via12_2adj)),
        ("M2.E.V12.2ADJ",     EnclosureCheck(inner=LayerRef(name="via_m1_m2"),
                                              outer=LayerRef(name="m2"),
                                              op=">=", threshold_um=m2_enc_via12_2adj)),
    ]
    for code, check in rules_to_seed:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))

    # On a collapsed PDK m0 and m1 share GDS; otherwise distinct.
    m0_layer = (67, 20)
    m1_layer = m0_layer if m0_collapsed else (68, 20)
    md = PDKMetadata(
        name="t", version="0",
        layers={
            "poly":      (66, 20),
            "diff":      (65, 20),
            "contact":   (66, 44),
            "m0":        m0_layer,
            "m1":        m1_layer,
            "m2":        (69, 20),
            "via_m0_m1": (67, 44),
            "via_m1_m2": (68, 44),
        },
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={},
    )
    mapping = BootstrapMapping(mapping={
        # contact
        "contact.size_um":                  "CO.W.1",
        "contact.enclosure_in_diff_um":     "CO.E.D.1",
        "m0.enclosure_of_contact_2adj_um":  "M0.E.CO.2ADJ",
        "m0.enclosure_of_contact_um":       "M0.E.CO.ALL",
        # via m0/m1
        "via_m0_m1.size_um":                  "V01.W.1",
        "m0.enclosure_of_via_m0_m1_2adj_um":  "M0.E.V01.2ADJ",
        "m0.enclosure_of_via_m0_m1_um":       "M0.E.V01.ALL",
        "m1.enclosure_of_via_m0_m1_2adj_um":  "M1.E.V01.2ADJ",
        "m1.enclosure_of_via_m0_m1_um":       "M1.E.V01.ALL",
        # via m1/m2
        "via_m1_m2.size_um":                  "V12.W.1",
        "m1.enclosure_of_via_m1_m2_2adj_um":  "M1.E.V12.2ADJ",
        "m2.enclosure_of_via_m1_m2_2adj_um":  "M2.E.V12.2ADJ",
    })
    return BootstrapRules(md, db, mapping)


# ── _stack_index ────────────────────────────────────────────────────────────

class TestStackIndex:
    def test_metals_indexed(self):
        assert _stack_index("m0") == 0
        assert _stack_index("m1") == 1
        assert _stack_index("m5") == 5

    def test_poly_and_diff_below_m0(self):
        # Both poly and diff connect to m0 via the same contact, so they
        # share the synthetic "below m0" index.
        assert _stack_index("poly") == -1
        assert _stack_index("diff") == -1

    def test_unknown_layer_raises(self):
        with pytest.raises(KeyError, match="canonical metal stack"):
            _stack_index("bogus")


# ── via_stack_between ───────────────────────────────────────────────────────

class TestViaStackBetween:
    def test_same_layer_no_transitions(self, tmp_path: Path):
        r = _rules(tmp_path)
        assert via_stack_between(r, "m0", "m0") == []

    def test_diff_to_m0_single_contact(self, tmp_path: Path):
        r = _rules(tmp_path)
        [t] = via_stack_between(r, "diff", "m0")
        assert isinstance(t, ViaTransition)
        assert t.via_layer == "contact"
        assert t.via_size  == pytest.approx(0.17)
        assert t.lower_metal == "diff"
        assert t.upper_metal == "m0"
        assert t.enc_lower == pytest.approx(0.04)        # contact in diff
        assert t.enc_upper == pytest.approx(0.06)        # m0 2adj
        assert t.enc_upper_opp == pytest.approx(0.04)    # m0 all-sides

    def test_m0_to_m1_single_via(self, tmp_path: Path):
        r = _rules(tmp_path)
        [t] = via_stack_between(r, "m0", "m1")
        assert t.via_layer == "via_m0_m1"
        assert t.lower_metal == "m0"
        assert t.upper_metal == "m1"
        assert t.enc_lower     == pytest.approx(0.085)
        assert t.enc_lower_opp == pytest.approx(0.06)
        assert t.enc_upper     == pytest.approx(0.085)
        assert t.enc_upper_opp == pytest.approx(0.06)

    def test_diff_to_m1_two_transitions(self, tmp_path: Path):
        r = _rules(tmp_path)
        ts = via_stack_between(r, "diff", "m1")
        assert [t.via_layer for t in ts] == ["contact", "via_m0_m1"]
        assert [t.lower_metal for t in ts] == ["diff", "m0"]
        assert [t.upper_metal for t in ts] == ["m0",   "m1"]

    def test_poly_to_m2_three_transitions(self, tmp_path: Path):
        r = _rules(tmp_path)
        ts = via_stack_between(r, "poly", "m2")
        assert [t.via_layer for t in ts] == ["contact", "via_m0_m1", "via_m1_m2"]

    def test_argument_order_is_normalised(self, tmp_path: Path):
        r = _rules(tmp_path)
        a = via_stack_between(r, "m0", "m2")
        b = via_stack_between(r, "m2", "m0")
        assert [t.via_layer for t in a] == [t.via_layer for t in b]

    def test_m0_collapsed_skips_m0_to_m1_via(self, tmp_path: Path):
        r = _rules(tmp_path, m0_collapsed=True)
        # m0 and m1 share a GDS layer → no m0→m1 cut.
        assert via_stack_between(r, "m0", "m1") == []
        # diff → m1 reduces to just the contact (no inter-metal via).
        ts = via_stack_between(r, "diff", "m1")
        assert [t.via_layer for t in ts] == ["contact"]
