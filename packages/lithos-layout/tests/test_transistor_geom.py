"""Transistor dimension-math tests against a synthetic BootstrapRules."""
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
    SpacingCheck,
    WidthCheck,
)

from lithos_layout import (
    BootstrapMapping,
    BootstrapRules,
    TransistorGeom,
    finger_count,
    sd_contact_columns,
    transistor_geom,
)


# ── Shared fixture: a generic m0/contact rule set in a synthetic DB ────────

def _rules(tmp_path: Path) -> BootstrapRules:
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
    for code, check in [
        ("PO.W.1",   WidthCheck(target=LayerRef(name="poly"), op=">=", threshold_um=0.15)),
        ("PO.E.1",   EnclosureCheck(inner=LayerRef(name="diff"), outer=LayerRef(name="poly"),
                                    op=">=", threshold_um=0.13)),
        ("CO.W.1",   WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.S.1",   SpacingCheck(layer_a=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.E.D.1", EnclosureCheck(inner=LayerRef(name="contact"), outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.04)),
        ("DI.W.1",   WidthCheck(target=LayerRef(name="diff"), op=">=", threshold_um=0.15)),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))

    metadata = PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20),
                "contact": (66, 44), "m0": (67, 20)},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={
            "nmos": {
                "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
            "pmos": {
                "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
        },
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "contact.size_um":              "CO.W.1",
        "contact.spacing_um":           "CO.S.1",
        "contact.enclosure_in_diff_um": "CO.E.D.1",
        "diff.width_min_um":            "DI.W.1",
    })
    return BootstrapRules(metadata, db, mapping)


# ── finger_count ────────────────────────────────────────────────────────────

def test_finger_count_one_when_fits(tmp_path: Path):
    r = _rules(tmp_path)
    assert finger_count(0.6, r, "nmos") == 1


def test_finger_count_grows_above_w_max(tmp_path: Path):
    r = _rules(tmp_path)
    # w_finger_max_um = 5.0 → W=8 needs ceil(8/5) = 2 fingers, each 4 µm.
    assert finger_count(8.0, r, "nmos") == 2


def test_finger_count_minimum_one(tmp_path: Path):
    r = _rules(tmp_path)
    # Tiny W still produces at least one finger.
    assert finger_count(0.01, r, "nmos") == 1


def test_finger_count_clamps_below_w_min(tmp_path: Path):
    """If too many fingers would push w_finger below min channel width,
    clamp down so each finger meets the minimum."""
    r = _rules(tmp_path)
    # Min channel width = 0.15. W = 0.30 with naive math: 1 finger of 0.30
    # (since 0.30 < w_finger_max = 5). Clamp ensures n <= W/w_min = 2.
    # So upper bound is 2; ceil(0.30/5) = 1. Result stays 1.
    assert finger_count(0.30, r, "nmos") == 1


# ── sd_contact_columns ─────────────────────────────────────────────────────

def test_sd_contact_columns_minimum_one_when_too_narrow(tmp_path: Path):
    r = _rules(tmp_path)
    assert sd_contact_columns(0.10, r) == 1


def test_sd_contact_columns_scales_with_width(tmp_path: Path):
    r = _rules(tmp_path)
    # contact size=0.17, spacing=0.17, enclosure=0.04 each side.
    # usable = w_finger - 0.08. Fits n when n*size + (n-1)*space <= usable.
    # For 1 µm: usable = 0.92; (0.92 + 0.17) / (0.17 + 0.17) = 1.09/0.34 ≈ 3.21
    # → 3 contacts.
    assert sd_contact_columns(1.0, r) == 3


# ── transistor_geom ────────────────────────────────────────────────────────

def test_transistor_geom_single_finger(tmp_path: Path):
    r = _rules(tmp_path)
    g = transistor_geom(0.52, 0.15, "nmos", r)
    assert isinstance(g, TransistorGeom)
    assert g.n_fingers == 1
    assert g.w_finger_um == pytest.approx(0.52)
    # S/D length: max(0.29, 0.17 + 2*0.04) = max(0.29, 0.25) = 0.29
    assert g.sd_length_um == pytest.approx(0.29)
    # total_x = (1+1)*0.29 + 1*0.15 = 0.58 + 0.15 = 0.73
    assert g.total_x_um == pytest.approx(0.73)
    # total_y = w_finger + 2*endcap = 0.52 + 2*0.13 = 0.78
    assert g.total_y_um == pytest.approx(0.78)


def test_transistor_geom_multi_finger(tmp_path: Path):
    r = _rules(tmp_path)
    g = transistor_geom(8.0, 0.15, "nmos", r)
    assert g.n_fingers == 2
    assert g.w_finger_um == pytest.approx(4.0)
    # total_x = (2+1)*0.29 + 2*0.15 = 0.87 + 0.30 = 1.17
    assert g.total_x_um == pytest.approx(1.17)
    # total_y = 4 + 2*0.13 = 4.26
    assert g.total_y_um == pytest.approx(4.26)


def test_transistor_geom_pmos_uses_pmos_device(tmp_path: Path):
    r = _rules(tmp_path)
    g_n = transistor_geom(0.5, 0.15, "nmos", r)
    g_p = transistor_geom(0.5, 0.15, "pmos", r)
    # Both share the same device record shape in this synthetic PDK.
    assert g_n.total_x_um == pytest.approx(g_p.total_x_um)
    assert g_p.device_type == "pmos"


def test_transistor_geom_unknown_device_raises(tmp_path: Path):
    r = _rules(tmp_path)
    with pytest.raises(KeyError, match="Device 'jfet'"):
        transistor_geom(0.5, 0.15, "jfet", r)
