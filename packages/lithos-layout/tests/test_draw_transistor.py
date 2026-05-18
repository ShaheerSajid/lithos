"""draw_transistor: end-to-end GDS emitter against a synthetic rule DB.

Verifies that the ported polygon construction produces a gdsfactory
Component with the expected layers populated and the G/S/D ports placed
correctly. The synthetic DB / metadata / bootstrap mapping is the
smallest sky130-flavoured set that exercises every code path
(diffusion, implant, gate, contacts, li1, asymmetric li1 enclosure,
NPC, N-well for PMOS).
"""
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
    draw_transistor,
)


# ── Fixtures: a sky130-flavoured synthetic PDK ──────────────────────────────

def _seeded_db(path: Path) -> RuleDB:
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-18T00:00:00Z")
    rules = [
        # poly
        ("PO.W.1",          WidthCheck(    target=LayerRef(name="poly"),     op=">=", threshold_um=0.15)),
        ("PO.E.1",          EnclosureCheck(inner=LayerRef(name="diff"),
                                           outer=LayerRef(name="poly"),
                                           op=">=", threshold_um=0.13)),
        # diff
        ("DI.W.1",          WidthCheck(    target=LayerRef(name="diff"),     op=">=", threshold_um=0.15)),
        ("DI.E.1",          EnclosureCheck(inner=LayerRef(name="poly"),
                                           outer=LayerRef(name="diff"),
                                           op=">=", threshold_um=0.25)),
        # contacts
        ("CO.W.1",          WidthCheck(    target=LayerRef(name="licon1"),   op=">=", threshold_um=0.17)),
        ("CO.S.1",          SpacingCheck(  layer_a=LayerRef(name="licon1"),  op=">=", threshold_um=0.17)),
        ("CO.E.D.1",        EnclosureCheck(inner=LayerRef(name="licon1"),
                                           outer=LayerRef(name="diff"),
                                           op=">=", threshold_um=0.04)),
        ("CO.E.LI.2ADJ",    EnclosureCheck(inner=LayerRef(name="licon1"),
                                           outer=LayerRef(name="li1"),
                                           op=">=", threshold_um=0.08,
                                           on_sides="two_adjacent")),
        ("CO.E.LI.ALL",     EnclosureCheck(inner=LayerRef(name="licon1"),
                                           outer=LayerRef(name="li1"),
                                           op=">=", threshold_um=0.0)),
        ("CO.SPACE.POLY",   SpacingCheck(  layer_a=LayerRef(name="licon1"),
                                           layer_b=LayerRef(name="poly"),
                                           op=">=", threshold_um=0.055)),
        # li1
        ("LI.W.1",          WidthCheck(    target=LayerRef(name="li1"),      op=">=", threshold_um=0.17)),
        ("LI.S.1",          SpacingCheck(  layer_a=LayerRef(name="li1"),     op=">=", threshold_um=0.17)),
        # implant
        ("IMP.E.1",         EnclosureCheck(inner=LayerRef(name="diff"),
                                           outer=LayerRef(name="nsdm"),
                                           op=">=", threshold_um=0.125)),
        # n-well
        ("NW.E.PDIFF",      EnclosureCheck(inner=LayerRef(name="diff"),
                                           outer=LayerRef(name="nwell"),
                                           op=">=", threshold_um=0.18)),
        # npc
        ("NPC.E.POLY",      EnclosureCheck(inner=LayerRef(name="poly"),
                                           outer=LayerRef(name="npc"),
                                           op=">=", threshold_um=0.10)),
    ]
    for code, check in rules:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    return db


def _metadata(with_npc: bool = True, with_nwell: bool = True) -> PDKMetadata:
    layers = {
        "poly":   (66, 20),
        "diff":   (65, 20),
        "licon1": (66, 44),
        "li1":    (67, 20),
        "nsdm":   (93, 44),
        "psdm":   (94, 20),
    }
    if with_nwell:
        layers["nwell"] = (64, 20)
    if with_npc:
        layers["npc"] = (95, 20)
    return PDKMetadata(
        name="t", version="0",
        layers=layers,
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={
            "nmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "nsdm", "bulk_layer": "pwell",
                "nwell": False, "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
            "pmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "psdm", "bulk_layer": "nwell",
                "nwell": True, "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
        },
    )


def _mapping(with_optional: bool = True) -> BootstrapMapping:
    base = {
        "poly.width_min_um":               "PO.W.1",
        "poly.endcap_over_diff_um":        "PO.E.1",
        "diff.width_min_um":               "DI.W.1",
        "contacts.size_um":                "CO.W.1",
        "contacts.spacing_um":             "CO.S.1",
        "contacts.enclosure_in_diff_um":   "CO.E.D.1",
        "contacts.enclosure_in_li1_2adj_um": "CO.E.LI.2ADJ",
        "contacts.enclosure_in_li1_um":      "CO.E.LI.ALL",
        "li1.width_min_um":                "LI.W.1",
    }
    if with_optional:
        base["implant.enclosure_of_diff_um"]  = "IMP.E.1"
        base["nwell.enclosure_of_pdiff_um"]   = "NW.E.PDIFF"
        base["npc.enclosure_of_poly_um"]      = "NPC.E.POLY"
    return BootstrapMapping(mapping=base)


def _rules(tmp_path: Path, **kw) -> BootstrapRules:
    db = _seeded_db(tmp_path / "rules.db")
    return BootstrapRules(_metadata(**kw), db, _mapping())


def _polygons_by_layer(component) -> dict[tuple[int, int], int]:
    """Return ``{(layer, datatype): polygon_count}`` for the component.

    gdsfactory 9.x doesn't expose a flat ``.polygons`` list; we iterate
    the underlying KLayout cell and map layer indices → (layer, datatype)
    via the layout's info table.
    """
    kc = component.kdb_cell
    layout = kc.layout()
    out: dict[tuple[int, int], int] = {}
    for layer_idx in range(layout.layers()):
        info = layout.get_info(layer_idx)
        n = sum(1 for _ in kc.each_shape(layer_idx))
        if n > 0:
            out[(info.layer, info.datatype)] = n
    return out


# ── Basic shape ─────────────────────────────────────────────────────────────

def test_returns_gdsfactory_component(tmp_path: Path):
    import gdsfactory as gf
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        assert isinstance(c, gf.Component)
    finally:
        r.db.close()


def test_nmos_has_diff_poly_implant_contacts_li1(tmp_path: Path):
    """All required layers appear on the drawn nmos component."""
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        present = _polygons_by_layer(c)
        assert r.layer("diff")   in present
        assert r.layer("poly")   in present
        assert r.layer("licon1") in present
        assert r.layer("li1")    in present
        assert r.layer("nsdm")   in present     # nmos implant
    finally:
        r.db.close()


def test_pmos_adds_nwell(tmp_path: Path):
    """PMOS component carries an N-well polygon enclosing the diffusion."""
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "pmos", r)
        present = _polygons_by_layer(c)
        assert r.layer("nwell") in present
        assert r.layer("psdm")  in present      # pmos implant
    finally:
        r.db.close()


def test_nmos_omits_nwell(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        present = _polygons_by_layer(c)
        assert r.layer("nwell") not in present
    finally:
        r.db.close()


# ── Optional layers gracefully omitted ─────────────────────────────────────

def test_no_npc_when_layer_missing(tmp_path: Path):
    """A PDK without an ``npc`` layer skips NPC drawing."""
    r = BootstrapRules(_metadata(with_npc=False), _seeded_db(tmp_path / "x.db"), _mapping())
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        with pytest.raises(KeyError):
            r.layer("npc")
        # Drawing still works — at least the diffusion and gate are present.
        present = _polygons_by_layer(c)
        assert r.layer("diff") in present
        assert r.layer("poly") in present
    finally:
        r.db.close()


# ── Port placement ─────────────────────────────────────────────────────────

def test_ports_g_s_d_present_and_oriented(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        ports = {p.name: p for p in c.ports}
        assert set(ports) == {"G", "S", "D"}
        # G faces up (top edge of poly), S faces west, D faces east.
        assert ports["G"].orientation == 90
        assert ports["S"].orientation == 180
        assert ports["D"].orientation == 0
    finally:
        r.db.close()


def test_port_x_positions_bracket_the_device(tmp_path: Path):
    """Source port should sit to the left of drain (single-finger transistor)."""
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.52, 0.15, "nmos", r)
        ports = {p.name: p for p in c.ports}
        sx = float(ports["S"].center[0])
        dx = float(ports["D"].center[0])
        assert sx < dx
    finally:
        r.db.close()


# ── Multi-finger ───────────────────────────────────────────────────────────

def test_multi_finger_draws_n_poly_fingers(tmp_path: Path):
    """W=8 with w_finger_max=5 → 2 fingers → 2 poly polygons."""
    r = _rules(tmp_path)
    try:
        c = draw_transistor(8.0, 0.15, "nmos", r)
        assert _polygons_by_layer(c)[r.layer("poly")] == 2
    finally:
        r.db.close()


def test_explicit_n_fingers_override(tmp_path: Path):
    """An explicit n_fingers override produces that count of poly polygons."""
    r = _rules(tmp_path)
    try:
        c = draw_transistor(0.6, 0.15, "nmos", r, n_fingers=3)
        assert _polygons_by_layer(c)[r.layer("poly")] == 3
    finally:
        r.db.close()


# ── skip_sd ────────────────────────────────────────────────────────────────

def test_skip_sd_omits_li1_strips_on_skipped_indices(tmp_path: Path):
    """Skipping an S/D index leaves no li1 / contact on that region.

    A 3-finger device has 4 S/D regions. Skipping the two internal ones
    should drop their li1 strips, leaving 2 li1 strips (source + drain).
    """
    r = _rules(tmp_path)
    try:
        c_full    = draw_transistor(0.6, 0.15, "nmos", r, n_fingers=3)
        c_skipped = draw_transistor(0.6, 0.15, "nmos", r, n_fingers=3, skip_sd={1, 2})
        li_layer  = r.layer("li1")
        assert _polygons_by_layer(c_full)[li_layer]    == 4
        assert _polygons_by_layer(c_skipped)[li_layer] == 2
    finally:
        r.db.close()
