"""lithos_layout.cells.tap — standalone well/substrate tap cell."""
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
    draw_tap_cell,
)


# ── Fixture ────────────────────────────────────────────────────────────────

_LAYERS = {
    "poly":      (66, 20),
    "diff":      (65, 20),
    "tap":       (65, 44),
    "contact":   (66, 44),
    "m0":        (67, 20),
    "m1":        (68, 20),
    "via_m0_m1": (67, 44),
    "nimplant":  (93, 44),
    "pimplant":  (94, 20),
    "nwell":     (64, 20),
}


def _seeded_db(path: Path) -> RuleDB:
    db = RuleDB(path)
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-19T00:00:00Z")
    for code, check in [
        ("CO.W.1",       WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.E.M0.2A",   EnclosureCheck(inner=LayerRef(name="contact"),
                                        outer=LayerRef(name="m0"),
                                        op=">=", threshold_um=0.08,
                                        on_sides="two_adjacent")),
        ("DI.S.1",       SpacingCheck(layer_a=LayerRef(name="diff"), op=">=", threshold_um=0.27)),
        ("M0.W.1",       WidthCheck(target=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M1.W.1",       WidthCheck(target=LayerRef(name="m1"), op=">=", threshold_um=0.23)),
        ("IMP.E.1",      EnclosureCheck(inner=LayerRef(name="diff"),
                                        outer=LayerRef(name="nimplant"),
                                        op=">=", threshold_um=0.125)),
        ("NW.E.PDIFF",   EnclosureCheck(inner=LayerRef(name="diff"),
                                        outer=LayerRef(name="nwell"),
                                        op=">=", threshold_um=0.18)),
        ("NW.W.1",       WidthCheck(target=LayerRef(name="nwell"),
                                    op=">=", threshold_um=0.84)),
        ("V01.W.1",      WidthCheck(target=LayerRef(name="via_m0_m1"),
                                    op=">=", threshold_um=0.17)),
        ("V01.E.M1.2A",  EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                        outer=LayerRef(name="m1"),
                                        op=">=", threshold_um=0.06,
                                        on_sides="two_adjacent")),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    return db


def _metadata(layers: dict | None = None) -> PDKMetadata:
    return PDKMetadata(
        name="t", version="0",
        layers=layers if layers is not None else _LAYERS,
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={},
    )


def _mapping() -> BootstrapMapping:
    return BootstrapMapping(mapping={
        "contact.size_um":                   "CO.W.1",
        "contact.enclosure_in_m0_2adj_um":   "CO.E.M0.2A",
        "diff.spacing_min_um":               "DI.S.1",
        "m0.width_min_um":                   "M0.W.1",
        "m1.width_min_um":                   "M1.W.1",
        "implant.enclosure_of_diff_um":      "IMP.E.1",
        "nwell.enclosure_of_pdiff_um":       "NW.E.PDIFF",
        "nwell.width_min_um":                "NW.W.1",
        "via_m0_m1.size_um":                 "V01.W.1",
        "m1.enclosure_of_via_m0_m1_2adj_um": "V01.E.M1.2A",
    })


def _rules(tmp_path: Path, *, layers: dict | None = None) -> BootstrapRules:
    db = _seeded_db(tmp_path / "rules.db")
    return BootstrapRules(_metadata(layers), db, _mapping())


def _layers_present(component) -> set[tuple[int, int]]:
    """All (gds_layer, datatype) pairs holding ≥1 shape after in-place flatten."""
    component.flatten()
    kc = component.kdb_cell
    layout = kc.layout()
    out: set[tuple[int, int]] = set()
    for layer_idx in range(layout.layers()):
        if any(True for _ in kc.each_shape(layer_idx)):
            info = layout.get_info(layer_idx)
            out.add((info.layer, info.datatype))
    return out


# ── Tests ──────────────────────────────────────────────────────────────────

def test_returns_gdsfactory_component(tmp_path: Path):
    import gdsfactory as gf
    r = _rules(tmp_path)
    try:
        c = draw_tap_cell(r)
        assert isinstance(c, gf.Component)
    finally:
        r.db.close()


def test_tap_cell_has_both_implants_and_nwell(tmp_path: Path):
    """ptap (pimplant) at bottom, ntap (nimplant) at top, nwell over ntap."""
    r = _rules(tmp_path)
    try:
        c = draw_tap_cell(r)
        present = _layers_present(c)
        assert _LAYERS["pimplant"] in present
        assert _LAYERS["nimplant"] in present
        assert _LAYERS["nwell"]    in present
    finally:
        r.db.close()


def test_tap_cell_draws_full_via_stack(tmp_path: Path):
    """Tap cell should land power on m1, going tap → contact → m0 → via → m1."""
    r = _rules(tmp_path)
    try:
        c = draw_tap_cell(r)
        present = _layers_present(c)
        assert _LAYERS["tap"]       in present
        assert _LAYERS["contact"]   in present
        assert _LAYERS["m0"]        in present
        assert _LAYERS["m1"]        in present
        assert _LAYERS["via_m0_m1"] in present
    finally:
        r.db.close()


def test_tap_cell_emits_vdd_and_gnd_ports(tmp_path: Path):
    r = _rules(tmp_path)
    try:
        c = draw_tap_cell(r)
        port_names = {p.name for p in c.ports}
        assert port_names == {"VDD", "GND"}
        ports = {p.name: p for p in c.ports}
        assert ports["VDD"].orientation == 90   # faces north
        assert ports["GND"].orientation == 270  # faces south
    finally:
        r.db.close()


def test_tap_cell_explicit_height(tmp_path: Path):
    """Caller-supplied cell_height drives where the VDD rail lands."""
    r = _rules(tmp_path)
    try:
        c = draw_tap_cell(r, cell_height=4.0)
        ports = {p.name: p for p in c.ports}
        rail_h = r.m1["width_min_um"]
        # VDD port sits half a rail-width below the top of the cell.
        assert ports["VDD"].center[1] == pytest.approx(4.0 - rail_h / 2)
        assert ports["GND"].center[1] == pytest.approx(rail_h / 2)
    finally:
        r.db.close()


def test_tap_cell_falls_back_to_diff_when_no_tap_layer(tmp_path: Path):
    """A PDK without a dedicated 'tap' layer reuses 'diff' for the tap diffusion."""
    layers = {k: v for k, v in _LAYERS.items() if k != "tap"}
    r = _rules(tmp_path, layers=layers)
    try:
        c = draw_tap_cell(r)
        present = _layers_present(c)
        assert _LAYERS["diff"] in present  # tap diffusion drawn on 'diff'
    finally:
        r.db.close()


def test_tap_cell_no_via_when_m0_collapsed_to_m1(tmp_path: Path):
    """When the PDK collapses m0 onto m1, no via_m0_m1 cut should appear."""
    layers = dict(_LAYERS)
    layers["m0"] = layers["m1"]
    r = _rules(tmp_path, layers=layers)
    try:
        assert r.m0_is_m1
        c = draw_tap_cell(r)
        present = _layers_present(c)
        # via_m0_m1 cut should be omitted.
        assert _LAYERS["via_m0_m1"] not in present
    finally:
        r.db.close()
