"""Tests for ``lithos_layout.synth.placer`` — floorplan resolution."""
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

from lithos_layout       import BootstrapMapping, BootstrapRules
from lithos_layout.synth.loader import (
    CellTemplate,
    DeviceSpec,
    NetSpec,
    PlacementDirective,
)
from lithos_layout.synth.placer import (
    Placer,
    PlacedDevice,
    SPACING_RULES,
    _apply_cell_width,
    _topo_order,
    resolve_spacing_rule,
)
from lithos_layout.transistor import TransistorGeom


# ── Shared fixture: a minimal BootstrapRules instance ──────────────────────

def _rules(tmp_path: Path) -> BootstrapRules:
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-20T00:00:00Z")
    for code, check in [
        ("PO.W.1",   WidthCheck(target=LayerRef(name="poly"), op=">=", threshold_um=0.15)),
        ("PO.S.1",   SpacingCheck(layer_a=LayerRef(name="poly"), op=">=", threshold_um=0.21)),
        ("PO.E.1",   EnclosureCheck(inner=LayerRef(name="diff"), outer=LayerRef(name="poly"),
                                    op=">=", threshold_um=0.13)),
        ("DI.W.1",   WidthCheck(target=LayerRef(name="diff"), op=">=", threshold_um=0.15)),
        ("DI.S.1",   SpacingCheck(layer_a=LayerRef(name="diff"), op=">=", threshold_um=0.27)),
        ("CO.W.1",   WidthCheck(target=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.S.1",   SpacingCheck(layer_a=LayerRef(name="contact"), op=">=", threshold_um=0.17)),
        ("CO.E.D.1", EnclosureCheck(inner=LayerRef(name="contact"), outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.04)),
        ("M0.W.1",   WidthCheck(target=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M0.S.1",   SpacingCheck(layer_a=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("NW.W.1",   WidthCheck(target=LayerRef(name="nwell"), op=">=", threshold_um=0.84)),
        ("NW.S.1",   SpacingCheck(layer_a=LayerRef(name="nwell"), op=">=", threshold_um=1.27)),
        ("NW.E.D.1", EnclosureCheck(inner=LayerRef(name="diff"), outer=LayerRef(name="nwell"),
                                    op=">=", threshold_um=0.18)),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))

    metadata = PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20),
                "contact": (66, 44), "m0": (67, 20), "nwell": (64, 20)},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={
            "nmos": {"w_finger_max_um": 5.0, "sd_length_min_um": 0.29},
            "pmos": {"w_finger_max_um": 5.0, "sd_length_min_um": 0.29},
        },
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.spacing_min_um":          "PO.S.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "diff.width_min_um":            "DI.W.1",
        "diff.spacing_min_um":          "DI.S.1",
        "contact.size_um":              "CO.W.1",
        "contact.spacing_um":           "CO.S.1",
        "contact.enclosure_in_diff_um": "CO.E.D.1",
        "m0.width_min_um":              "M0.W.1",
        "m0.spacing_min_um":            "M0.S.1",
        "nwell.width_min_um":           "NW.W.1",
        "nwell.spacing_min_um":         "NW.S.1",
        "nwell.enclosure_of_pdiff_um":  "NW.E.D.1",
    })
    return BootstrapRules(metadata, db, mapping)


def _inverter_template() -> CellTemplate:
    return CellTemplate(
        name              = "inv",
        description       = "",
        devices           = {
            "N": DeviceSpec(name="N", template="planar_mosfet", device_type="nmos",
                            terminals={"G": "IN", "D": "OUT", "S": "GND"}),
            "P": DeviceSpec(name="P", template="planar_mosfet", device_type="pmos",
                            terminals={"G": "IN", "D": "OUT", "S": "VDD"}),
        },
        nets              = {
            "VDD": NetSpec(name="VDD", net_type="power", rail="top"),
            "GND": NetSpec(name="GND", net_type="power", rail="bottom"),
            "IN":  NetSpec(name="IN",  net_type="signal"),
            "OUT": NetSpec(name="OUT", net_type="signal"),
        },
        ports             = {},
        named_constraints = {},
        layout_mode       = "directives",
        placement_directives = [
            PlacementDirective(name="N", origin=(0.0, 0.0)),
            PlacementDirective(name="P", relative_to="N", relation="align_gate",
                               alignment="gate"),
        ],
    )


# ── Pure helpers ────────────────────────────────────────────────────────────

class TestTopoOrder:
    def test_no_dependencies(self):
        devs = {
            "A": DeviceSpec(name="A", template="planar_mosfet", device_type="nmos",
                            terminals={}, region="bottom"),
            "B": DeviceSpec(name="B", template="planar_mosfet", device_type="pmos",
                            terminals={}, region="top"),
        }
        order = _topo_order(devs)
        assert set(order) == {"A", "B"}
        # Bottom devices come first when there's no dependency.
        assert order.index("A") < order.index("B")

    def test_x_dependency(self):
        devs = {
            "A": DeviceSpec(name="A", template="planar_mosfet", device_type="nmos",
                            terminals={}),
            "B": DeviceSpec(name="B", template="planar_mosfet", device_type="nmos",
                            terminals={}, x_spec="A_x + A.total_x"),
        }
        order = _topo_order(devs)
        assert order.index("A") < order.index("B")

    def test_y_dependency(self):
        devs = {
            "first":  DeviceSpec(name="first", template="planar_mosfet", device_type="nmos",
                                 terminals={}),
            "second": DeviceSpec(name="second", template="planar_mosfet", device_type="pmos",
                                 terminals={}, y_offset_expr="first.total_y + 0.01"),
        }
        assert _topo_order(devs)[0] == "first"


class TestApplyCellWidth:
    def _device(self, name: str, x: float, total_x: float = 1.0) -> PlacedDevice:
        return PlacedDevice(
            name = name,
            spec = DeviceSpec(name=name, template="planar_mosfet",
                              device_type="nmos", terminals={}),
            geom = TransistorGeom(
                w_um=0.5, l_um=0.15, device_type="nmos", n_fingers=1,
                w_finger_um=0.5, sd_length_um=0.2, n_contacts_y=1,
                total_x_um=total_x, total_y_um=0.6,
            ),
            x = x, y = 0.0,
        )

    def test_no_change_when_devices_wider(self):
        placed = {"A": self._device("A", 0.0, total_x=2.0)}
        _apply_cell_width(placed, target_width=1.0)
        assert placed["A"].x == 0.0

    def test_centres_devices_inside_target_width(self):
        placed = {"A": self._device("A", 0.0, total_x=1.0)}
        _apply_cell_width(placed, target_width=3.0)
        # 1µm device in a 3µm cell → centred at x=1.0.
        assert placed["A"].x == pytest.approx(1.0)

    def test_empty_no_op(self):
        placed = {}
        _apply_cell_width(placed, target_width=2.0)
        assert placed == {}


# ── Spacing-rule registry ───────────────────────────────────────────────────

class TestSpacingRules:
    def test_registry_keys(self):
        assert set(SPACING_RULES) >= {
            "min_diff_spacing", "inter_cell_gap",
            "cross_couple_wiring", "min_well_separation",
        }

    def test_min_diff_spacing(self, tmp_path: Path):
        r = _rules(tmp_path)
        assert resolve_spacing_rule("min_diff_spacing", r) == pytest.approx(0.27)

    def test_unknown_raises(self, tmp_path: Path):
        r = _rules(tmp_path)
        with pytest.raises(KeyError, match="Unknown spacing_rule"):
            resolve_spacing_rule("bogus", r)


# ── End-to-end Placer.place() ───────────────────────────────────────────────

class TestPlaceInverter:
    def test_two_devices_placed(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        assert set(placed) == {"N", "P"}

    def test_nmos_at_origin(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        assert placed["N"].x == 0.0
        assert placed["N"].y == 0.0

    def test_pmos_above_nmos(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        # PMOS is gate-aligned (same X) and stacked above (Y > NMOS).
        assert placed["P"].x == placed["N"].x
        assert placed["P"].y > placed["N"].y + placed["N"].geom.total_y_um

    def test_grid_snap(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        grid   = rules.mfg_grid
        for d in placed.values():
            assert abs(round(d.x / grid) * grid - d.x) < 1e-9
            assert abs(round(d.y / grid) * grid - d.y) < 1e-9


class TestPlaceWithDeviceOverrides:
    def test_per_device_w_override_picked_up(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        tpl.device_params["N"] = {"w": 1.0}
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        assert placed["N"].geom.w_um == pytest.approx(1.0)
        assert placed["P"].geom.w_um == pytest.approx(0.52)


class TestPlaceFixedCellWidth:
    def test_centres_within_target_width(self, tmp_path: Path):
        rules = _rules(tmp_path)
        tpl   = _inverter_template()
        # Active area is well under 3.6 um, so the cell width should
        # centre the devices.
        tpl.cell_dimensions.width = 3.6
        placed = Placer(rules, params={"w": 0.52, "l": 0.15}).place(tpl)
        x_min = min(d.x for d in placed.values())
        x_max = max(d.x + d.geom.total_x_um for d in placed.values())
        # Left margin should approximately equal right margin.
        left  = x_min
        right = 3.6 - x_max
        assert left == pytest.approx(right, abs=rules.mfg_grid)
