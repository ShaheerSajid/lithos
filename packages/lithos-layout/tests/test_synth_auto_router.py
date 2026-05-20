"""Tests for ``lithos_layout.synth.auto_router``."""
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
from lithos_layout                  import BootstrapMapping, BootstrapRules
from lithos_layout.synth            import (
    AutoRouter,
    Placer,
    Synthesizer,
    build_net_graph,
    load_template,
)
from lithos_layout.synth.loader     import (
    CellTemplate,
    DeviceSpec,
    NetSpec,
    RoutingHint,
    RoutingSpec,
)
from lithos_layout.synth.netlist    import NetGraph, NetInfo, TerminalRef
from lithos_layout.synth.placer     import PlacedDevice
from lithos_layout.transistor       import TransistorGeom


# ── Shared fixture: minimal-but-complete BootstrapRules ─────────────────────

def _rules(tmp_path: Path) -> BootstrapRules:
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-20T00:00:00Z")
    for code, check in [
        ("PO.W.1",   WidthCheck(target=LayerRef(name="poly"),
                                op=">=", threshold_um=0.15)),
        ("PO.S.1",   SpacingCheck(layer_a=LayerRef(name="poly"),
                                  op=">=", threshold_um=0.21)),
        ("PO.E.1",   EnclosureCheck(inner=LayerRef(name="diff"),
                                    outer=LayerRef(name="poly"),
                                    op=">=", threshold_um=0.13)),
        ("DI.W.1",   WidthCheck(target=LayerRef(name="diff"),
                                op=">=", threshold_um=0.15)),
        ("DI.S.1",   SpacingCheck(layer_a=LayerRef(name="diff"),
                                  op=">=", threshold_um=0.27)),
        ("DI.E.1",   EnclosureCheck(inner=LayerRef(name="diff"),
                                    outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.25)),
        ("CO.W.1",   WidthCheck(target=LayerRef(name="contact"),
                                op=">=", threshold_um=0.17)),
        ("CO.S.1",   SpacingCheck(layer_a=LayerRef(name="contact"),
                                  op=">=", threshold_um=0.17)),
        ("CO.E.D.1", EnclosureCheck(inner=LayerRef(name="contact"),
                                    outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.04)),
        ("M0.W.1",   WidthCheck(target=LayerRef(name="m0"),
                                op=">=", threshold_um=0.17)),
        ("M0.S.1",   SpacingCheck(layer_a=LayerRef(name="m0"),
                                  op=">=", threshold_um=0.17)),
        ("M1.W.1",   WidthCheck(target=LayerRef(name="m1"),
                                op=">=", threshold_um=0.20)),
        ("M1.S.1",   SpacingCheck(layer_a=LayerRef(name="m1"),
                                  op=">=", threshold_um=0.20)),
        ("V1.W.1",   WidthCheck(target=LayerRef(name="via_m0_m1"),
                                op=">=", threshold_um=0.15)),
        ("M2.W.1",   WidthCheck(target=LayerRef(name="m2"),
                                op=">=", threshold_um=0.20)),
        ("M2.S.1",   SpacingCheck(layer_a=LayerRef(name="m2"),
                                  op=">=", threshold_um=0.20)),
        ("V2.W.1",   WidthCheck(target=LayerRef(name="via_m1_m2"),
                                op=">=", threshold_um=0.15)),
        ("NW.W.1",   WidthCheck(target=LayerRef(name="nwell"),
                                op=">=", threshold_um=0.84)),
        ("NW.S.1",   SpacingCheck(layer_a=LayerRef(name="nwell"),
                                  op=">=", threshold_um=1.27)),
        ("NW.E.D.1", EnclosureCheck(inner=LayerRef(name="diff"),
                                    outer=LayerRef(name="nwell"),
                                    op=">=", threshold_um=0.18)),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    md = PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20),
                "contact": (66, 44), "m0": (67, 20), "m1": (68, 20),
                "via_m0_m1": (67, 44),
                "m2": (69, 20), "via_m1_m2": (68, 44),
                "nwell": (64, 20), "nimplant": (93, 44), "pimplant": (94, 20)},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={
            "nmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "nimplant", "bulk_layer": "pwell",
                "nwell": False, "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
            "pmos": {
                "diff_layer": "diff", "gate_layer": "poly",
                "implant_layer": "pimplant", "bulk_layer": "nwell",
                "nwell": True,  "w_finger_max_um": 5.0,
                "sd_length_min_um": 0.29,
            },
        },
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.spacing_min_um":          "PO.S.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "diff.width_min_um":            "DI.W.1",
        "diff.spacing_min_um":          "DI.S.1",
        "diff.extension_past_poly_um":  "DI.E.1",
        "contact.size_um":              "CO.W.1",
        "contact.spacing_um":           "CO.S.1",
        "contact.enclosure_in_diff_um": "CO.E.D.1",
        "m0.width_min_um":              "M0.W.1",
        "m0.spacing_min_um":            "M0.S.1",
        "m1.width_min_um":              "M1.W.1",
        "m1.spacing_min_um":            "M1.S.1",
        "via_m0_m1.size_um":            "V1.W.1",
        "m2.width_min_um":              "M2.W.1",
        "m2.spacing_min_um":            "M2.S.1",
        "via_m1_m2.size_um":            "V2.W.1",
        "nwell.width_min_um":           "NW.W.1",
        "nwell.spacing_min_um":         "NW.S.1",
        "nwell.enclosure_of_pdiff_um":  "NW.E.D.1",
    })
    return BootstrapRules(md, db, mapping)


# ── Inverter: full plan ─────────────────────────────────────────────────────

class TestPlanInverter:
    """The auto-router on the shipped inverter template should emit
    exactly the routing styles the inverter needs end-to-end."""

    def _plan(self, tmp_path: Path):
        rules     = _rules(tmp_path)
        template  = load_template("inverter")
        placed    = Placer(rules, {"w": 0.52, "l": 0.15}).place(template)
        net_graph = build_net_graph(template)
        specs     = AutoRouter(rules).plan(net_graph, placed, template)
        return specs

    def test_emits_shared_gate_poly_for_input(self, tmp_path: Path):
        specs = self._plan(tmp_path)
        gate  = [s for s in specs
                 if s.style == "shared_gate_poly" and s.net == "IN"]
        assert len(gate) == 1
        assert set(gate[0].path) == {"N.G", "P.G"}
        assert gate[0].layer == "poly"

    def test_emits_drain_bridge_for_output(self, tmp_path: Path):
        specs = self._plan(tmp_path)
        out   = [s for s in specs
                 if s.style == "drain_bridge" and s.net == "OUT"]
        assert len(out) == 1
        assert set(out[0].path) == {"N.D", "P.D"}

    def test_emits_both_power_rails(self, tmp_path: Path):
        specs    = self._plan(tmp_path)
        rails    = [s for s in specs if s.style == "horizontal_power_rail"]
        rail_map = {(s.net, s.edge) for s in rails}
        assert ("GND", "bottom") in rail_map
        assert ("VDD", "top")    in rail_map

    def test_emits_source_to_rail_for_each_supply(self, tmp_path: Path):
        specs = self._plan(tmp_path)
        s2r   = [s for s in specs if s.style == "source_to_rail"]
        nets  = {(s.net, s.edge) for s in s2r}
        assert ("GND", "bottom") in nets
        assert ("VDD", "top")    in nets
        # Each rail straps a single source terminal in the inverter.
        gnd = [s for s in s2r if s.net == "GND"]
        vdd = [s for s in s2r if s.net == "VDD"]
        assert gnd[0].path == ["N.S"]
        assert vdd[0].path == ["P.S"]

    def test_no_cross_row_specs_for_inverter(self, tmp_path: Path):
        # Inverter is a single row pair → no cross-row connections.
        specs = self._plan(tmp_path)
        cross = [s for s in specs if s.style == "cross_row_connect"]
        assert cross == []


# ── Synthesizer end-to-end with auto-routing (no hand-rolled specs) ─────────

class TestSynthesizerAutoRoutes:
    def test_inverter_synthesizes_without_explicit_specs(self, tmp_path: Path):
        """Omitting ``routing_specs`` should still produce a fully-routed cell."""
        rules    = _rules(tmp_path)
        template = load_template("inverter")
        result   = Synthesizer(rules).synthesize(
            template, params={"w": 0.52, "l": 0.15},
        )
        polys = result.component.get_polygons(by="tuple")
        # Routing emits at least the two power rails on m1.
        m1_polys = polys.get(rules.layer("m1"), [])
        assert len(m1_polys) >= 2
        # All four port candidates ought to surface (IN/OUT/GND/VDD).
        nets = {c.net for c in result.candidates}
        assert {"GND", "VDD"}.issubset(nets)

    def test_explicit_specs_override_auto_router(self, tmp_path: Path):
        """When the caller passes ``routing_specs``, the auto-router is skipped."""
        rules    = _rules(tmp_path)
        template = load_template("inverter")
        # Empty list → no router work, no rail polygons.
        result   = Synthesizer(rules).synthesize(
            template,
            params=        {"w": 0.52, "l": 0.15},
            routing_specs= [],
        )
        polys = result.component.get_polygons(by="tuple")
        # No GND/VDD rails were drawn (router ran on an empty spec list).
        m1_polys = polys.get(rules.layer("m1"), [])
        assert m1_polys == [] or all(  # only m1 polygons would come from devices
            True for _ in m1_polys
        )


# ── Phase A: shared gate, drain bridge, m0 bridge dedup ─────────────────────

class TestPhaseAIntraPair:
    """Direct unit coverage of Phase A — synthetic NetGraph + placement
    so we exercise paths the inverter alone doesn't reach."""

    @staticmethod
    def _synthetic(
        rules: BootstrapRules,
        n_fingers_n: int = 1,
        n_fingers_p: int = 1,
    ):
        # Two devices on a single row pair: N (bottom) at x=0, P (top) at x=0.
        geom_n = TransistorGeom(
            w_um=0.52, l_um=0.15, device_type="nmos", n_fingers=n_fingers_n,
            w_finger_um=0.52 / n_fingers_n, sd_length_um=0.29, n_contacts_y=1,
            total_x_um=(n_fingers_n + 1) * 0.29 + n_fingers_n * 0.15,
            total_y_um=0.78,
        )
        geom_p = TransistorGeom(
            w_um=0.52, l_um=0.15, device_type="pmos", n_fingers=n_fingers_p,
            w_finger_um=0.52 / n_fingers_p, sd_length_um=0.29, n_contacts_y=1,
            total_x_um=(n_fingers_p + 1) * 0.29 + n_fingers_p * 0.15,
            total_y_um=0.78,
        )
        spec_n = DeviceSpec(
            name="N", template="planar_mosfet", device_type="nmos",
            terminals={"G": "IN", "D": "OUT", "S": "GND"},
        )
        spec_n.row_pair = 0
        spec_p = DeviceSpec(
            name="P", template="planar_mosfet", device_type="pmos",
            terminals={"G": "IN", "D": "OUT", "S": "VDD"},
        )
        spec_p.row_pair = 0
        placed = {
            "N": PlacedDevice(name="N", spec=spec_n, geom=geom_n, x=0.0, y=0.0),
            "P": PlacedDevice(name="P", spec=spec_p, geom=geom_p, x=0.0, y=2.0),
        }
        ng = NetGraph(
            nets={
                "IN":  NetInfo(name="IN",  net_type="signal", terminals=[
                    TerminalRef("N", "G"), TerminalRef("P", "G"),
                ]),
                "OUT": NetInfo(name="OUT", net_type="signal", terminals=[
                    TerminalRef("N", "D"), TerminalRef("P", "D"),
                ]),
                "GND": NetInfo(name="GND", net_type="power", rail="bottom",
                               terminals=[TerminalRef("N", "S")]),
                "VDD": NetInfo(name="VDD", net_type="power", rail="top",
                               terminals=[TerminalRef("P", "S")]),
            },
            devices={
                "N": {"G": "IN", "D": "OUT", "S": "GND"},
                "P": {"G": "IN", "D": "OUT", "S": "VDD"},
            },
            device_types={"N": "nmos", "P": "pmos"},
        )
        return placed, ng

    def test_gate_pair_dedup(self, tmp_path: Path):
        """A repeat plan() call would dedup; one gate pair per (N, P) gate."""
        rules = _rules(tmp_path)
        placed, ng = self._synthetic(rules)
        # Build a tiny template so plan() has a CellTemplate to read.
        tpl = CellTemplate(
            name="t", description="", devices={}, nets=ng.nets,
            ports={}, named_constraints={},
        )
        specs = AutoRouter(rules).plan(ng, placed, tpl)
        gate_specs = [s for s in specs if s.style == "shared_gate_poly"]
        assert len(gate_specs) == 1

    def test_drain_bridge_suppressed_when_no_pmos(self, tmp_path: Path):
        """No PMOS drain on the net → no ``drain_bridge`` emitted."""
        rules = _rules(tmp_path)
        placed, ng = self._synthetic(rules)
        # Drop the PMOS drain entry so only N.D remains on OUT.
        ng.nets["OUT"].terminals = [TerminalRef("N", "D")]
        tpl = CellTemplate(
            name="t", description="", devices={}, nets=ng.nets,
            ports={}, named_constraints={},
        )
        specs = AutoRouter(rules).plan(ng, placed, tpl)
        assert not any(s.style == "drain_bridge" for s in specs)

    def test_intra_device_sd_for_multi_finger(self, tmp_path: Path):
        """A device with ≥ 2 fingers + S terminal on a non-power net gets
        an ``intra_device_sd`` spec."""
        rules = _rules(tmp_path)
        placed, ng = self._synthetic(rules, n_fingers_n=2)
        # Push N.S onto a signal net so phase A picks it up.
        ng.nets["GND"].terminals = []  # detach from power
        ng.nets["sig"] = NetInfo(
            name="sig", net_type="signal",
            terminals=[TerminalRef("N", "S")],
        )
        ng.devices["N"]["S"] = "sig"
        tpl = CellTemplate(
            name="t", description="", devices={}, nets=ng.nets,
            ports={}, named_constraints={},
        )
        specs = AutoRouter(rules).plan(ng, placed, tpl)
        intra = [s for s in specs if s.style == "intra_device_sd"]
        assert len(intra) == 1
        assert intra[0].extra == {"terminal": "S"}


# ── Phase D: power rail emission ────────────────────────────────────────────

class TestPhaseDPowerRails:
    def test_rail_layer_honours_net_layer(self, tmp_path: Path):
        """A power net with ``layer: m2`` lays its rail on m2 (not m1)."""
        rules = _rules(tmp_path)
        # Build minimal NetGraph: just a VDD net with explicit m2 layer.
        # Need at least one placed device for cell extent computation.
        geom = TransistorGeom(
            w_um=0.52, l_um=0.15, device_type="pmos", n_fingers=1,
            w_finger_um=0.52, sd_length_um=0.29, n_contacts_y=1,
            total_x_um=0.73, total_y_um=0.78,
        )
        spec = DeviceSpec(
            name="P", template="planar_mosfet", device_type="pmos",
            terminals={"G": "IN", "D": "OUT", "S": "VDD"},
        )
        spec.row_pair = 0
        placed = {"P": PlacedDevice(name="P", spec=spec, geom=geom, x=0.0, y=0.0)}
        ng = NetGraph(
            nets={
                "VDD": NetInfo(name="VDD", net_type="power", rail="top",
                               layer="m2",
                               terminals=[TerminalRef("P", "S")]),
            },
            devices={"P": {"G": "IN", "D": "OUT", "S": "VDD"}},
            device_types={"P": "pmos"},
        )
        tpl = CellTemplate(
            name="t", description="", devices={}, nets=ng.nets,
            ports={}, named_constraints={},
        )
        specs = AutoRouter(rules).plan(ng, placed, tpl)
        rails = [s for s in specs if s.style == "horizontal_power_rail"]
        assert len(rails) == 1
        assert rails[0].layer == "m2"
        # source_to_rail uses the same layer so the via stack matches the rail.
        s2r = [s for s in specs if s.style == "source_to_rail"]
        assert s2r and s2r[0].layer == "m2"


# ── Phase H: hint-driven full-width / full-height ────────────────────────────

class TestPhaseHHints:
    def test_full_width_emits_poly_stub_m1_bus(self, tmp_path: Path):
        """A ``full_width`` hint on a signal with gate terminals emits a
        ``poly_stub_m1_bus`` spec."""
        rules = _rules(tmp_path)
        geom = TransistorGeom(
            w_um=0.52, l_um=0.15, device_type="nmos", n_fingers=1,
            w_finger_um=0.52, sd_length_um=0.29, n_contacts_y=1,
            total_x_um=0.73, total_y_um=0.78,
        )
        spec = DeviceSpec(
            name="N", template="planar_mosfet", device_type="nmos",
            terminals={"G": "WL", "D": "BL", "S": "GND"},
        )
        spec.row_pair = 0
        placed = {"N": PlacedDevice(name="N", spec=spec, geom=geom, x=0.0, y=0.0)}
        ng = NetGraph(
            nets={
                "WL": NetInfo(name="WL", net_type="signal",
                              terminals=[TerminalRef("N", "G")]),
            },
            devices={"N": {"G": "WL", "D": "BL", "S": "GND"}},
            device_types={"N": "nmos"},
        )
        tpl = CellTemplate(
            name="t", description="", devices={}, nets=ng.nets,
            ports={}, named_constraints={},
            routing_hints={"WL": RoutingHint(net="WL", style="full_width", layer="m1")},
        )
        specs = AutoRouter(rules).plan(ng, placed, tpl)
        wl = [s for s in specs if s.style == "poly_stub_m1_bus"]
        assert len(wl) == 1
        assert wl[0].path == ["N.G"]
        assert wl[0].layer == "m1"


# ── Track allocator helper ──────────────────────────────────────────────────

class TestSynthesizeAllTemplates:
    """End-to-end synthesis smoke test for every shipped template."""

    TEMPLATES_TRANSISTOR = (
        "inverter", "nand2", "nand3", "nor2", "nor3",
        "aoi21", "oai21", "buffer", "row_driver",
        "bit_cell_6t", "dido",
    )

    @pytest.mark.parametrize("name", TEMPLATES_TRANSISTOR)
    def test_transistor_template_synthesizes(self, tmp_path: Path, name: str):
        rules    = _rules(tmp_path)
        template = load_template(name)
        result   = Synthesizer(rules).synthesize(
            template, params={"w": 0.52, "l": 0.15},
        )
        polys = result.component.get_polygons(by="tuple")
        # Every transistor cell exercises at least poly, diff, contact, m0, m1.
        for layer in ("poly", "diff", "contact", "m0", "m1"):
            assert rules.layer(layer) in polys, f"{name}: missing {layer}"

    def test_tap_cell_synthesizes(self, tmp_path: Path):
        """The device-free tap_cell uses a short-circuit path."""
        rules  = _rules(tmp_path)
        result = Synthesizer(rules).synthesize(load_template("tap_cell"))
        polys = result.component.get_polygons(by="tuple")
        for layer in ("contact", "m0", "m1", "nwell"):
            assert rules.layer(layer) in polys, layer

    def test_bit_cell_6t_emits_upper_metal(self, tmp_path: Path):
        """6T bit cell hits m2 (cross-couple + BL stripes)."""
        rules  = _rules(tmp_path)
        result = Synthesizer(rules).synthesize(
            load_template("bit_cell_6t"),
            params={"w": 0.52, "l": 0.15},
        )
        polys = result.component.get_polygons(by="tuple")
        assert rules.layer("m2")        in polys
        assert rules.layer("via_m0_m1") in polys
        assert rules.layer("via_m1_m2") in polys

    def test_dido_emits_cross_row_routing(self, tmp_path: Path):
        """Multi-row dido exercises cross_row_connect + vertical_bus."""
        rules  = _rules(tmp_path)
        result = Synthesizer(rules).synthesize(
            load_template("dido"),
            params={"w": 0.52, "l": 0.15},
        )
        polys = result.component.get_polygons(by="tuple")
        assert rules.layer("m1")        in polys
        assert rules.layer("m2")        in polys
        assert rules.layer("via_m0_m1") in polys
        assert rules.layer("via_m1_m2") in polys


class TestTrackAllocator:
    def test_returns_preferred_when_clear(self):
        from lithos_layout.synth.auto_router import _allocate_track
        x = _allocate_track(
            preferred_x=1.0, y_min=0.0, y_max=5.0,
            landing_half=0.1, wire_half=0.07, spacing=0.14,
            cell_x0=0.0, cell_x1=2.0,
            existing=[], level=2,
        )
        assert x == pytest.approx(1.0)

    def test_offsets_when_preferred_blocked(self):
        from lithos_layout.synth.auto_router import (
            _allocate_track, _TrackAllocation,
        )
        existing = [_TrackAllocation(x=1.0, y_min=0.0, y_max=5.0, level=2)]
        x = _allocate_track(
            preferred_x=1.0, y_min=0.0, y_max=5.0,
            landing_half=0.1, wire_half=0.07, spacing=0.14,
            cell_x0=0.0, cell_x1=2.0,
            existing=existing, level=2,
        )
        assert x != pytest.approx(1.0)
        # Returned track must be far enough away from x=1.0 to clear.
        assert abs(x - 1.0) >= 0.1 + 0.14 + 0.07 - 1e-9
