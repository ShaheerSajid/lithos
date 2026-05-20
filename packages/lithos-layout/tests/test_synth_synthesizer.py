"""Tests for ``lithos_layout.synth.synthesizer`` — end-to-end pipeline."""
from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
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
from lithos_layout                 import BootstrapMapping, BootstrapRules
from lithos_layout.synth           import (
    SynthResult,
    Synthesizer,
    load_template,
)
from lithos_layout.synth.loader    import RoutingSpec
from lithos_layout.synth.synthesizer import _compute_skip_sd


# ── Shared rules ────────────────────────────────────────────────────────────

def _rules(tmp_path: Path) -> BootstrapRules:
    """A reasonably-complete BootstrapRules: enough for an inverter end-to-end."""
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
        "nwell.width_min_um":           "NW.W.1",
        "nwell.spacing_min_um":         "NW.S.1",
        "nwell.enclosure_of_pdiff_um":  "NW.E.D.1",
    })
    return BootstrapRules(md, db, mapping)


# ── _compute_skip_sd ────────────────────────────────────────────────────────

class TestComputeSkipSD:
    def test_no_directives_no_skips(self):
        from lithos_layout.synth.loader import CellTemplate
        t = CellTemplate(
            name="t", description="", devices={}, nets={},
            ports={}, named_constraints={},
        )
        assert _compute_skip_sd(t, {}) == {}


# ── Synthesizer.synthesize (inverter) ───────────────────────────────────────

class TestSynthesizeInverter:
    def _synth(self, tmp_path: Path,
               routing_specs: list[RoutingSpec] | None = None) -> SynthResult:
        rules = _rules(tmp_path)
        tpl   = load_template("inverter")
        synth = Synthesizer(rules)
        return synth.synthesize(
            tpl,
            params        = {"w": 0.52, "l": 0.15},
            routing_specs = routing_specs,
        )

    def test_returns_synthresult(self, tmp_path: Path):
        result = self._synth(tmp_path)
        assert isinstance(result, SynthResult)
        assert result.iterations == 1
        assert set(result.placed) == {"N", "P"}

    def test_component_is_a_gf_component(self, tmp_path: Path):
        result = self._synth(tmp_path)
        assert isinstance(result.component, gf.Component)

    def test_emits_transistor_layers(self, tmp_path: Path):
        result = self._synth(tmp_path)
        rules  = _rules(tmp_path)
        polys  = result.component.get_polygons(by="tuple")
        # Poly, diffusion, contact, m0 (S/D strips) must all appear.
        for layer in ("poly", "diff", "contact", "m0"):
            assert rules.layer(layer) in polys, layer

    def test_nwell_merged_for_pmos(self, tmp_path: Path):
        result = self._synth(tmp_path)
        rules  = _rules(tmp_path)
        polys  = result.component.get_polygons(by="tuple")
        # Single PMOS device → at least one N-well rectangle (from
        # draw_transistor, plus the merged-cluster pass when applicable).
        assert rules.layer("nwell") in polys

    def test_implants_per_row(self, tmp_path: Path):
        result = self._synth(tmp_path)
        rules  = _rules(tmp_path)
        polys  = result.component.get_polygons(by="tuple")
        # NMOS row → nimplant; PMOS row → pimplant.
        assert rules.layer("nimplant") in polys
        assert rules.layer("pimplant") in polys

    def test_routing_specs_executed(self, tmp_path: Path):
        """Hand-rolled routing specs produce additional polygons."""
        specs = [
            RoutingSpec(net="GND", style="horizontal_power_rail",
                        layer="m1", edge="bottom"),
            RoutingSpec(net="VDD", style="horizontal_power_rail",
                        layer="m1", edge="top"),
        ]
        result = self._synth(tmp_path, routing_specs=specs)
        rules  = _rules(tmp_path)
        polys  = result.component.get_polygons(by="tuple")
        # Routing emits two m1 strips on top of any device m1 (none here).
        m1_polys = polys.get(rules.layer("m1"), [])
        assert len(m1_polys) >= 2

    def test_candidates_returned(self, tmp_path: Path):
        specs = [
            RoutingSpec(net="GND", style="horizontal_power_rail",
                        layer="m1", edge="bottom"),
        ]
        result = self._synth(tmp_path, routing_specs=specs)
        nets = {c.net for c in result.candidates}
        assert "GND" in nets


# ── expose_terminal (router) ────────────────────────────────────────────────

class TestExposeTerminal:
    def test_emits_port_candidate(self, tmp_path: Path):
        """expose_terminal returns a candidate at the terminal centre."""
        from lithos_layout.synth import Router, PlacedDevice
        from lithos_layout.synth.loader      import DeviceSpec
        from lithos_layout.transistor        import TransistorGeom
        rules = _rules(tmp_path)
        dev = PlacedDevice(
            name="N",
            spec=DeviceSpec(name="N", template="planar_mosfet",
                            device_type="nmos",
                            terminals={"G": "IN", "D": "OUT", "S": "GND"}),
            geom=TransistorGeom(
                w_um=0.5, l_um=0.15, device_type="nmos", n_fingers=1,
                w_finger_um=0.5, sd_length_um=0.2, n_contacts_y=1,
                total_x_um=1.0, total_y_um=0.6,
            ),
            x=0.0, y=0.0,
        )
        spec = RoutingSpec(
            net="IN", style="expose_terminal", layer="poly",
            path=["N.G"], extra={"orientation": 180},
        )
        cands = Router(rules).route(gf.Component(), [spec], placed={"N": dev})
        assert len(cands) == 1
        c = cands[0]
        assert c.net == "IN"
        assert c.layer == "poly"
        assert c.orientation == 180
