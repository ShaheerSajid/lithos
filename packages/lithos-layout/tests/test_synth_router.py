"""Tests for ``lithos_layout.synth.router`` — registry + first style handler."""
from __future__ import annotations

import gdsfactory as gf
import pytest

from pathlib import Path

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
from lithos_layout              import BootstrapMapping, BootstrapRules
from lithos_layout.synth.loader import RoutingSpec
from lithos_layout.synth.placer import PlacedDevice
from lithos_layout.synth.router import (
    Router,
    _drawn_poly_contacts,
    _nudge_for_poly_spacing,
    _power_rail_gap,
    register_style,
    registered_styles,
)
from lithos_layout.transistor import TransistorGeom
from lithos_layout.synth.loader import DeviceSpec


# ── Shared minimal rules + placed devices ──────────────────────────────────

def _rules(tmp_path: Path, *, m0_collapsed: bool = False) -> BootstrapRules:
    db = RuleDB(tmp_path / "rules.db")
    db.open()
    db.set_pdk(name="t", version="0", ingested_at="2026-05-20T00:00:00Z")
    for code, check in [
        ("PO.W.1",   WidthCheck(target=LayerRef(name="poly"),
                                op=">=", threshold_um=0.15)),
        ("PO.E.1",   EnclosureCheck(inner=LayerRef(name="diff"),
                                    outer=LayerRef(name="poly"),
                                    op=">=", threshold_um=0.13)),
        ("DI.E.1",   EnclosureCheck(inner=LayerRef(name="diff"),
                                    outer=LayerRef(name="diff"),
                                    op=">=", threshold_um=0.25)),
        ("CO.W.1",   WidthCheck(target=LayerRef(name="contact"),
                                op=">=", threshold_um=0.17)),
        ("M0.W.1",   WidthCheck(target=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M0.S.1",   SpacingCheck(layer_a=LayerRef(name="m0"), op=">=", threshold_um=0.17)),
        ("M1.W.1",   WidthCheck(target=LayerRef(name="m1"), op=">=", threshold_um=0.20)),
        ("M1.S.1",   SpacingCheck(layer_a=LayerRef(name="m1"), op=">=", threshold_um=0.20)),
    ]:
        db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    m0_layer = (67, 20)
    m1_layer = m0_layer if m0_collapsed else (68, 20)
    md = PDKMetadata(
        name="t", version="0",
        layers={"poly": (66, 20), "diff": (65, 20),
                "contact": (66, 44), "m0": m0_layer, "m1": m1_layer},
        grid={"manufacturing_um": 0.005},
        drc_decks={},
        devices={},
    )
    mapping = BootstrapMapping(mapping={
        "poly.width_min_um":            "PO.W.1",
        "poly.endcap_over_diff_um":     "PO.E.1",
        "diff.extension_past_poly_um":  "DI.E.1",
        "contact.size_um":              "CO.W.1",
        "m0.width_min_um":              "M0.W.1",
        "m0.spacing_min_um":            "M0.S.1",
        "m1.width_min_um":              "M1.W.1",
        "m1.spacing_min_um":            "M1.S.1",
    })
    return BootstrapRules(md, db, mapping)


def _placed_device(x: float = 0.0, y: float = 0.0,
                   total_x: float = 1.0, total_y: float = 0.6,
                   name: str = "N", dev_type: str = "nmos") -> PlacedDevice:
    return PlacedDevice(
        name = name,
        spec = DeviceSpec(name=name, template="planar_mosfet",
                          device_type=dev_type, terminals={}),
        geom = TransistorGeom(
            w_um=0.5, l_um=0.15, device_type=dev_type, n_fingers=1,
            w_finger_um=0.5, sd_length_um=0.2, n_contacts_y=1,
            total_x_um=total_x, total_y_um=total_y,
        ),
        x = x, y = y,
    )


# ── Registry ───────────────────────────────────────────────────────────────

class TestRegistry:
    def test_horizontal_power_rail_registered(self):
        assert "horizontal_power_rail" in registered_styles()

    def test_register_then_lookup(self, tmp_path: Path):
        captured: list[str] = []
        def _h(comp, spec, placed, rules):
            captured.append(spec.net)
            return []
        register_style("__test_style", _h)
        try:
            r = _rules(tmp_path)
            placed = {"N": _placed_device()}
            spec = RoutingSpec(net="X", style="__test_style", layer="m0")
            Router(r).route(gf.Component(), [spec], placed)
            assert captured == ["X"]
        finally:
            from lithos_layout.synth.router import _REGISTRY
            _REGISTRY.pop("__test_style", None)

    def test_unknown_style_warns_and_skips(self, tmp_path: Path):
        r = _rules(tmp_path)
        placed = {"N": _placed_device()}
        spec = RoutingSpec(net="Y", style="not_a_real_style")
        with pytest.warns(UserWarning, match="No handler registered"):
            Router(r).route(gf.Component(), [spec], placed)


# ── Helpers ─────────────────────────────────────────────────────────────────

class TestPowerRailGap:
    def test_zero_when_m0_distinct_from_m1(self, tmp_path: Path):
        r = _rules(tmp_path, m0_collapsed=False)
        assert _power_rail_gap(r) == pytest.approx(0.0)

    def test_positive_when_m0_collapsed(self, tmp_path: Path):
        r = _rules(tmp_path, m0_collapsed=True)
        # Gap = max(0, m1_spacing - poly_endcap) + 10 nm.
        # m1.spacing = 0.20, poly.endcap = 0.13 → 0.07 + 0.01 = 0.08.
        assert _power_rail_gap(r) == pytest.approx(0.08, abs=1e-6)


class TestNudgeForPolySpacing:
    def test_no_neighbour_returns_unchanged(self):
        cx = _nudge_for_poly_spacing(
            cx=0.0, pad_half_x=0.05,
            own_gate_range=(-0.02, 0.02),
            all_gate_ranges=[(-0.02, 0.02, "self")],
            poly_sp=0.21,
        )
        assert cx == pytest.approx(0.0)

    def test_shifts_away_from_right_neighbour(self):
        cx = _nudge_for_poly_spacing(
            cx=0.10, pad_half_x=0.05,
            own_gate_range=(-0.02, 0.02),
            all_gate_ranges=[(-0.02, 0.02, "self"), (0.20, 0.30, "other")],
            poly_sp=0.21,
        )
        # Pad right edge would be 0.15 vs neighbour at 0.20 (gap=0.05);
        # required spacing 0.21 means pad shifts LEFT by 0.16 + eps.
        assert cx < 0.10


# ── horizontal_power_rail end-to-end ───────────────────────────────────────

class TestHorizontalPowerRail:
    def test_bottom_rail_writes_rect_and_returns_candidate(self, tmp_path: Path):
        r = _rules(tmp_path)
        placed = {"N": _placed_device(x=0.0, y=0.0, total_x=2.0, total_y=0.6)}
        comp   = gf.Component()
        spec   = RoutingSpec(net="GND", style="horizontal_power_rail",
                             layer="m1", edge="bottom")
        cands  = Router(r).route(comp, [spec], placed)
        assert len(cands) == 1
        c = cands[0]
        assert c.net == "GND"
        assert c.location_key == "bottom_rail_center"
        assert c.orientation == 270
        # Rail is below the device (y < 0).
        assert c.y < 0
        # GDS polygons in the m1 layer.
        m1_layer = r.layer("m1")
        polys = comp.get_polygons(by="tuple")
        assert m1_layer in polys
        assert len(polys[m1_layer]) >= 1

    def test_top_rail_above_devices(self, tmp_path: Path):
        r = _rules(tmp_path)
        placed = {"N": _placed_device(x=0.0, y=0.0, total_x=2.0, total_y=0.6)}
        spec = RoutingSpec(net="VDD", style="horizontal_power_rail",
                           layer="m1", edge="top")
        [c] = Router(r).route(gf.Component(), [spec], placed)
        assert c.location_key == "top_rail_center"
        assert c.orientation == 90
        assert c.y > 0.6

    def test_intermediate_rail_at_explicit_y_pos(self, tmp_path: Path):
        r = _rules(tmp_path)
        placed = {"N": _placed_device(x=0.0, y=0.0, total_x=2.0, total_y=0.6)}
        spec = RoutingSpec(
            net="VSS", style="horizontal_power_rail", layer="m1",
            extra={"y_pos": 5.0},
        )
        [c] = Router(r).route(gf.Component(), [spec], placed)
        assert "rail_VSS_5.000" in c.location_key
        assert c.y == pytest.approx(5.0, abs=0.2)
        assert c.orientation == 90

    def test_route_clears_drawn_poly_contacts(self, tmp_path: Path):
        """Router.route should reset the per-call ``_drawn_poly_contacts``
        cache so a fresh pass doesn't see state from a previous one."""
        _drawn_poly_contacts[(0.0, "N")] = (1.0, 1.0)
        r = _rules(tmp_path)
        placed = {"N": _placed_device()}
        Router(r).route(gf.Component(), [], placed)
        assert _drawn_poly_contacts == {}


# ── shared_gate_poly ────────────────────────────────────────────────────────

def _paired_devices(tmp_path: Path) -> tuple[BootstrapRules, dict[str, PlacedDevice]]:
    """Build a minimal rules + N (bottom) / P (top) placement."""
    rules = _rules(tmp_path)
    # NMOS at (0, 0), PMOS stacked above with a 0.2 µm Y gap.
    n = _placed_device(name="N", dev_type="nmos", x=0.0, y=0.0,
                       total_x=1.0, total_y=0.6)
    p = _placed_device(name="P", dev_type="pmos", x=0.0, y=0.8,
                       total_x=1.0, total_y=0.6)
    return rules, {"N": n, "P": p}


class TestSharedGatePoly:
    def test_emits_two_port_candidates(self, tmp_path: Path):
        rules, placed = _paired_devices(tmp_path)
        spec = RoutingSpec(net="IN", style="shared_gate_poly",
                           layer="poly", path=["N.G", "P.G"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert len(cands) == 2
        assert {c.location_key for c in cands} == {
            "IN_gate_left_edge_mid_y", "gate_left_edge_mid_y",
        }
        for c in cands:
            assert c.net == "IN"
            assert c.layer == "poly"
            assert c.orientation == 180

    def test_writes_poly_polygons(self, tmp_path: Path):
        rules, placed = _paired_devices(tmp_path)
        comp = gf.Component()
        spec = RoutingSpec(net="IN", style="shared_gate_poly",
                           layer="poly", path=["N.G", "P.G"])
        Router(rules).route(comp, [spec], placed)
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("poly") in polys, polys.keys()

    def test_too_short_path_no_op(self, tmp_path: Path):
        rules, placed = _paired_devices(tmp_path)
        spec = RoutingSpec(net="X", style="shared_gate_poly",
                           layer="poly", path=["N.G"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert cands == []


# ── intra_device_sd ─────────────────────────────────────────────────────────

class TestIntraDeviceSD:
    def _multifinger(self, tmp_path: Path) -> tuple[BootstrapRules,
                                                    dict[str, PlacedDevice]]:
        rules = _rules(tmp_path)
        # 3-finger device: S/D indices 0, 1, 2, 3 → S at 0,2 ; D at 1,3.
        dev = PlacedDevice(
            name = "N",
            spec = DeviceSpec(name="N", template="planar_mosfet",
                              device_type="nmos", terminals={}),
            geom = TransistorGeom(
                w_um=1.5, l_um=0.15, device_type="nmos", n_fingers=3,
                w_finger_um=0.5, sd_length_um=0.2, n_contacts_y=1,
                total_x_um=1.05, total_y_um=0.76,
            ),
            x = 0.0, y = 0.0,
        )
        return rules, {"N": dev}

    def test_drain_strap_drawn(self, tmp_path: Path):
        rules, placed = self._multifinger(tmp_path)
        comp = gf.Component()
        spec = RoutingSpec(
            net="OUT", style="intra_device_sd", layer="m0",
            path=["N.D"], extra={"terminal": "D"},
        )
        Router(rules).route(comp, [spec], placed)
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("m0") in polys

    def test_single_strip_no_op(self, tmp_path: Path):
        rules = _rules(tmp_path)
        # 1-finger device: only one S and one D strip — no strap needed.
        placed = {"N": _placed_device(name="N", total_x=1.0, total_y=0.6)}
        comp = gf.Component()
        spec = RoutingSpec(
            net="OUT", style="intra_device_sd", layer="m0",
            path=["N.D"], extra={"terminal": "D"},
        )
        Router(rules).route(comp, [spec], placed)
        assert rules.layer("m0") not in comp.get_polygons(by="tuple")

    def test_empty_path_no_op(self, tmp_path: Path):
        rules = _rules(tmp_path)
        placed = {"N": _placed_device(name="N", total_x=1.0, total_y=0.6)}
        spec = RoutingSpec(net="X", style="intra_device_sd",
                           layer="m0", path=[])
        Router(rules).route(gf.Component(), [spec], placed)


# ── m0_bridge ───────────────────────────────────────────────────────────────

class TestM0Bridge:
    def _two_devices(self, tmp_path: Path) -> tuple[BootstrapRules,
                                                    dict[str, PlacedDevice]]:
        rules = _rules(tmp_path)
        a = _placed_device(name="A", x=0.0, y=0.0, total_x=1.0, total_y=0.6)
        b = _placed_device(name="B", x=2.0, y=0.0, total_x=1.0, total_y=0.6)
        return rules, {"A": a, "B": b}

    def test_bridge_emits_candidate_and_polygon(self, tmp_path: Path):
        rules, placed = self._two_devices(tmp_path)
        comp = gf.Component()
        spec = RoutingSpec(net="Q", style="m0_bridge", layer="m0",
                           path=["A.D", "B.S"])
        [c] = Router(rules).route(comp, [spec], placed)
        assert c.net == "Q"
        assert c.layer == "m0"
        assert c.location_key == "Q_bridge_center"
        assert rules.layer("m0") in comp.get_polygons(by="tuple")

    def test_too_short_path_no_op(self, tmp_path: Path):
        rules, placed = self._two_devices(tmp_path)
        spec = RoutingSpec(net="Q", style="m0_bridge", layer="m0",
                           path=["A.D"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert cands == []


# ── drain_bridge ────────────────────────────────────────────────────────────

class TestDrainBridge:
    def test_inverter_like_bridge_writes_polygons(self, tmp_path: Path):
        """N drain + P drain bridged across the N-P gap on m0."""
        rules, placed = _paired_devices(tmp_path)
        comp = gf.Component()
        spec = RoutingSpec(net="OUT", style="drain_bridge", layer="m0",
                           path=["N.D", "P.D"])
        [c] = Router(rules).route(comp, [spec], placed)
        assert c.net == "OUT"
        assert c.location_key == "drain_bridge_right_edge_mid_y"
        assert c.orientation == 0
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("m0") in polys

    def test_bridge_on_m1_drops_via_stacks(self, tmp_path: Path):
        """When the bus rides m1, the handler must also emit m0/contact/via
        polygons via draw_via_stack at every drain."""
        # Build a richer rules set with via_m0_m1 mapping so draw_via_stack
        # can actually emit cuts.
        rules = _rules_with_via(tmp_path)
        n = _placed_device(name="N", dev_type="nmos", x=0.0, y=0.0,
                           total_x=1.0, total_y=0.6)
        p = _placed_device(name="P", dev_type="pmos", x=0.0, y=0.8,
                           total_x=1.0, total_y=0.6)
        comp = gf.Component()
        spec = RoutingSpec(net="OUT", style="drain_bridge", layer="m1",
                           path=["N.D", "P.D"])
        Router(rules).route(comp, [spec], placed={"N": n, "P": p})
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("m1") in polys
        assert rules.layer("via_m0_m1") in polys

    def test_too_short_path_no_op(self, tmp_path: Path):
        rules, placed = _paired_devices(tmp_path)
        spec = RoutingSpec(net="OUT", style="drain_bridge", layer="m0",
                           path=["N.D"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert cands == []


# ── source_to_rail ──────────────────────────────────────────────────────────

class TestSourceToRail:
    def test_bottom_rail_m0_only_strap(self, tmp_path: Path):
        """rail_layer == m0 → single m0 strap from S edge down to rail."""
        rules = _rules(tmp_path)
        placed = {"N": _placed_device(name="N", x=0.0, y=0.0,
                                       total_x=2.0, total_y=0.6)}
        comp = gf.Component()
        spec = RoutingSpec(net="GND", style="source_to_rail", layer="m0",
                           edge="bottom", path=["N.S"])
        Router(rules).route(comp, [spec], placed)
        assert rules.layer("m0") in comp.get_polygons(by="tuple")

    def test_top_rail_m1_drops_via_stack(self, tmp_path: Path):
        """rail_layer == m1 → via stack at each strip + m1 strap to rail."""
        rules = _rules_with_via(tmp_path)
        placed = {"P": _placed_device(name="P", dev_type="pmos",
                                       x=0.0, y=0.0,
                                       total_x=2.0, total_y=0.6)}
        comp = gf.Component()
        spec = RoutingSpec(net="VDD", style="source_to_rail", layer="m1",
                           edge="top", path=["P.S"])
        Router(rules).route(comp, [spec], placed)
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("m1") in polys
        assert rules.layer("via_m0_m1") in polys

    def test_unknown_device_skipped(self, tmp_path: Path):
        rules = _rules(tmp_path)
        placed = {"N": _placed_device(name="N")}
        spec = RoutingSpec(net="GND", style="source_to_rail", layer="m0",
                           edge="bottom", path=["BOGUS.S"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert cands == []


# ── gate_to_drain ───────────────────────────────────────────────────────────

class TestGateToDrain:
    """Same-row gate→drain through the N-P gap (AOI/OAI stage chaining)."""

    def _two_devices_same_row(
        self, tmp_path: Path, dev_type: str = "nmos",
    ) -> tuple[BootstrapRules, dict[str, PlacedDevice]]:
        rules = _rules(tmp_path)
        # A: gate device at x=0. B: drain device at x=2.0 (to the right of A).
        a = _placed_device(name="A", dev_type=dev_type,
                           x=0.0, y=0.0, total_x=1.0, total_y=0.6)
        b = _placed_device(name="B", dev_type=dev_type,
                           x=2.0, y=0.0, total_x=1.0, total_y=0.6)
        return rules, {"A": a, "B": b}

    def test_m0_route_emits_poly_contact_and_m0_trunk(self, tmp_path: Path):
        """Default ``spec.layer == m0`` → contact + poly pad + m0 trunk."""
        rules, placed = self._two_devices_same_row(tmp_path)
        _drawn_poly_contacts.clear()
        comp = gf.Component()
        spec = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                           path=["A.G", "B.D"])
        cands = Router(rules).route(comp, [spec], placed)
        assert cands == []   # handler emits no port candidates
        polys = comp.get_polygons(by="tuple")
        # Poly contact pieces + m0 trunk are all on these three layers.
        assert rules.layer("contact") in polys
        assert rules.layer("poly")    in polys
        assert rules.layer("m0")      in polys

    def test_drain_left_of_gate_routes_leftward(self, tmp_path: Path):
        """When drain X < gate X, the m0 trunk extends to the left."""
        rules = _rules(tmp_path)
        # Swap positions: B (drain) at x=0, A (gate) at x=2.0.
        a = _placed_device(name="A", x=2.0, y=0.0, total_x=1.0, total_y=0.6)
        b = _placed_device(name="B", x=0.0, y=0.0, total_x=1.0, total_y=0.6)
        _drawn_poly_contacts.clear()
        comp = gf.Component()
        spec = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                           path=["A.G", "B.D"])
        Router(rules).route(comp, [spec], {"A": a, "B": b})
        # m0 trunk must extend from gate (around x=2.5) to drain (around x=0.2).
        polys = comp.get_polygons(by="tuple").get(rules.layer("m0"), [])
        xs = [pt.x / 1000.0 for poly in polys for pt in poly.each_point_hull()]
        assert xs and min(xs) < 0.5 and max(xs) > 2.0

    def test_pmos_gate_routes_downward(self, tmp_path: Path):
        """PMOS gate: poly contact + trunk sit *below* the device."""
        rules, placed = self._two_devices_same_row(tmp_path, dev_type="pmos")
        # Push PMOS up so the trunk has room below.
        placed["A"].y = 2.0
        placed["B"].y = 2.0
        _drawn_poly_contacts.clear()
        comp = gf.Component()
        spec = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                           path=["A.G", "B.D"])
        Router(rules).route(comp, [spec], placed)
        polys = comp.get_polygons(by="tuple").get(rules.layer("m0"), [])
        # Trunk Y must be below the PMOS poly bottom (y=2.0).
        ys = [pt.y / 1000.0 for poly in polys for pt in poly.each_point_hull()]
        assert ys and min(ys) < 2.0

    def test_m1_route_drops_via_stacks(self, tmp_path: Path):
        """``spec.layer == m1`` → handler emits via stacks at both ends."""
        rules = _rules_with_via(tmp_path)
        a = _placed_device(name="A", x=0.0, y=0.0, total_x=1.0, total_y=0.6)
        b = _placed_device(name="B", x=2.0, y=0.0, total_x=1.0, total_y=0.6)
        _drawn_poly_contacts.clear()
        comp = gf.Component()
        spec = RoutingSpec(net="X", style="gate_to_drain", layer="m1",
                           path=["A.G", "B.D"])
        Router(rules).route(comp, [spec], {"A": a, "B": b})
        polys = comp.get_polygons(by="tuple")
        assert rules.layer("m1")        in polys
        assert rules.layer("via_m0_m1") in polys

    def test_shared_contact_dedup(self, tmp_path: Path):
        """Two routes hitting the same gate reuse the existing poly contact."""
        rules, placed = self._two_devices_same_row(tmp_path)
        _drawn_poly_contacts.clear()
        comp = gf.Component()
        spec1 = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                            path=["A.G", "B.D"])
        spec2 = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                            path=["A.G", "B.D"])
        Router(rules).route(comp, [spec1, spec2], placed)
        # _drawn_poly_contacts must hold exactly one entry for A's gate.
        keyed = [k for k in _drawn_poly_contacts if k[1] == "poly_contact"]
        assert len(keyed) == 1

    def test_too_short_path_no_op(self, tmp_path: Path):
        rules, placed = self._two_devices_same_row(tmp_path)
        spec = RoutingSpec(net="X", style="gate_to_drain", layer="m0",
                           path=["A.G"])
        cands = Router(rules).route(gf.Component(), [spec], placed)
        assert cands == []


def _rules_with_via(tmp_path: Path) -> BootstrapRules:
    """Variant of ``_rules`` with via_m0_m1 keys so draw_via_stack works."""
    rules = _rules(tmp_path)
    # Add the missing via keys post-hoc into the bootstrap mapping.
    rules._mapping.mapping.update({
        "via_m0_m1.size_um":                  "V01.W.1",
        "m0.enclosure_of_via_m0_m1_2adj_um":  "M0.E.V01.2ADJ",
        "m1.enclosure_of_via_m0_m1_2adj_um":  "M1.E.V01.2ADJ",
    })
    # Seed the matching rules in the DB.
    for code, check in [
        ("V01.W.1",       WidthCheck(target=LayerRef(name="via_m0_m1"),
                                     op=">=", threshold_um=0.17)),
        ("M0.E.V01.2ADJ", EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                         outer=LayerRef(name="m0"),
                                         op=">=", threshold_um=0.06)),
        ("M1.E.V01.2ADJ", EnclosureCheck(inner=LayerRef(name="via_m0_m1"),
                                         outer=LayerRef(name="m1"),
                                         op=">=", threshold_um=0.06)),
    ]:
        rules.db.upsert_rule(Rule(
            code=code, category="x", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    # Also expose via_m0_m1 in the metadata layer table.
    rules.metadata.layers["via_m0_m1"] = (67, 44)
    # Bust the get() cache so the new keys resolve.
    rules._cache.clear()
    return rules
