"""Tests for ``lithos_layout.synth.port_resolver`` — pure-logic helpers."""
from __future__ import annotations

from lithos_layout.synth.loader        import (
    CellTemplate,
    DeviceSpec,
    NetSpec,
    PortSpec,
)
from lithos_layout.synth.netlist       import build_net_graph
from lithos_layout.synth.port_resolver import (
    PortCandidate,
    _best_candidate_for_side,
    _SIDE_ORIENTATION,
    generate_expose_specs,
)


def _template(ports: dict[str, PortSpec]) -> CellTemplate:
    return CellTemplate(
        name              = "t",
        description       = "",
        devices           = {
            "N": DeviceSpec(name="N", template="planar_mosfet", device_type="nmos",
                            terminals={"G": "IN", "D": "OUT", "S": "GND"}),
        },
        nets              = {
            "IN":  NetSpec(name="IN",  net_type="signal"),
            "OUT": NetSpec(name="OUT", net_type="signal"),
            "GND": NetSpec(name="GND", net_type="power", rail="bottom"),
        },
        ports             = ports,
        named_constraints = {},
    )


# ── Compass-side mapping ────────────────────────────────────────────────────

class TestSideOrientation:
    def test_compass_mappings(self):
        assert _SIDE_ORIENTATION["west"]  == 180
        assert _SIDE_ORIENTATION["east"]  == 0
        assert _SIDE_ORIENTATION["north"] == 90
        assert _SIDE_ORIENTATION["south"] == 270

    def test_aliases(self):
        assert _SIDE_ORIENTATION["left"]   == _SIDE_ORIENTATION["west"]
        assert _SIDE_ORIENTATION["right"]  == _SIDE_ORIENTATION["east"]
        assert _SIDE_ORIENTATION["top"]    == _SIDE_ORIENTATION["north"]
        assert _SIDE_ORIENTATION["bottom"] == _SIDE_ORIENTATION["south"]


# ── _best_candidate_for_side ────────────────────────────────────────────────

class TestBestCandidateForSide:
    def _make(self, x: float, y: float) -> PortCandidate:
        return PortCandidate(net="X", location_key="k", x=x, y=y,
                             layer="m1", width=0.17, orientation=0)

    def test_single_candidate_returned(self):
        c = self._make(1.0, 2.0)
        assert _best_candidate_for_side([c], "east", 0, 5, 0, 5) is c

    def test_picks_closest_to_east_edge(self):
        c0 = self._make(0.5, 1.0)
        c1 = self._make(4.8, 1.0)            # closest to right edge (x=5)
        assert _best_candidate_for_side([c0, c1], "east", 0, 5, 0, 5) is c1

    def test_picks_closest_to_west_edge(self):
        c0 = self._make(0.1, 1.0)            # closest to left edge (x=0)
        c1 = self._make(4.8, 1.0)
        assert _best_candidate_for_side([c0, c1], "west", 0, 5, 0, 5) is c0

    def test_picks_closest_to_north_edge(self):
        c0 = self._make(2.0, 0.2)
        c1 = self._make(2.0, 4.9)            # closest to top (y=5)
        assert _best_candidate_for_side([c0, c1], "north", 0, 5, 0, 5) is c1

    def test_picks_closest_to_south_edge(self):
        c0 = self._make(2.0, 0.2)            # closest to bottom (y=0)
        c1 = self._make(2.0, 4.9)
        assert _best_candidate_for_side([c0, c1], "south", 0, 5, 0, 5) is c0


# ── generate_expose_specs ───────────────────────────────────────────────────

class TestGenerateExposeSpecs:
    def test_no_ports_no_specs(self):
        tpl = _template(ports={})
        ng  = build_net_graph(tpl)
        assert generate_expose_specs(tpl, ng, placed={}) == []

    def test_terminal_port_emits_expose_spec(self):
        tpl = _template(ports={
            "IN": PortSpec(name="IN", side="west", terminal="N.G"),
        })
        ng  = build_net_graph(tpl)
        specs = generate_expose_specs(tpl, ng, placed={})
        assert len(specs) == 1
        s = specs[0]
        assert s.net   == "IN"
        assert s.style == "expose_terminal"
        assert s.layer == "m0"
        assert s.path  == ["N.G"]
        assert s.extra["orientation"]  == 180
        assert s.extra["location_key"] == "IN_port"

    def test_port_without_terminal_is_skipped(self):
        """Ports without a ``terminal:`` are resolved from candidates,
        not via expose specs, so the generator skips them."""
        tpl = _template(ports={
            "GND": PortSpec(name="GND", side="south"),     # no terminal
        })
        ng  = build_net_graph(tpl)
        assert generate_expose_specs(tpl, ng, placed={}) == []

    def test_orientation_follows_side(self):
        tpl = _template(ports={
            "p_north": PortSpec(name="p_north", side="north", terminal="N.G"),
            "p_south": PortSpec(name="p_south", side="south", terminal="N.D"),
        })
        ng  = build_net_graph(tpl)
        specs = {s.net: s for s in generate_expose_specs(tpl, ng, placed={})}
        assert specs["p_north"].extra["orientation"] == 90
        assert specs["p_south"].extra["orientation"] == 270
