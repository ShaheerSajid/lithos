"""Tests for ``lithos_layout.synth.netlist`` — net graph from templates."""
from __future__ import annotations

from lithos_layout.synth.loader  import CellTemplate, DeviceSpec, NetSpec
from lithos_layout.synth.netlist import (
    NetGraph,
    NetInfo,
    TerminalRef,
    build_net_graph,
)


def _make_template(
    devices: dict[str, dict],
    declared_nets: dict[str, dict] | None = None,
) -> CellTemplate:
    declared_nets = declared_nets or {}
    return CellTemplate(
        name              = "test",
        description       = "",
        devices           = {
            name: DeviceSpec(
                name        = name,
                template    = "planar_mosfet",
                device_type = d.get("type", "nmos"),
                terminals   = d.get("terminals", {}),
            )
            for name, d in devices.items()
        },
        nets              = {
            name: NetSpec(name=name, **spec) for name, spec in declared_nets.items()
        },
        ports             = {},
        named_constraints = {},
    )


# ── TerminalRef ─────────────────────────────────────────────────────────────

class TestTerminalRef:
    def test_ref_property(self):
        t = TerminalRef(device="N", terminal="G")
        assert t.ref == "N.G"

    def test_repr(self):
        assert repr(TerminalRef(device="N_A", terminal="D")) == "N_A.D"


# ── NetInfo ─────────────────────────────────────────────────────────────────

class TestNetInfo:
    def test_gate_terminals_filter(self):
        info = NetInfo(
            name      = "IN",
            net_type  = "signal",
            terminals = [
                TerminalRef("N", "G"),
                TerminalRef("P", "G"),
                TerminalRef("N", "S"),
            ],
        )
        assert {t.device for t in info.gate_terminals} == {"N", "P"}

    def test_sd_terminals_filter(self):
        info = NetInfo(
            name      = "OUT",
            net_type  = "signal",
            terminals = [
                TerminalRef("N", "D"),
                TerminalRef("P", "D"),
                TerminalRef("N", "G"),
            ],
        )
        sd = info.sd_terminals
        assert len(sd) == 2
        assert all(t.terminal in ("S", "D") for t in sd)

    def test_is_power_and_internal(self):
        assert NetInfo(name="VDD", net_type="power").is_power
        assert NetInfo(name="x",   net_type="internal").is_internal
        assert not NetInfo(name="IN", net_type="signal").is_power


# ── build_net_graph ─────────────────────────────────────────────────────────

class TestBuildNetGraph:
    def test_inverter(self):
        tpl = _make_template(
            devices={
                "N": {"type": "nmos", "terminals": {"G": "IN", "D": "OUT", "S": "GND"}},
                "P": {"type": "pmos", "terminals": {"G": "IN", "D": "OUT", "S": "VDD"}},
            },
            declared_nets={
                "VDD": {"net_type": "power", "rail": "top"},
                "GND": {"net_type": "power", "rail": "bottom"},
                "IN":  {"net_type": "signal"},
                "OUT": {"net_type": "signal"},
            },
        )
        g = build_net_graph(tpl)
        assert isinstance(g, NetGraph)
        assert set(g.nets) == {"VDD", "GND", "IN", "OUT"}
        assert g.nets["IN"].net_type == "signal"
        assert g.nets["VDD"].is_power and g.nets["VDD"].rail == "top"
        # IN connects both gates.
        in_terms = g.terminals_on_net("IN")
        assert {t.device for t in in_terms} == {"N", "P"}
        assert {t.terminal for t in in_terms} == {"G"}
        # OUT connects both drains.
        out_terms = g.terminals_on_net("OUT")
        assert {t.terminal for t in out_terms} == {"D"}

    def test_undeclared_internal_net_auto_created(self):
        tpl = _make_template(
            devices={
                "N_A": {"type": "nmos", "terminals": {"S": "GND", "D": "midnet"}},
                "N_B": {"type": "nmos", "terminals": {"S": "midnet", "D": "OUT"}},
            },
            declared_nets={
                "GND": {"net_type": "power", "rail": "bottom"},
                "OUT": {"net_type": "signal"},
            },
        )
        g = build_net_graph(tpl)
        assert "midnet" in g.nets
        assert g.nets["midnet"].net_type == "internal"
        # Both N_A.D and N_B.S sit on midnet.
        terms = {(t.device, t.terminal) for t in g.terminals_on_net("midnet")}
        assert terms == {("N_A", "D"), ("N_B", "S")}

    def test_body_terminal_skipped(self):
        tpl = _make_template(
            devices={
                "N": {"type": "nmos", "terminals": {"G": "IN", "D": "OUT", "S": "GND", "B": "BULK"}},
            },
        )
        g = build_net_graph(tpl)
        # B terminal should NOT have created a BULK net or any TerminalRef.
        assert "BULK" not in g.nets
        for net in g.nets.values():
            assert all(t.terminal != "B" for t in net.terminals)

    def test_device_types_tracked(self):
        tpl = _make_template(
            devices={
                "N": {"type": "nmos", "terminals": {"G": "IN", "D": "OUT", "S": "GND"}},
                "P": {"type": "pmos", "terminals": {"G": "IN", "D": "OUT", "S": "VDD"}},
            },
        )
        g = build_net_graph(tpl)
        assert g.device_types == {"N": "nmos", "P": "pmos"}

    def test_nets_for_device(self):
        tpl = _make_template(
            devices={
                "N": {"type": "nmos", "terminals": {"G": "IN", "D": "OUT", "S": "GND"}},
            },
        )
        g = build_net_graph(tpl)
        assert g.nets_for_device("N") == {"G": "IN", "D": "OUT", "S": "GND"}
