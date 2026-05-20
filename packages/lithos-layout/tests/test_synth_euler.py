"""Tests for ``lithos_layout.synth.euler`` — diffusion Euler-path ordering."""
from __future__ import annotations

from lithos_layout.synth.euler import (
    build_diffusion_graph,
    common_euler_order,
    euler_order,
    euler_path,
    has_euler_path,
)
from lithos_layout.synth.loader import CellTemplate, DeviceSpec, NetSpec


def _make_template(devices: dict[str, dict]) -> CellTemplate:
    """Tiny CellTemplate builder for testing."""
    dev_specs = {
        name: DeviceSpec(
            name        = name,
            template    = "planar_mosfet",
            device_type = d.get("type", "nmos"),
            terminals   = d.get("terminals", {}),
        )
        for name, d in devices.items()
    }
    nets: dict[str, NetSpec] = {}
    for dev in dev_specs.values():
        for net in dev.terminals.values():
            nets.setdefault(net, NetSpec(name=net, net_type="internal"))
    return CellTemplate(
        name              = "test",
        description       = "",
        devices           = dev_specs,
        nets              = nets,
        ports             = {},
        named_constraints = {},
    )


class TestBuildDiffusionGraph:
    def test_empty_template_empty_graph(self):
        tpl = _make_template({})
        g = build_diffusion_graph(tpl, "nmos")
        assert g.edges == []
        assert g.nodes == []

    def test_inverter_nmos_graph(self):
        tpl = _make_template({
            "N": {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
            "P": {"type": "pmos", "terminals": {"S": "VDD", "D": "OUT"}},
        })
        g = build_diffusion_graph(tpl, "nmos")
        assert len(g.edges) == 1
        assert g.edges[0].dev_name == "N"
        assert set(g.nodes) == {"GND", "OUT"}

    def test_device_type_filter(self):
        tpl = _make_template({
            "N":   {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
            "P":   {"type": "pmos", "terminals": {"S": "VDD", "D": "OUT"}},
        })
        g = build_diffusion_graph(tpl, "pmos")
        assert len(g.edges) == 1
        assert g.edges[0].dev_name == "P"


class TestHasEulerPath:
    def test_empty_graph_trivially_eulerian(self):
        tpl = _make_template({})
        assert has_euler_path(build_diffusion_graph(tpl, "nmos")) is True

    def test_nand2_nmos_series_chain_is_eulerian(self):
        # N_A: GND→net1, N_B: net1→OUT — two odd-degree nodes (GND, OUT).
        tpl = _make_template({
            "N_A": {"type": "nmos", "terminals": {"S": "GND",  "D": "net1"}},
            "N_B": {"type": "nmos", "terminals": {"S": "net1", "D": "OUT"}},
        })
        assert has_euler_path(build_diffusion_graph(tpl, "nmos")) is True


class TestEulerPath:
    def test_empty_returns_empty_list(self):
        tpl = _make_template({})
        assert euler_path(build_diffusion_graph(tpl, "nmos")) == []

    def test_nand2_series_chain_order(self):
        tpl = _make_template({
            "N_A": {"type": "nmos", "terminals": {"S": "GND",  "D": "net1"}},
            "N_B": {"type": "nmos", "terminals": {"S": "net1", "D": "OUT"}},
        })
        order = euler_path(build_diffusion_graph(tpl, "nmos"))
        # Either direction is valid — the path is GND-net1-OUT.
        assert order in (["N_A", "N_B"], ["N_B", "N_A"])

    def test_single_device(self):
        tpl = _make_template({
            "N": {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
        })
        order = euler_path(build_diffusion_graph(tpl, "nmos"))
        assert order == ["N"]


class TestCommonEulerOrder:
    def test_inverter_returns_two_devices(self):
        tpl = _make_template({
            "N": {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
            "P": {"type": "pmos", "terminals": {"S": "VDD", "D": "OUT"}},
        })
        order = common_euler_order(tpl)
        assert set(order) == {"N", "P"}

    def test_nmos_first_then_pmos(self):
        tpl = _make_template({
            "P_A": {"type": "pmos", "terminals": {"S": "VDD", "D": "OUT"}},
            "N_A": {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
        })
        order = common_euler_order(tpl)
        assert order is not None
        # NMOS devices precede PMOS devices in the merged list.
        nmos_indices = [i for i, name in enumerate(order) if name.startswith("N")]
        pmos_indices = [i for i, name in enumerate(order) if name.startswith("P")]
        assert max(nmos_indices) < min(pmos_indices)


class TestEulerOrderFallback:
    def test_no_devices_returns_empty(self):
        tpl = _make_template({})
        assert euler_order(tpl) == []

    def test_single_device(self):
        tpl = _make_template({
            "N": {"type": "nmos", "terminals": {"S": "GND", "D": "OUT"}},
        })
        assert euler_order(tpl) == ["N"]
