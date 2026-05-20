"""Tests for the shipped cell templates under ``templates/cells/``.

Each template ships in the package and must:
  1. Load through ``load_template`` without raising.
  2. Resolve by bare name (e.g. ``"inverter"``).
  3. Carry no sky130-style layer names (``li1``, ``met1``, ``met2``,
     ``mcon``, ``licon1``, ``via1``) on any field a downstream consumer
     would read.

Plus a handful of cell-specific spot checks for layout-affecting fields.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_layout import CellTemplate, load_template


CELL_NAMES = [
    "inverter",
    "nand2",
    "nand3",
    "nor2",
    "nor3",
    "aoi21",
    "oai21",
    "buffer",
    "bit_cell_6t",
    "dido",
    "row_driver",
    "tap_cell",
]


FORBIDDEN_LAYER_TOKENS = {"li1", "met1", "met2", "mcon", "licon1", "via1"}


# ── Smoke: every template resolves and parses ────────────────────────────────

class TestTemplatesLoad:
    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_load_by_name(self, cell: str):
        tpl = load_template(cell)
        assert isinstance(tpl, CellTemplate)
        assert tpl.name
        # tap_cell ships device-less; every other template names devices.
        if cell != "tap_cell":
            assert tpl.devices, f"{cell} has no devices"

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_source_path_is_shipped_template(self, cell: str):
        tpl = load_template(cell)
        assert tpl.source_path is not None
        assert tpl.source_path.name == f"{cell}.yaml"
        # Must come from the in-package templates dir.
        assert "packages/lithos-layout/templates/cells" in str(tpl.source_path)


# ── No sky130-flavoured layer names anywhere ─────────────────────────────────

class TestNoSky130LayerNames:
    """No port, net, or routing-hint layer field in any shipped template
    may carry an sky130-style layer token. ``_normalize_layer`` lowercases
    unknown names but does not rewrite them, so this catches drift back
    to legacy names.
    """

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_port_layers_canonical(self, cell: str):
        tpl = load_template(cell)
        for port in tpl.ports.values():
            assert port.layer not in FORBIDDEN_LAYER_TOKENS, (
                f"{cell}: port {port.name!r} layer={port.layer!r}"
            )

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_net_layers_canonical(self, cell: str):
        tpl = load_template(cell)
        for net in tpl.nets.values():
            assert net.layer not in FORBIDDEN_LAYER_TOKENS, (
                f"{cell}: net {net.name!r} layer={net.layer!r}"
            )

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_routing_hint_layers_canonical(self, cell: str):
        tpl = load_template(cell)
        for hint in tpl.routing_hints.values():
            assert hint.layer not in FORBIDDEN_LAYER_TOKENS, (
                f"{cell}: routing hint {hint.net!r} layer={hint.layer!r}"
            )

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_label_layers_unset(self, cell: str):
        """Shipped templates must not carry sky130 GDS values for
        label datatypes; those come from PDK metadata at render time.
        """
        tpl = load_template(cell)
        assert tpl.label_layers.m1 is None, f"{cell} hard-codes m1 label tuple"
        assert tpl.label_layers.m2 is None, f"{cell} hard-codes m2 label tuple"


# ── Spot checks per cell ─────────────────────────────────────────────────────

class TestInverter:
    def test_topology(self):
        tpl = load_template("inverter")
        assert tpl.name == "cmos_inverter"
        assert set(tpl.devices) == {"N", "P"}
        assert tpl.devices["N"].device_type == "nmos"
        assert tpl.devices["P"].device_type == "pmos"
        assert tpl.ports["GND"].layer == "m1"
        assert tpl.ports["VDD"].layer == "m1"
        # Placement uses ``placement_logic:`` (directives form) — verify
        # both devices got a directive.
        names = {d.name for d in tpl.placement_directives}
        assert names == {"N", "P"}


class TestNand2:
    def test_orientation_mirror_on_pb(self):
        tpl = load_template("nand2")
        directives = {d.name: d for d in tpl.placement_directives}
        assert directives["P_B"].orientation == "MY"

    def test_internal_net_present(self):
        tpl = load_template("nand2")
        assert "net1" in tpl.nets
        assert tpl.nets["net1"].net_type == "internal"


class TestNor2:
    def test_sd_flip_on_nb(self):
        tpl = load_template("nor2")
        directives = {d.name: d for d in tpl.placement_directives}
        assert directives["N_B"].sd_flip is True


class TestBitCell6T:
    def test_routing_hints_canonical_layers(self):
        tpl = load_template("bit_cell_6t")
        assert tpl.routing_hints["WL"].layer == "m1"
        assert tpl.routing_hints["BL"].layer == "m2"
        assert tpl.routing_hints["BL_"].layer == "m2"

    def test_cell_dimensions_and_abutment(self):
        tpl = load_template("bit_cell_6t")
        assert tpl.cell_dimensions.width == pytest.approx(3.600)
        assert tpl.abutment.pitch_x == pytest.approx(3.600)
        assert tpl.abutment.mirror_x is True

    def test_local_strategy_on_internal_nets(self):
        tpl = load_template("bit_cell_6t")
        assert tpl.routing_hints["Q"].strategy == "local"
        assert tpl.routing_hints["Q_"].strategy == "local"


class TestDido:
    def test_stacked_layout(self):
        tpl = load_template("dido")
        assert tpl.layout_mode == "stacked"
        assert len(tpl.row_pairs) == 10

    def test_param_overrides_picked_up(self):
        tpl = load_template("dido")
        assert tpl.device_params["N_WP0"]["w"] == pytest.approx(1.0)
        assert tpl.device_params["N_WP1"]["w"] == pytest.approx(1.0)

    def test_canonical_port_layers(self):
        tpl = load_template("dido")
        assert tpl.ports["BL"].layer == "m2"
        assert tpl.ports["VDD"].layer == "m1"


class TestRowDriver:
    def test_finger_overrides(self):
        tpl = load_template("row_driver")
        assert tpl.devices["N_INV"].fingers == 5
        assert tpl.devices["P_INV"].fingers == 5
        assert tpl.devices["P_INV"].w == pytest.approx(2.60)

    def test_net_layer_canonical(self):
        tpl = load_template("row_driver")
        assert tpl.nets["VDD"].layer == "m1"
        assert tpl.nets["Y_nd"].layer == "m1"


class TestTapCell:
    def test_loads_and_has_power_ports(self):
        tpl = load_template("tap_cell")
        assert "VDD" in tpl.ports
        assert "GND" in tpl.ports

    def test_no_sky130_stack_names_in_raw_yaml(self):
        """The tap_cell ``stack`` block is not parsed by the loader, so we
        guard it directly: the raw file must reference canonical layer
        names only.
        """
        path = Path(__file__).resolve().parents[1] / "templates" / "cells" / "tap_cell.yaml"
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_LAYER_TOKENS:
            assert token not in text, f"tap_cell.yaml references {token!r}"
