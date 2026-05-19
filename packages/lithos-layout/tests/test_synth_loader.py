"""lithos_layout.synth.loader — topology YAML → typed dataclasses."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lithos_layout import (
    AbutmentSpec,
    CellDimensions,
    CellTemplate,
    DeviceSpec,
    LabelLayerSpec,
    NetSpec,
    PlacementDirective,
    PortSpec,
    RoutingHint,
    RowPairSpec,
    load_template,
)
from lithos_layout.synth.loader import _normalize_layer


# ── Helpers ────────────────────────────────────────────────────────────────

def _write(tmp_path: Path, raw: dict, name: str = "tpl.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


# ── _normalize_layer ───────────────────────────────────────────────────────

class TestNormalizeLayer:
    @pytest.mark.parametrize("raw,canonical", [
        ("M0",  "m0"),
        ("M1",  "m1"),
        ("M2",  "m2"),
        ("m0",  "m0"),
        ("m12", "m12"),
        ("",    ""),
    ])
    def test_metal_shorthand(self, raw: str, canonical: str):
        assert _normalize_layer(raw) == canonical

    def test_passes_through_non_metal_names_lowercased(self):
        assert _normalize_layer("Contact")    == "contact"
        assert _normalize_layer("via_m0_m1")  == "via_m0_m1"
        assert _normalize_layer("Poly")       == "poly"

    def test_strips_whitespace(self):
        assert _normalize_layer("  M1  ") == "m1"


# ── Devices ────────────────────────────────────────────────────────────────

def test_devices_parsed_with_defaults(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "IN", "D": "OUT", "S": "VSS"}},
            "P1": {"type": "pmos", "terminals": {"G": "IN", "D": "OUT", "S": "VDD"},
                   "w": 0.6, "l": 0.15, "fingers": 2},
        },
    })
    tpl = load_template(p)
    assert tpl.devices["N1"].device_type == "nmos"
    assert tpl.devices["N1"].template    == "planar_mosfet"  # default
    assert tpl.devices["N1"].fingers     == 0                 # default = auto
    assert tpl.devices["P1"].fingers     == 2
    assert tpl.devices["P1"].w           == pytest.approx(0.6)
    assert tpl.devices["P1"].l           == pytest.approx(0.15)


def test_device_w_and_l_merge_into_device_params(tmp_path: Path):
    """Per-device w/l on a DeviceSpec also surfaces in CellTemplate.device_params."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "IN"}, "w": 0.42},
            "N2": {"type": "nmos", "terminals": {"G": "IN"}},  # no w/l
        },
    })
    tpl = load_template(p)
    assert tpl.device_params["N1"]["w"] == pytest.approx(0.42)
    assert "N2" not in tpl.device_params


# ── Nets ───────────────────────────────────────────────────────────────────

def test_nets_dict_with_explicit_types(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "nets": {
            "VDD": {"type": "power", "rail": "top"},
            "VSS": {"type": "power", "rail": "bottom"},
            "IN":  {"type": "signal", "layer": "M1"},
        },
    })
    tpl = load_template(p)
    assert tpl.nets["VDD"].net_type == "power" and tpl.nets["VDD"].rail == "top"
    assert tpl.nets["IN"].layer == "m1"   # M1 normalised → m1


def test_nets_list_form_infers_power_rails(tmp_path: Path):
    p = _write(tmp_path, {"devices": {}, "nets": ["VDD", "VSS", "IN"]})
    tpl = load_template(p)
    assert tpl.nets["VDD"].net_type == "power" and tpl.nets["VDD"].rail == "top"
    assert tpl.nets["VSS"].net_type == "power" and tpl.nets["VSS"].rail == "bottom"
    assert tpl.nets["IN"].net_type  == "signal"


def test_undeclared_terminal_nets_promoted_to_internal(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "IN", "D": "MID", "S": "VSS"}},
            "N2": {"type": "nmos", "terminals": {"G": "MID", "D": "OUT", "S": "VSS"}},
        },
        "nets": ["VDD", "VSS", "IN", "OUT"],
    })
    tpl = load_template(p)
    assert tpl.nets["MID"].net_type == "internal"


# ── Ports ──────────────────────────────────────────────────────────────────

def test_ports_parsed_and_layer_normalised(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "ports": {
            "IN":  {"side": "west", "layer": "M1"},
            "OUT": {"side": "east"},
        },
    })
    tpl = load_template(p)
    assert tpl.ports["IN"].side  == "west"
    assert tpl.ports["IN"].layer == "m1"
    assert tpl.ports["OUT"].layer == ""    # absent → empty string


# ── Placement: standard mode ──────────────────────────────────────────────

def test_standard_pairs_assigns_regions_and_x_specs(tmp_path: Path):
    """Standard mode lays NMOS row at bottom, PMOS row at top with gate-Y offset."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "N2": {"type": "nmos", "terminals": {"G": "B"}},
            "P1": {"type": "pmos", "terminals": {"G": "A"}},
            "P2": {"type": "pmos", "terminals": {"G": "B"}},
        },
        "placement": {
            "mode": "standard",
            "pairs": [{"nmos": ["N1", "N2"], "pmos": ["P1", "P2"]}],
        },
    })
    tpl = load_template(p)
    assert tpl.layout_mode == "standard"
    # NMOS row pinned to bottom; first device anchored at "left".
    assert tpl.devices["N1"].region == "bottom"
    assert tpl.devices["N1"].x_spec == "left"
    assert tpl.devices["N2"].x_spec.startswith("N1_x")
    # PMOS row sits above NMOS row.
    assert tpl.devices["P1"].region   == "top"
    assert tpl.devices["P1"].in_nwell is True
    assert "inter_cell_gap" in tpl.devices["P1"].y_offset_expr


def test_standard_relations_shared_diffusion_emits_abutment_x_spec(tmp_path: Path):
    """A shared_diffusion relation forces an abutment x_spec on the second device."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "N2": {"type": "nmos", "terminals": {"G": "B"}},
        },
        "placement": {
            "mode": "standard",
            "pairs": [{"nmos": ["N1", "N2"], "pmos": []}],
            "relations": {"shared_diffusion": [["N1", "N2"]]},
        },
    })
    tpl = load_template(p)
    # Abutment: x_spec subtracts one S/D region from the trailing device.
    assert "- N1.sd" in tpl.devices["N2"].x_spec
    # Relations preserved verbatim for downstream consumers.
    assert tpl.placement_relations["shared_diffusion"] == [["N1", "N2"]]


def test_standard_relations_gate_align_pins_pmos_x_to_nmos(tmp_path: Path):
    """A gate_align relation places the PMOS device at the same X as its NMOS twin."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "P1": {"type": "pmos", "terminals": {"G": "A"}},
        },
        "placement": {
            "mode": "standard",
            "pairs": [{"nmos": ["N1"], "pmos": ["P1"]}],
            "relations": {"gate_align": [["N1", "P1"]]},
        },
    })
    tpl = load_template(p)
    assert tpl.devices["P1"].x_spec == "N1_x"


def test_standard_relations_cross_couple_gap_adds_named_spacing(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "N2": {"type": "nmos", "terminals": {"G": "B"}},
        },
        "placement": {
            "mode": "standard",
            "pairs": [{"nmos": ["N1", "N2"], "pmos": []}],
            "relations": {"cross_couple_gap": [["N1", "N2"]]},
        },
    })
    tpl = load_template(p)
    assert "cross_gap" in tpl.devices["N2"].x_spec


# ── Placement: stacked mode ────────────────────────────────────────────────

def test_stacked_mode_assigns_row_pair_and_region(tmp_path: Path):
    """Stacked mode populates RowPairSpec and pins per-device region/nwell flags."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "P1": {"type": "pmos", "terminals": {"G": "A"}},
        },
        "placement": {
            "mode": "stacked",
            "row_pairs": [
                {"id": 0, "nmos": ["N1"], "pmos": ["P1"],
                 "rail_top": "VDD", "rail_bottom": "VSS",
                 "sd_flip": {"N1": True}},
            ],
        },
    })
    tpl = load_template(p)
    assert tpl.layout_mode == "stacked"
    assert len(tpl.row_pairs) == 1
    rp = tpl.row_pairs[0]
    assert isinstance(rp, RowPairSpec)
    assert rp.rail_top == "VDD" and rp.rail_bottom == "VSS"
    assert tpl.devices["N1"].region   == "bottom"
    assert tpl.devices["N1"].sd_flip  is True
    assert tpl.devices["N1"].row_pair == 0
    assert tpl.devices["P1"].region   == "top"
    assert tpl.devices["P1"].in_nwell is True


# ── Placement: directives mode ────────────────────────────────────────────

def test_directives_mode_via_placement_logic(tmp_path: Path):
    """``placement_logic`` is parsed into PlacementDirective objects."""
    p = _write(tmp_path, {
        "devices": {
            "N1": {"type": "nmos", "terminals": {"G": "A"}},
            "P1": {"type": "pmos", "terminals": {"G": "A"}},
        },
        "placement_logic": [
            {"name": "N1", "origin": [0.0, 0.0]},
            {"name": "P1", "relative_to": "N1", "relation": "align_gate",
             "alignment": "gate", "orientation": "MX", "sd_flip": True},
        ],
    })
    tpl = load_template(p)
    assert [pd.name for pd in tpl.placement_directives] == ["N1", "P1"]
    n1, p1 = tpl.placement_directives
    assert n1.origin == (0.0, 0.0)
    assert p1.relative_to == "N1"
    assert p1.orientation == "MX"
    assert p1.sd_flip is True


def test_directives_mode_via_placement_list(tmp_path: Path):
    """A bare ``placement`` list also feeds the directives parser."""
    p = _write(tmp_path, {
        "devices": {"N1": {"type": "nmos", "terminals": {"G": "A"}}},
        "placement": [
            {"name": "N1", "origin": [0.5, 0.0]},
        ],
    })
    tpl = load_template(p)
    assert tpl.layout_mode == "directives"
    assert tpl.placement_directives[0].name   == "N1"
    assert tpl.placement_directives[0].origin == (0.5, 0.0)


# ── Routing hints ──────────────────────────────────────────────────────────

def test_routing_hints_dict_form(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "routing": {
            "OUT": {"layer": "M2", "path_type": "trunk"},
        },
    })
    tpl = load_template(p)
    h = tpl.routing_hints["OUT"]
    assert isinstance(h, RoutingHint)
    assert h.layer == "m2"
    assert h.style == "full_width"   # inferred from path_type=trunk


def test_routing_hints_list_form_and_port_side_string(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "routing": [
            {"net": "WL", "layer": "M1", "path_type": "rail", "port_side": "west"},
            {"net": "BL", "layer": "M2", "coverage": "full_height"},
        ],
    })
    tpl = load_template(p)
    wl = tpl.routing_hints["WL"]
    bl = tpl.routing_hints["BL"]
    assert wl.layer == "m1" and wl.style == "full_width"
    assert wl.port_side == ["west"]
    assert bl.layer == "m2" and bl.style == "full_height"


# ── Cell dimensions / abutment / label layers ──────────────────────────────

def test_cell_dimensions_and_abutment(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "cell_dimensions": {"width": 1.0, "height": 2.72},
        "abutment": {"pitch_x": 0.46, "mirror_x": True, "rail_align": False},
    })
    tpl = load_template(p)
    assert tpl.cell_dimensions == CellDimensions(width=1.0, height=2.72)
    assert tpl.abutment.pitch_x   == pytest.approx(0.46)
    assert tpl.abutment.mirror_x  is True
    assert tpl.abutment.rail_align is False
    # pitch_y wasn't specified → default 0.0
    assert tpl.abutment.pitch_y   == 0.0


def test_label_layers_default_none(tmp_path: Path):
    p = _write(tmp_path, {"devices": {}})
    tpl = load_template(p)
    assert isinstance(tpl.label_layers, LabelLayerSpec)
    assert tpl.label_layers.m1 is None
    assert tpl.label_layers.m2 is None


def test_label_layers_populated_from_yaml(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "label_layers": {"m1": [68, 5], "m2": [69, 5]},
    })
    tpl = load_template(p)
    assert tpl.label_layers.m1 == (68, 5)
    assert tpl.label_layers.m2 == (69, 5)


# ── Diffusion merges ───────────────────────────────────────────────────────

def test_diffusion_merges_parsed_into_pairs(tmp_path: Path):
    p = _write(tmp_path, {
        "devices": {},
        "diffusion_merge": [["N1", "N2"], ["P1", "P2"]],
    })
    tpl = load_template(p)
    assert tpl.diffusion_merges == [("N1", "N2"), ("P1", "P2")]


# ── Path resolution ────────────────────────────────────────────────────────

def test_load_by_path_existing_yaml(tmp_path: Path):
    p = _write(tmp_path, {"name": "demo", "devices": {}}, name="demo.yaml")
    tpl = load_template(p)
    assert tpl.name == "demo"
    assert tpl.source_path == p


def test_load_by_name_via_search_dirs(tmp_path: Path):
    """A bare template name resolves against ``search_dirs``."""
    cells = tmp_path / "cells"
    cells.mkdir()
    target = cells / "inverter.yaml"
    target.write_text(yaml.safe_dump({"name": "inverter", "devices": {}}), encoding="utf-8")
    tpl = load_template("inverter", search_dirs=[tmp_path])
    assert tpl.name == "inverter"
    assert tpl.source_path == target


def test_load_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Template 'nope' not found"):
        load_template("nope", search_dirs=[tmp_path])


# ── Smoke: top-level CellTemplate shape ────────────────────────────────────

def test_minimal_template_has_default_dataclass_instances(tmp_path: Path):
    p = _write(tmp_path, {"devices": {}})
    tpl = load_template(p)
    assert isinstance(tpl, CellTemplate)
    assert isinstance(tpl.cell_dimensions, CellDimensions)
    assert isinstance(tpl.abutment, AbutmentSpec)
    assert isinstance(tpl.label_layers, LabelLayerSpec)
    assert tpl.row_pairs           == []
    assert tpl.placement_directives == []
    assert tpl.diffusion_merges    == []
