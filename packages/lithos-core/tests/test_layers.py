"""Tests for lithos_core.layers — unified layers.yaml schema + loader."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lithos_core import (
    CategoryConfig,
    LayerDef,
    LayersFile,
    PDKMetadata,
    load_layers_file,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_layers_yaml(tmp_path: Path) -> Path:
    """Three-layer example exercising every per-layer field."""
    p = tmp_path / "layers.yaml"
    p.write_text(textwrap.dedent("""
        name: testpdk
        version: "1.0"

        grid:
          manufacturing_um: 0.005
          routing_um:       0.005

        devices:
          nmos:
            diffusion_layer: diff
            gate_layer: poly
            implant: nimplant

        layers:
          poly:
            gds: {layer: 66, datatype: 20}
            foundry_aliases: [PO]
            rule_prefixes: ["PO."]
            pdf_section_pattern: '(?i)polysilicon'
            preferred_direction: vertical
            semantic_rules:
              width_min_um:   PO.W.1
              spacing_min_um: PO.S.1

          m1:
            gds: {layer: 68, datatype: 20}
            label_datatype: 5
            foundry_aliases: [MET1]
            rule_prefixes: ["M1."]
            preferred_direction: vertical
            semantic_rules:
              width_min_um: M1.W.1

          m3:
            gds: {layer: 71, datatype: 20}
            rule_prefixes: ["M3."]
            pdf_aliases:   ["Mx"]

        extra_categories:
          - name: antenna
            code_prefixes: ["ANT.", "A."]
            enabled: false
            priority: 90
    """).strip())
    return p


# ── Loader ─────────────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_basic_fields(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert isinstance(lf, LayersFile)
        assert lf.name == "testpdk"
        assert lf.version == "1.0"
        assert lf.schema_version == 1
        assert lf.grid["manufacturing_um"] == pytest.approx(0.005)
        assert set(lf.layers) == {"poly", "m1", "m3"}

    def test_per_layer_fields(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        poly = lf.layers["poly"]
        assert isinstance(poly, LayerDef)
        assert poly.gds == (66, 20)
        assert poly.foundry_aliases == ["PO"]
        assert poly.rule_prefixes == ["PO."]
        assert poly.preferred_direction == "vertical"
        assert poly.semantic_rules == {
            "width_min_um":   "PO.W.1",
            "spacing_min_um": "PO.S.1",
        }
        assert poly.pdf_aliases == []

    def test_pdf_alias_field(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert lf.layers["m3"].pdf_aliases == ["Mx"]

    def test_label_datatype(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert lf.layers["m1"].label_datatype == 5
        assert lf.layers["poly"].label_datatype is None

    def test_extra_categories(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert len(lf.extra_categories) == 1
        ant = lf.extra_categories[0]
        assert ant.name == "antenna"
        assert ant.enabled is False
        assert ant.code_prefixes == ["ANT.", "A."]

    def test_requires_name_and_version(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("layers: {}\n")
        with pytest.raises(ValueError, match="must define both"):
            load_layers_file(p)

    def test_rejects_bad_layers_section(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nversion: '1.0'\nlayers: not_a_mapping\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_layers_file(p)


# ── Adapter views ──────────────────────────────────────────────────────────

class TestPdkMetadataAdapter:
    def test_returns_pdkmetadata(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        md = lf.as_pdk_metadata()
        assert isinstance(md, PDKMetadata)
        assert md.name == "testpdk"
        assert md.version == "1.0"

    def test_gds_layers_propagate(self, minimal_layers_yaml: Path):
        md = load_layers_file(minimal_layers_yaml).as_pdk_metadata()
        assert md.layer("poly") == (66, 20)
        assert md.layer("m1")   == (68, 20)
        assert md.layer("m3")   == (71, 20)

    def test_label_layers_from_label_datatype(self, minimal_layers_yaml: Path):
        md = load_layers_file(minimal_layers_yaml).as_pdk_metadata()
        # label_layers ends up as (gds_layer, label_datatype).
        assert md.label_layers == {"m1": (68, 5)}

    def test_preferred_direction(self, minimal_layers_yaml: Path):
        md = load_layers_file(minimal_layers_yaml).as_pdk_metadata()
        assert md.preferred_direction == {"poly": "vertical", "m1": "vertical"}


class TestBootstrapMappingAdapter:
    def test_flat_dotted_keys(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        m = lf.as_bootstrap_mapping_dict()
        assert m == {
            "poly.width_min_um":   "PO.W.1",
            "poly.spacing_min_um": "PO.S.1",
            "m1.width_min_um":     "M1.W.1",
        }


class TestCategoryConfigAdapter:
    def test_one_category_per_layer_with_prefixes(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        cc = lf.as_category_config()
        assert isinstance(cc, CategoryConfig)
        names = [c.name for c in cc.categories]
        # Per-layer categories first (in YAML order), then extras.
        assert names == ["poly", "m1", "m3", "antenna"]

    def test_classifies_codes(self, minimal_layers_yaml: Path):
        cc = load_layers_file(minimal_layers_yaml).as_category_config()
        assert cc.category_for("PO.W.1") == "poly"
        assert cc.category_for("M1.W.1") == "m1"
        assert cc.category_for("M3.W.1") == "m3"
        assert cc.category_for("OD.W.1") == "unknown"


# ── pdf_aliases_for ────────────────────────────────────────────────────────

class TestPdfAliasLookup:
    def test_returns_alias_form(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert lf.pdf_aliases_for("M3.W.1") == ["Mx.W.1"]
        assert lf.pdf_aliases_for("M3.S.3") == ["Mx.S.3"]

    def test_no_alias_for_layer_without_pdf_aliases(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert lf.pdf_aliases_for("PO.W.1") == []
        assert lf.pdf_aliases_for("M1.W.1") == []

    def test_unknown_code_returns_empty(self, minimal_layers_yaml: Path):
        lf = load_layers_file(minimal_layers_yaml)
        assert lf.pdf_aliases_for("NW.W.1") == []
        assert lf.pdf_aliases_for("") == []

    def test_multiple_aliases(self, tmp_path: Path):
        p = tmp_path / "layers.yaml"
        p.write_text(textwrap.dedent("""
            name: t
            version: '1'
            layers:
              m8:
                gds: {layer: 80, datatype: 20}
                rule_prefixes: ["M8."]
                pdf_aliases: ["My", "Mz"]
        """).strip())
        lf = load_layers_file(p)
        assert lf.pdf_aliases_for("M8.W.1") == ["My.W.1", "Mz.W.1"]
