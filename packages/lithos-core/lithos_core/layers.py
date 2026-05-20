"""lithos_core.layers — unified per-layer PDK descriptor.

`layers.yaml` is the single-file form of a PDK descriptor: per-layer GDS,
foundry aliases, DRC rule prefixes, PDF aliases / section hints, and
semantic-rule bindings, plus PDK-global facts (grid, deck paths,
devices). See `docs/LAYERS_FILE.md` for the rationale and migration
context.

This module loads `layers.yaml` and provides three adapter views so
existing consumers stay unchanged:

* :meth:`LayersFile.as_pdk_metadata` → :class:`PDKMetadata`
* :meth:`LayersFile.as_bootstrap_mapping_dict` → flat semantic mapping
* :meth:`LayersFile.as_category_config` → :class:`CategoryConfig`

It also surfaces :meth:`pdf_aliases_for` so the chunker can fall back
to PDF placeholders when an explicit code doesn't appear in the doc
(e.g. `M3.W.1` → `Mx.W.1`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lithos_core.categories import CategoryConfig, CategoryDef
from lithos_core.metadata import PDKMetadata


# ── Schema ─────────────────────────────────────────────────────────────────

class LayerDef(BaseModel):
    """One layer's per-layer descriptor.

    All fields are optional except ``gds`` for physical layers. A purely
    synthetic layer (alias of another) can omit ``gds``.
    """
    model_config = ConfigDict(frozen=True)

    name:                str
    gds:                 Optional[tuple[int, int]] = None
    label_datatype:      Optional[int]             = None
    foundry_aliases:     list[str]                 = Field(default_factory=list)
    rule_prefixes:       list[str]                 = Field(default_factory=list)
    pdf_aliases:         list[str]                 = Field(default_factory=list)
    pdf_section_pattern: Optional[str]             = None
    preferred_direction: str                       = ""
    semantic_rules:      dict[str, str]            = Field(default_factory=dict)
    alias_of:            Optional[str]             = None
    description:         str                       = ""


class LayersFile(BaseModel):
    """A unified PDK descriptor loaded from ``layers.yaml``."""
    model_config = ConfigDict(frozen=True)

    name:              str
    version:           str
    schema_version:    int                            = 1
    grid:              dict[str, float]               = Field(default_factory=dict)
    drc_decks:         dict[str, Path]                = Field(default_factory=dict)
    devices:           dict[str, dict[str, Any]]      = Field(default_factory=dict)
    layers:            dict[str, LayerDef]            = Field(default_factory=dict)
    extra_categories:  list[CategoryDef]              = Field(default_factory=list)
    source_path:       Optional[Path]                 = None

    # ── Adapter views ─────────────────────────────────────────────────

    def as_pdk_metadata(self) -> PDKMetadata:
        """Return the :class:`PDKMetadata` view of this descriptor."""
        gds_layers: dict[str, tuple[int, int]] = {}
        label_layers: dict[str, tuple[int, int]] = {}
        preferred: dict[str, str] = {}
        for canonical, ldef in self.layers.items():
            if ldef.gds is not None:
                gds_layers[canonical] = ldef.gds
            if ldef.label_datatype is not None and ldef.gds is not None:
                # Label layer tuple defaults to (gds_layer, label_datatype)
                # — the per-layer datatype the foundry uses for ports.
                label_layers[canonical] = (ldef.gds[0], ldef.label_datatype)
            if ldef.preferred_direction:
                preferred[canonical] = ldef.preferred_direction
        return PDKMetadata(
            name                = self.name,
            version             = self.version,
            layers              = gds_layers,
            grid                = dict(self.grid),
            drc_decks           = dict(self.drc_decks),
            preferred_direction = preferred,
            label_layers        = label_layers,
            devices             = dict(self.devices),
            metadata_path       = self.source_path,
        )

    def as_bootstrap_mapping_dict(self) -> dict[str, str]:
        """Return the flat semantic→code mapping for `BootstrapMapping`.

        Keys are dotted ``<canonical-layer>.<semantic-key>``, e.g.
        ``"poly.width_min_um"`` → ``"PO.W.1"``.
        """
        out: dict[str, str] = {}
        for canonical, ldef in self.layers.items():
            for k, v in ldef.semantic_rules.items():
                out[f"{canonical}.{k}"] = v
        return out

    def as_category_config(self) -> CategoryConfig:
        """Derive a :class:`CategoryConfig` from per-layer ``rule_prefixes``.

        Each layer with a non-empty ``rule_prefixes`` becomes one
        category named after the layer. Anything in ``extra_categories``
        is appended after the per-layer entries.
        """
        cats: list[CategoryDef] = []
        priority = 10
        for canonical, ldef in self.layers.items():
            if not ldef.rule_prefixes:
                continue
            cats.append(CategoryDef(
                name                = canonical,
                code_prefixes       = list(ldef.rule_prefixes),
                pdf_section_pattern = ldef.pdf_section_pattern,
                enabled             = True,
                priority            = priority,
                description         = ldef.description,
            ))
            priority += 1
        cats.extend(self.extra_categories)
        return CategoryConfig(categories=cats, default_category="unknown")

    # ── PDF-alias lookup ──────────────────────────────────────────────

    def pdf_aliases_for(self, rule_code: str) -> list[str]:
        """Return the list of alternative codes to try in the PDF.

        For each layer whose ``rule_prefixes`` claim this code, generate
        candidates by substituting the layer's matched prefix with each
        ``pdf_aliases`` entry. Example::

            layer m3:
              rule_prefixes: ["M3."]
              pdf_aliases:   ["Mx"]

            pdf_aliases_for("M3.W.1") == ["Mx.W.1"]

        Returns ``[]`` when no layer claims the code or no aliases exist.
        """
        out: list[str] = []
        for ldef in self.layers.values():
            for pfx in ldef.rule_prefixes:
                if rule_code.startswith(pfx) and ldef.pdf_aliases:
                    tail = rule_code[len(pfx):]
                    # Strip the trailing dot from the prefix when building
                    # the alias-prefixed candidate, so "M3." + tail "W.1"
                    # becomes "Mx" + "." + "W.1" = "Mx.W.1".
                    pfx_stem = pfx.rstrip(".")
                    for alias in ldef.pdf_aliases:
                        alias_stem = alias.rstrip(".")
                        candidate = (
                            f"{alias_stem}.{tail}" if tail else alias_stem
                        )
                        if candidate != rule_code and candidate not in out:
                            out.append(candidate)
                    break    # one matching prefix is enough per layer
        return out


# ── Loader ─────────────────────────────────────────────────────────────────

def load_layers_file(path: Path | str) -> LayersFile:
    """Load a ``layers.yaml`` file and resolve deck paths + env vars."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if "name" not in data or "version" not in data:
        raise ValueError(
            f"layers.yaml at {path} must define both 'name' and 'version'."
        )

    # Per-layer parsing
    layers: dict[str, LayerDef] = {}
    raw_layers = data.get("layers") or {}
    if not isinstance(raw_layers, dict):
        raise ValueError(
            f"layers.yaml at {path}: 'layers' must be a mapping, got "
            f"{type(raw_layers).__name__}"
        )
    for canonical, entry in raw_layers.items():
        entry = entry or {}
        gds = entry.get("gds")
        gds_tuple: Optional[tuple[int, int]] = None
        if isinstance(gds, dict) and "layer" in gds and "datatype" in gds:
            gds_tuple = (int(gds["layer"]), int(gds["datatype"]))
        layers[canonical] = LayerDef(
            name                = canonical,
            gds                 = gds_tuple,
            label_datatype      = (
                int(entry["label_datatype"])
                if "label_datatype" in entry and entry["label_datatype"] is not None
                else None
            ),
            foundry_aliases     = list(entry.get("foundry_aliases") or []),
            rule_prefixes       = list(entry.get("rule_prefixes") or []),
            pdf_aliases         = list(entry.get("pdf_aliases") or []),
            pdf_section_pattern = entry.get("pdf_section_pattern"),
            preferred_direction = str(entry.get("preferred_direction") or ""),
            semantic_rules      = dict(entry.get("semantic_rules") or {}),
            alias_of            = entry.get("alias_of"),
            description         = str(entry.get("description") or ""),
        )

    # PDK-global parsing
    decks: dict[str, Path] = {}
    for tool, raw in (data.get("drc_decks") or {}).items():
        env_key = f"DRC_DECK_{tool.upper()}"
        resolved = os.environ.get(env_key) or os.path.expandvars(str(raw))
        p = Path(resolved)
        if not p.is_absolute():
            p = path.parent / p
        decks[tool] = p

    grid = {k: float(v) for k, v in (data.get("grid") or {}).items()}
    devices = data.get("devices") or {}
    if not isinstance(devices, dict):
        raise ValueError(
            f"layers.yaml at {path}: 'devices' must be a mapping, got "
            f"{type(devices).__name__}"
        )

    extras_raw = data.get("extra_categories") or []
    extras: list[CategoryDef] = []
    for c in extras_raw:
        if not isinstance(c, dict) or "name" not in c:
            continue
        extras.append(CategoryDef(
            name                = c["name"],
            code_prefixes       = list(c.get("code_prefixes") or []),
            code_pattern        = c.get("code_pattern"),
            pdf_section_pattern = c.get("pdf_section_pattern"),
            enabled             = bool(c.get("enabled", True)),
            priority            = int(c.get("priority", 100)),
            description         = str(c.get("description") or ""),
        ))

    return LayersFile(
        name             = str(data["name"]),
        version          = str(data["version"]),
        schema_version   = int(data.get("schema_version", 1)),
        grid             = grid,
        drc_decks        = decks,
        devices          = dict(devices),
        layers           = layers,
        extra_categories = extras,
        source_path      = path,
    )
