# `layers.yaml` — unified PDK / layer / rule descriptor

**Status:** schema + initial implementation. Loader lives in
`lithos_core.layers`; CLI consumes it via `lithos-ingest full --layers`.

## Why

A lithos PDK is described today by **three** separate YAMLs:

| file                | schema                  | consumed by                       |
| ------------------- | ----------------------- | --------------------------------- |
| `metadata.yaml`     | `PDKMetadata`           | `lithos-layout` (GDS, grid, decks) |
| `bootstrap.yaml`    | `BootstrapMapping`      | `lithos-layout` (semantic → code)  |
| `categories.yaml`   | `CategoryConfig`        | `lithos-ingest` (rule scoping)     |

Each file has its own loader, its own schema, and a different organising
axis (PDK-global vs. layer-by-layer vs. topical-rule-group). To onboard
a new PDK a user has to hand-author all three and keep them consistent
— e.g. the canonical name `m0` lives in both `metadata.yaml` (as a GDS
key) and `bootstrap.yaml` (as a section under which rule codes are
mapped), and `m0`'s foundry rule prefix (e.g. `LI.`) has to land in
`categories.yaml` too.

The user-facing concept that unifies all three is the **layer**. Every
fact a PDK file carries is either:

1. PDK-global (grid, deck paths, devices), or
2. Per-layer (gds tuple, foundry name aliases, DRC rule prefixes, PDF
   section hints, label datatype, preferred direction, semantic-rule
   bindings).

Hence: one `layers.yaml` per PDK, organised by layer.

## Proposed schema

```yaml
# Top of the file — PDK identity + globals.
name: examplepdk180
version: 2.15_2a

grid:
  manufacturing_um: 0.005
  routing_um:       0.005

drc_decks:
  klayout: $PDK_ROOT/.../examplepdk180.lydrc
  magic:   $PDK_ROOT/.../examplepdk180.magicrc

# Per-device-type structural facts (unchanged from PDKMetadata.devices).
devices:
  nmos: {diffusion_layer: diff, gate_layer: poly, implant: nimplant}
  pmos: {diffusion_layer: diff, gate_layer: poly, implant: pimplant, in_well: nwell}

# Per-layer descriptors. The canonical lithos name is the key.
# Every field below is optional except `gds`.
layers:
  poly:
    gds: {layer: 66, datatype: 20}
    foundry_aliases: [PO, POLY]
    rule_prefixes:   ["PO.", "POLY."]
    pdf_section_pattern: '(?i)polysilicon|poly\b|gate poly'
    preferred_direction: vertical
    semantic_rules:                       # → BootstrapMapping facet
      width_min_um:        PO.W.1
      spacing_min_um:      PO.S.1
      endcap_over_diff_um: PO.E.1

  # ── PDF alias example: a metal whose PDF documents it under a class
  # placeholder (e.g. `Mx.W.1` covers `M2.W.1`, `M3.W.1`, `M4.W.1`,
  # `M5.W.1` in some foundry DRMs). When the chunker can't find the
  # explicit code, it retries each `pdf_aliases` entry as a prefix
  # substitution: `M3.W.1` → `Mx.W.1`.
  m3:
    gds: {layer: 71, datatype: 20}
    rule_prefixes: ["M3."]
    pdf_aliases:   ["Mx"]                 # M3.X.Y → Mx.X.Y in the PDF
    preferred_direction: horizontal

  diff:
    gds: {layer: 65, datatype: 20}
    foundry_aliases: [OD, DIFF, RX]
    rule_prefixes:   ["OD.", "DIFF.", "RX."]
    pdf_section_pattern: '(?i)active|diffusion|od\b'
    semantic_rules:
      width_min_um:            OD.W.1
      extension_past_poly_um:  DI.E.1

  contact:
    gds: {layer: 66, datatype: 44}
    foundry_aliases: [CO, LICON]
    rule_prefixes:   ["CO.", "CONT.", "LICON."]
    pdf_section_pattern: '(?i)contact|cut'
    semantic_rules:
      size_um:                   CO.W.1
      spacing_um:                CO.S.1
      enclosure_in_diff_um:      CO.E.1

  m0:
    gds: {layer: 67, datatype: 20}
    label_datatype: 16
    foundry_aliases: [LI, LI1]            # collapsed onto m1 in non-sky PDKs
    rule_prefixes:   ["LI.", "M1."]
    pdf_section_pattern: '(?i)local interconnect|li\b|m1\b'
    preferred_direction: horizontal
    semantic_rules:
      width_min_um:    LI.W.1
      spacing_min_um:  LI.S.1

  m1:
    gds: {layer: 68, datatype: 20}
    label_datatype: 5
    foundry_aliases: [MET1]
    rule_prefixes:   ["M1.", "MET1."]
    pdf_section_pattern: '(?i)metal[\s\-]?1|met1'
    preferred_direction: vertical

  # …m2, m3, …, via_m0_m1, via_m1_m2, …

# Layer-spanning categories that don't belong to one specific layer.
# These are the residue of CategoryConfig (e.g. antenna rules).
extra_categories:
  - name: antenna
    code_prefixes: [ANTENNA., A.]
    enabled: false                        # opt-in
    priority: 90
```

## Properties

- **One source of truth per layer** — adding a new layer is one block.
- **Lithos-canonical name is the key** — every PDK file uses the same
  layer names (`m0`, `poly`, `contact`, `via_m0_m1`, …); the PDK only
  provides translation (`foundry_aliases`, `gds`, `rule_prefixes`).
- **Self-contained categorisation** — the ingest pipeline derives
  `CategoryConfig` from `layers[*].rule_prefixes`. Each layer's
  per-layer rule prefixes are turned into a `CategoryDef` named after
  the layer (e.g. `category="m0"`); cell code already speaks
  in canonical layer names, so this matches.
- **Backward-compatible** — `layers.yaml` loads into the same three
  in-memory objects (`PDKMetadata`, `BootstrapMapping`,
  `CategoryConfig`) so no consumer changes.

## Migration plan

1. Add `lithos_core/layers.py` with `load_layers_file(path)` → returns
   a `LayersFile` view and three derived helpers:
   `LayersFile.as_pdk_metadata() → PDKMetadata`,
   `.as_bootstrap_mapping() → BootstrapMapping`,
   `.as_category_config() → CategoryConfig`.
2. Existing `metadata.yaml` / `bootstrap.yaml` / `categories.yaml`
   continue to work — `layers.yaml` is an *alternative* entry point.
3. Update `lithos-ingest full --layers <file>` and similar CLI flags
   on `lithos-layout` to accept the unified file.
4. Once the user has migrated their PDKs, deprecate the three legacy
   loaders (no rush; keep both supported for one release cycle).

## Open questions

1. **Layer-spanning rules** — cross-layer spacing like `M1.M2.S.1`.
   Either keep them under whichever layer "owns" them, or have a
   top-level `cross_layer_rules:` section. Proposal: keep under the
   layer that's the rule's *first argument* (foundry convention).
2. **Multi-PDK aliases** — when a PDK collapses `m0` onto `m1` (no
   local interconnect), the `m0` layer block becomes synthetic
   (`m0: {alias_of: m1}`). Worth supporting.
3. **Versioning** — should `layers.yaml` carry a schema-version field
   so we can evolve it? (Recommended: yes, `schema_version: 1`.)
4. **PDF section hints maintenance** — these regexes are fragile (vary
   across PDF revisions). A separate `pdf_section_overrides.yaml`
   keyed by PDK + version could keep `layers.yaml` clean.

## Out of scope (deliberately)

- Rule-content extraction quality. `layers.yaml` doesn't change how
  rules are parsed, only how they're *scoped*. The coverage problems
  on heavily parameterised codes (e.g. `ADP.C.1_PL_V1_V2` not matching
  `ADP.C.1` in the PDF) need a separate chunker fix (base-code
  fallback).
- LLM extraction policy. Whether to enrich a rule via Ollama / Claude
  / GGUF stays a per-run flag, not a layer property.
