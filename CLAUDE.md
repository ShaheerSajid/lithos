# lithos — agent context

`lithos` is a PDK-agnostic layout generator with an LLM-built rule
database. It's the successor to `layout_gen`, restructured as a uv
workspace of seven packages (one per concern). The end goal is to
synthesise GDS for arbitrary standard cells from a topology YAML by
combining a parsed DRC deck (`lithos-core` + `lithos-ingest`), a typed
cell-generation layer (`lithos-layout`), and a learned DRC repair loop
(`lithos-repair` + `lithos-rl`, both designed but not yet built).

## Packages and current state

| package          | status        | what's in it                                                                                                                                       |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lithos-core`    | content       | rule DB schema, Pydantic IR (`Constraint`, `LayerExpr`, `CheckExpr`, `FixMetadata`, `ExistenceCheck`), `PDKMetadata`, `CategoryConfig`.            |
| `lithos-ingest`  | content       | SVRF + KLayout-DRC parsers, PDF/HTML/RST/CSV loaders, code-anchored chunker, joiner, writer, CLI, `FixMetadataExtractor` (Ollama / llama-cpp / Anthropic adapters). |
| `lithos-drc`     | content       | `DRCRunner` interface + KLayout and Magic backends, alias resolver.                                                                                |
| `lithos-layout`  | content       | `BootstrapRules`, transistor math + GDS emitter, `cells/` (via stacks + tap), `synth/loader.py` (topology YAML → typed specs). See sections below. |
| `lithos-repair`  | **stub only** | designed in [docs/REPAIR_ARCHITECTURE.md](docs/REPAIR_ARCHITECTURE.md).                                                                            |
| `lithos-rl`      | **stub only** | RL stack still lives in the old `layout_gen` repo (commit `23cb778` on `drc-repair-engine`); awaits port.                                          |
| `lithos-lvs`     | **stub only** | netgen + magic extraction yet to be ported.                                                                                                        |

Open the [docs/PORTING_PLAN.md](docs/PORTING_PLAN.md) for the full
step-by-step roadmap (and which steps are done).

## Project invariants

### 1. PDK-agnostic metal-stack naming

`lithos-*` code never uses foundry-specific layer names. Internally
everything is the canonical metal stack; per-PDK YAMLs map abstract
names onto physical (gds_layer, datatype) pairs.

- **Metals**: `m0`, `m1`, `m2`, … (bottom-up from local interconnect)
- **Cuts**: `contact` (poly/diff → m0), `via_m0_m1`, `via_m1_m2`, …
- **Active / gate**: `poly`, `diff`, `tap`, `nwell`, `nimplant`, `pimplant`,
  `npc` (sky130-only, optional)
- **Bootstrap semantic keys**: `m0.width_min_um`, `contact.size_um`,
  `contact.enclosure_in_m0_2adj_um`, `via_m0_m1.size_um`,
  `m1.enclosure_of_via_m0_m1_2adj_um`, etc.
- **Cell function names**: `via_poly_m0`, `via_diff_m0`, `via_m0_m1`,
  `via_m1_m2`, plus composite stacks `via_poly_m1`, `via_poly_m2`,
  `via_m0_m2`.
- **Property**: `BootstrapRules.m0_is_m1` — true when a PDK collapses
  local interconnect onto m1 (GF180-style); via cells use it to skip
  the m0→m1 cut.

Names like `li1`, `met1`, `mcon`, `licon1`, `via1` appear only inside
per-PDK metadata YAMLs and bootstrap mappings, never in Python code or
in port-`layer:` strings inside topology templates.

### 2. Real DRC for runs

`generate` / `train` flows must use a real `DRCRunner` (KLayout or
Magic). Fake DRC is permitted only in unit tests. KLayout is at
`/usr/bin/klayout` (Python module also installed); Magic is at
`/usr/local/bin/magic`. `PDK_ROOT` is unset by default.

### 3. CPU-only by default

There is no NVIDIA GPU available locally. Any new RL/training code
should default to CPU torch wheels and not assume CUDA at import time.

## Running tests

`lithos` is a uv workspace; with uv installed:

```bash
cd /path/to/lithos
uv run pytest -q
```

Without uv (e.g. on a fresh machine using the old `layout_gen` venv),
the explicit-PYTHONPATH form is:

```bash
cd /path/to/lithos
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
PYTHONPATH="packages/lithos-core:packages/lithos-ingest:packages/lithos-drc:packages/lithos-layout:packages/lithos-repair:packages/lithos-rl:packages/lithos-lvs" \
    /path/to/venv/bin/python -m pytest -q
```

Current suite: **251 passing tests**.

## lithos-layout deep-dive

The cell-generation half. Imports `lithos_core` (rule IR + `PDKMetadata`),
exposes everything the synthesiser/placer/router needs:

- **`BootstrapRules`** (`rules.py`) — wraps `PDKMetadata + RuleDB +
  BootstrapMapping`. Two access styles: flat
  (`rules.get("poly.width_min_um")`) and dict-section
  (`rules.poly["width_min_um"]`). Property `m0_is_m1` reflects whether
  the PDK collapses m0 onto m1.

- **Transistor primitive** (`transistor.py`) — `TransistorGeom`
  dataclass, `finger_count`, `sd_contact_columns`, `transistor_geom`
  (pure math) and `draw_transistor` (gdsfactory emitter). Returns a
  `gdsfactory.Component` with named `G`/`S`/`D` ports.

- **Via and tap cells** (`cells/`) — single-cut factories
  (`via_poly_m0`, `via_diff_m0`, `via_m0_m1`, `via_m1_m2`), composite
  stacks (`via_poly_m1`, `via_poly_m2`, `via_m0_m2`), and `draw_tap_cell`
  for substrate/n-well taps. All take a `BootstrapRules` and return a
  `gdsfactory.Component`.

- **Topology loader** (`synth/loader.py`) — `load_template(name_or_path,
  search_dirs=None)` → `CellTemplate`. Parses devices, nets, ports,
  three placement modes (standard / stacked / directives), routing
  hints, label layers, abutment, diffusion merges. All layer strings
  pass through `_normalize_layer` (`M1` → `m1`). Default search path:
  `packages/lithos-layout/templates/`; supply `search_dirs=[…]` to
  resolve template names from elsewhere. Templates themselves are not
  yet ported — see step 4 in the porting plan.

## Repos and references

- **Local clone**: `/home/shaheer/Documents/github/lithos`
- **Remote**: `github.com/ShaheerSajid/lithos`
- **Old code** (porting source): `/home/shaheer/Documents/github/layout_gen`
- **TSMC180 Calibre deck** (validation corpus): `/home/shaheer/Downloads/pdk`
- **RL stack pre-port** (in `layout_gen`, branch `drc-repair-engine`):
  see [docs/PORTING_PLAN.md](docs/PORTING_PLAN.md) section "RL phase
  status".

## Working norms

- **Environment setup is user-run**: I update `pyproject.toml` and hand
  the user a `uv sync` command; the user runs venv creation and big
  dep installs themselves.
- **One-line memory link in this file is enough**: deeper context
  belongs in `docs/`, not in `CLAUDE.md`.
