# Porting plan — `layout_gen` → `lithos`

`layout_gen` is the prototype that proved the synthesis pipeline works.
`lithos` is the cleaned-up successor: same end goal (PDK-agnostic
generator from topology YAML) but restructured as a uv workspace with a
PDK-agnostic core (`lithos-core`/`lithos-ingest`/`lithos-drc`) and a
learned repair loop replacing the old hard-coded primitives.

This document is the single source of truth for *what's done*, *what's
left*, and *in what order*.

## Roadmap

Each step blocks the next unless noted otherwise.

| # | Item                                                              | Status     | Notes                                                                                                                                                          |
| - | ----------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Port `cells/standard.py` + `cells/vias.py` + `cells/tap.py`       | **done**   | Includes the project-wide M0/M1/M2 rename. See "Cells port" below.                                                                                             |
| 2 | Port `synth/loader.py` (topology YAML → typed specs; zero PDK dep) | **done**   | `LabelLayerSpec` fields renamed to `m1`/`m2` with `None` defaults; sky130 magic values removed.                                                                |
| 3 | Port `synth/placer.py` + `synth/router.py` + `synth/auto_router.py` (+ `synthesizer.py`, `netlist.py`, `port_resolver.py`, `constraints.py`, `euler.py`, `geo/`) | pending    | Heavy lift: placer ≈ 725 LoC, router ≈ 1652 LoC, auto_router ≈ 874 LoC, plus support files. Consumes the loader output and drives the cell factories.          |
| 4 | Port `templates/cells/*.yaml`                                     | pending    | 12 files (inverter, nand2/3, nor2/3, aoi21, oai21, bit_cell_6t, buffer, dido, row_driver, tap_cell). Rewrite layer refs to canonical `m0`/`m1`/`m2`/`contact`. Lands at `packages/lithos-layout/templates/cells/`. |
| 5 | Port `repair/` heuristic primitives into `lithos-repair`           | pending    | See [REPAIR_ARCHITECTURE.md](REPAIR_ARCHITECTURE.md) for the redesigned plan. Old code is heuristic; new design is LLM understanding + closed action vocab + learned policy. |
| 6 | Port `rl/` (env + policy + training) into `lithos-rl`              | pending    | Lives in `layout_gen` at commit `23cb778` on branch `drc-repair-engine`. See "RL phase status" below.                                                          |
| 7 | Port `lvs/` (netgen + magic extraction) into `lithos-lvs`          | pending    | Needed to close the loop on routing quality.                                                                                                                   |
| 8 | Bulk-LLM enrich the ~88 unstructured TSMC180 rules via Ollama      | optional   | Bumps the structured-constraint fraction above ~80%. Cosmetic but useful before any real-cell validation work.                                                 |

## Cells port (step 1, completed)

The cells port doubled as a project-wide rename that removed all
sky130-flavored layer names from `lithos-*` code. The rule is captured
in [CLAUDE.md](../CLAUDE.md#project-invariants) and is now an
**enforced invariant**: lithos code never writes `li1`, `met1`, `mcon`,
`licon1`, or `via1`. Per-PDK YAMLs are the only place those names live.

Files touched:

- `lithos_layout/cells/__init__.py` — new subpackage entry point.
- `lithos_layout/cells/standard.py` — port of `_sd_x` / `_gate_x` /
  `_diff_y` / `_inter_cell_gap` / `_routing_gap` / `_snap` / `_rect`.
- `lithos_layout/cells/vias.py` — port of the via factories under the
  new `via_<from>_<to>` naming.
- `lithos_layout/cells/tap.py` — port of `draw_tap_cell`. Dropped the
  YAML-template loader dependency (the inline defaults were sufficient)
  and the `RULES` global default (now caller-supplied).
- `lithos_layout/rules.py` — added `m0_is_m1` property; renamed
  semantic-key example block in the docstring.
- `lithos_layout/transistor.py` — switched every `contacts.*` →
  `contact.*`, `li1.*` → `m0.*`, `rules.layer("licon1")` →
  `rules.layer("contact")`, etc.
- `lithos_layout/__init__.py` — re-exports for the cell factories.

Tests:

- Existing `tests/test_bootstrap_rules.py`, `tests/test_transistor_geom.py`,
  `tests/test_draw_transistor.py` — rewritten fixtures using `m0` /
  `contact` / `nimplant` / `pimplant`.
- New `tests/test_cells_standard.py` — 9 tests for the geometry
  helpers.
- New `tests/test_cells_vias.py` — 8 tests covering all seven via
  factories plus the `m0_is_m1` collapse case.
- New `tests/test_cells_tap.py` — 7 tests including the `diff`-as-tap
  fallback path.

## Loader port (step 2, completed)

`lithos_layout/synth/loader.py` is a near-mechanical port of
`layout_gen/synth/loader.py` (≈ 620 → ≈ 480 LoC after dropping
sky130 defaults). Key adjustments:

- `_normalize_layer` now emits `m0`/`m1`/`m2` (was: `met1`/`met2`). Bare
  shorthand (`M0`, `m12`) is normalised; non-metal names are
  lowercased and passed through.
- `LabelLayerSpec`: fields renamed `met1` → `m1`, `met2` → `m2`;
  defaults changed from `(68, 5)` / `(69, 5)` (sky130) to `None`. The
  caller is responsible for filling these from PDK metadata.
- `load_template` gained a `search_dirs` argument; default search path
  now points at `packages/lithos-layout/templates/` (currently empty,
  awaiting step 4).
- All layer strings on `NetSpec`, `PortSpec`, and `RoutingHint` pass
  through `_normalize_layer` at parse time.
- Three placement modes preserved: `standard` (pairs section with
  optional relations), `stacked` (`row_pairs`), and `directives`
  (`placement_logic` list or bare `placement` list).

Tests: `tests/test_synth_loader.py` — 31 tests covering
`_normalize_layer`, device/net/port parsing, all three placement
modes, routing hints (dict + list forms), label layers, diffusion
merges, and search-dir resolution.

## RL phase status (deferred to step 6)

The RL stack referenced in step 6 currently lives in `layout_gen` on
branch `drc-repair-engine` (HEAD `23cb778` at the time of the cells
port). It is **Phase 4 complete**: the full pipeline runs end-to-end
(topology YAML → GNN → PLACE → ROUTE → REPAIR → real-DRC → GDS) and
the trainer wires up against a real cell with real klayout. The
remaining work is actual training + downstream artifacts (eval
metrics, LVS reward) — not more code.

When step 6 starts, the port targets `packages/lithos-rl/`. Useful
references inside `layout_gen` at that commit:

- `layout_gen/rl/policy.py` — `LayoutPolicy` + `MaskableLayoutPolicy`
  (poly/violation transformers + pointer-style target head).
- `layout_gen/rl/env.py` — gymnasium `LayoutEnv` with cached DRC, the
  `MultiDiscrete` action space, padded poly+violation observations.
- `layout_gen/rl/topology/{parser,encoder}.py` — bipartite GNN for
  conditioning on the topology graph.
- `layout_gen/rl/scripts/train_ppo.py`, `generate.py`,
  `inspect_gds.py` — the CLI entry points.
- `layout_gen/rl/tests/` — 82 unit tests.

Things still missing for high-quality RL output (track for step 6 +
ongoing):

1. **LVS reward** — ROUTE quality stagnates without a connectivity
   signal. Magic's LVS is the source; a `CachedLVS` analogue + a
   per-net penalty term would unblock progress.
2. **Per-net "all terminals reachable" oracle** — even without full
   LVS, a connected-union check on each net's segments would help.
3. **BC corpus for PLACE / ROUTE** — `mine_trajectories.py` currently
   only emits REPAIR primitives.
4. **Curriculum** — train repair-only, then add place, then route.
5. **Eval protocol** — a `scripts/eval.py` that reports DRC-clean
   rate, per-cluster-issue rate, mean reward over N episodes.

Decommission `synth/placer.py` + `synth/router.py` is **not** on the
plan: they stay as the rule-based baseline until RL reaches parity, and
parity-measurement infrastructure (step 5 above) doesn't exist yet.

## Conventions when working on this repo

- All new project work goes in `lithos/`, not `layout_gen/`. `layout_gen`
  is the porting reference and will be retired once steps 3 / 6 are
  done.
- Test-driven: every ported file gets a test in
  `packages/<pkg>/tests/`. The suite has to stay green
  (`uv run pytest -q` should report ≥ 251 passing as of the loader
  port; later steps will grow this number).
- Layer-naming compliance: any new file that mentions `li1`, `met1`,
  `mcon`, `licon1`, or `via1` outside a PDK YAML is a port mistake. See
  [CLAUDE.md](../CLAUDE.md#1-pdk-agnostic-metal-stack-naming).
- Read [REPAIR_ARCHITECTURE.md](REPAIR_ARCHITECTURE.md) before
  implementing anything in `lithos-repair` or changing the action
  vocabulary.
