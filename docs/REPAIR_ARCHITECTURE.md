# Repair architecture — design plan

Status: design, not yet built. Captured from the 2026-05-18 design session
to guide the next implementation pass.

## The goal in one paragraph

A PDK-agnostic DRC repair system. Understanding is done upstream by an
LLM (rule comprehension into structured `FixMetadata`); the repair
algorithm itself learns *which fixes to apply where*, against rule
categories rather than specific PDKs. The system bootstraps via
synthetic data generation (polygon-diffusion-style — perturb a clean
GDS to manufacture a known violation, the inverse perturbation is the
ground-truth fix), refines via imitation learning from user-supplied
fix traces, and fine-tunes with the existing `layout_gen` RL stack
once a real training corpus exists.

## Three layers

### 1. Understanding (LLM + analyzer)

The "intelligence" of the system. Given a DRC violation, produce a
rich classification that turns "rule M2.S.1 fired at (x, y)" into
something the policy can act on.

* **Rule DB + FixMetadata** (lithos-core, built). Every rule code
  carries: category, usage_class, structured constraint AST, intent
  (one-sentence "why this rule exists"), allowed_action_classes,
  forbidden_action_classes, affected_layers, fix branches.
* **LLM extractor** (lithos-ingest, built). Produces FixMetadata from
  PDF / HTML / RST / CSV / deck-body. Local backend
  `FixMetadataExtractor.from_ollama("qwen2.5-coder:3b")` — no API key.
* **Context analyzer** (to build, in `lithos-repair`). Per-violation
  feature extraction from the GDS:
  - which polygons participate (often two, sometimes a polygon and a
    derived layer)
  - their device hierarchy (which transistor / which cell instance)
  - free space N/S/E/W
  - same-net vs different-net neighbours
  - power-rail / clock / signal classification
  - whether the polygon is on the routing grid
  - whether it's isolated or part of a regular array
* **Low-confidence escalation**: when the extractor's
  `confidence['fix_metadata']` is below threshold or context analysis
  is ambiguous, the violation re-enters the LLM, which acts directly
  on the geometry via the function library. The result is logged for
  training.

### 2. Action library (deterministic functions)

A closed vocabulary of verbs, each implemented as a deterministic
function with a typed parameter schema. Every verb has an **inverse**
— this is what makes synthetic data generation work.

Three granularities of action:

| Level | Examples | When it applies |
|---|---|---|
| **Polygon** | `widen`, `narrow`, `shift_*`, `extend`, `shrink`, `snap_to_grid`, `remove`, `redraw`, `merge`, `split` | Localized geometric edits |
| **Device** | `move_device`, `mirror_device`, `rotate_device` (drags all sub-polygons + triggers re-route over the moved terminals) | Cascading fixes — e.g. NW-to-PW spacing requires moving the transistor that sits in the well |
| **Region** | `move_region`, `density_fill`, `density_remove` | Bulk repairs over a rectangle — usually density rules or large-scale well repairs |
| **Routing** | `route`, `reroute`, `add_via`, `remove_via`, `change_layer` | Wraps existing deterministic routers (A* / maze / Lee) |

Routing in particular is **not** RL — it's deterministic, with 60
years of mature algorithms. RL picks *when* to route and *between
which terminals*; the router does the planning.

Each verb has a matching inverse (`widen ↔ narrow`, `shift_north ↔
shift_south`, `add_via ↔ remove_via`, etc.). Synthesised violations
re-use the same library with sign-flipped parameters.

### 3. Training (3 phases, increasing in cost)

Three phases planned for the learning component, layered:

* **Phase A — Synthetic (polygon diffusion).**
  Forward process `q(x_t | x_{t-1})`: apply a random inverse-fix
  action from the library. `x_0` = clean DRC-passing cell, `x_T` =
  maximally violated. Reverse process `p_θ(x_{t-1} | x_t)`: a trained
  model that predicts the inverse-fix to apply. Conditioning =
  netlist + rule DB. Math identical to pixel diffusion, state space
  is polygon-set instead of pixel-grid. Closer to D3PM /
  categorical-diffusion than score-based because actions are
  discrete-verb-with-parameters. Permutation-invariant encoder
  needed (set transformer or graph net).
* **Phase B — Imitation from LLM + user traces.**
  The fix-log table records every `(state, action, outcome)` from
  both LLM-driven and user-driven repair runs. BC-pretrain a small
  per-category policy that imitates these. Once it matches the
  source on a category, route violations of that category to the
  policy instead of the LLM. The user's offer to hand-fix examples
  is exactly this corpus.
* **Phase C — RL fine-tuning.**
  Re-purpose the `layout_gen` IBRL infrastructure (MaskablePPO with
  BC initialization). State now includes FixMetadata + context
  features rather than just raw geometry. Reward = DRC delta minus
  cost of secondary violations introduced. Action space is
  hierarchical (level + verb + parameters). This is what "most works
  use RL" means — but every working chip-layout RL result starts
  with imitation; we do the same.

## Inference loop

```
GDS  →  run DRC
        │
        ▼
   violation v
        │
        ▼
   [Understanding]
     - rule_db.get_rule(v.code)        → metadata, FixMetadata
     - context_analyzer.analyze(v, gds) → features
        │
        ▼
   [Decision]
     - if confidence(metadata, features) ≥ threshold:
         policy.pick(metadata, features)  → action(verb, params)
       else:
         llm.repair(v, gds)              → action(verb, params)
        │
        ▼
   [Execution]
     - function_library[verb](*params)
        │
        ▼
   record (state, action, outcome) to fix-log
        │
        ▼
   re-run DRC → loop until clean or budget exhausted
```

## Sequencing — next implementation artifacts

In order, each blocks the next:

1. **Context-feature schema** (`lithos-repair/lithos_repair/features.py`).
   YAML or Pydantic listing every geometric feature the analyzer
   extracts and what action choices each enables. Foundation for
   everything else.
2. **Action library v1** (`lithos-repair/lithos_repair/actions.py`).
   Closed vocabulary of polygon/device/region/routing verbs as
   typed Python functions. Each carries its inverse. Schema doubles
   as the LLM's `allowed_action_classes` vocabulary (constrain
   extraction to this set).
3. **Demo-recording harness**
   (`lithos-repair/lithos_repair/fix_log.py`). Captures every
   `(state, action, outcome)` triple from a repair run into a
   SQLite table next to the rule DB. Used by both Phase B and
   Phase C training.
4. **Polygon-diffusion training protocol**
   (`docs/POLYGON_DIFFUSION.md`). Detailed design: state encoder
   (set / graph), action noise schedule, conditioning vector,
   reverse-model architecture, dataset size targets. Phase A.
5. **Imitation baseline**
   (`lithos-repair/lithos_repair/policy.py` +
   `lithos-rl/lithos_rl/bc_trainer.py`). Per-category BC policies
   trained from the fix-log. Phase B.
6. **RL fine-tune harness**
   (port from `layout_gen/rl/training/ppo_train.py`). Phase C.

## Open decisions to settle on resume

* **Action vocabulary scope** — PDK-agnostic (one global vocabulary
  the LLM is locked to) or PDK-extensible (base + PDK-specific
  additions for exotic verbs like FinFET dummy fingers, MIMCAP
  redraw)? Trade simplicity vs reach.
* **Skill library form** — Python module with typed functions, or
  YAML spec the env interprets, or hybrid (Python functions +
  YAML/Pydantic registry naming each one)?
* **Fix-log location** — third table inside the rule DB (joinable
  against rule metadata at query time) or a separate journal file
  per repair run (easier to ship around / version with the cells
  under test)?
* **Phase A vs Phase B first** — full polygon-diffusion training as
  the bootstrap, or simpler imitation-from-LLM baseline first and
  revisit diffusion only if it hits a ceiling? Diffusion is the
  better long-term framing but heavier to engineer.

## What's already in place

Cross-references to existing pieces this builds on:

* `lithos-core` — rule DB schema, Constraint / CheckExpr / LayerExpr
  IR, FixMetadata, PDKMetadata, CategoryConfig.
* `lithos-ingest` — SVRF + KLayout-DRC parsers, multi-format
  loaders, code-anchored chunker, joiner, writer, CLI,
  `FixMetadataExtractor` with Ollama / llama.cpp / Anthropic
  backends.
* `lithos-drc` — DRCRunner interface + KLayout & Magic backends +
  alias resolver.
* `lithos-layout` — `BootstrapRules` (rule-DB ↔ cell-code bridge),
  `TransistorGeom` + `transistor_geom` (dimension math),
  `draw_transistor` (gdsfactory GDS emitter).
* `layout_gen/rl/` — existing IBRL infrastructure (env + policy +
  MaskablePPO + BC pretrain) to be ported into `lithos-rl` for
  Phase C.

## What's still stub

* `lithos-repair` — the home for everything in this document
  (context analyzer, action library, fix log, imitation policy).
* `lithos-rl` — RL training infrastructure; receives the
  `layout_gen` port when Phase C kicks off.
* `lithos-lvs` — netgen / magic LVS wrappers.
