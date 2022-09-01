# lithos-repair — implementation plan

Status: **plan to execute**. Updated 2026-05-20 after the TSMC180 real-PDK
validation pass surfaced 1044 violations across 12 cells in ~6 rule families
— small enough to drive a concrete imitation-first bootstrap.

The longer-term architectural framing lives in
[REPAIR_ARCHITECTURE.md](REPAIR_ARCHITECTURE.md). This file is the
short-term work plan: file-by-file, milestone-by-milestone, with
acceptance criteria per step.

## Goal

Close the loop end-to-end:

    synthesize cell → DRC → parse violations → LLM proposes fix
                          → apply fix → re-DRC → iterate

…for **one rule family at a time**, starting with implant enclosure
(the biggest TSMC180 family, ~270 violations). Each milestone is a
useful, shippable step that incrementally automates more of what we
did manually today.

## Decisions (locked in based on today's evidence)

| open question | decision | why |
|---|---|---|
| Phase A (diffusion) vs Phase B (imitation) first | **Phase B first** | 1044 violations × 6 families = small enough corpus to bootstrap from a single hand-fix round. Diffusion stays a Phase C option once we hit a Phase B ceiling. |
| Action vocabulary scope | **PDK-agnostic v1** | TSMC180 vs sky130 violations need the same verbs (widen implant, shift contact, snap to grid) with different *parameters*. Per-PDK extension is a YAGNI. |
| Skill library form | **Python typed functions + Pydantic registry** | Functions are unit-testable; the registry doubles as the LLM's `allowed_action_classes` vocabulary. |
| Fix-log location | **Third table in the rule DB** | Lets us `JOIN rule_alias` to recover canonical codes across tools (KLayout / Calibre / Magic emit different names for the same rule). |
| First rule family to automate | **Implant enclosure (PP.E.1 / NP.E.1)** | Biggest count (~270), simplest geometric fix (widen implant box), already-mapped semantic key (`implant.enclosure_of_diff_um`). |

## Milestones

Each milestone is independently mergeable + testable. Estimates are
rough; the unit is "focused sessions" not calendar days.

### M1 — Synthesizer reads `implant.enclosure_of_diff_um` from rules

**1 session.** The smallest end-to-end win: fix one rule family by
plumbing one bootstrap key, no agent involved yet. Validates the
hypothesis that "real PDKs surface real bugs the synthesizer was
already designed to handle but never wired up".

* Add `implant.enclosure_of_diff_um` to `pdks/tsmc180/bootstrap.yaml`
  (mapped to `NP.E.1` — threshold 0.18).
* Add the same to `pdks/sky130A/bootstrap.yaml` (existing key, 0.125).
* Change `lithos_layout/synth/synthesizer.py::_merge_implants` to
  read the value via `rules.implant.get("enclosure_of_diff_um",
  0.125)` instead of the hardcoded 0.125.
* **Acceptance**: TSMC180 implant violations drop from ~270 to near
  0; sky130 results unchanged (default already matched). No code
  regressions.

### M2 — Action library v1 + feature schema

**2 sessions.** Foundation for everything downstream. No agent yet.

* `packages/lithos-repair/lithos_repair/features.py` — Pydantic
  `ViolationContext`: which polygons participate, device hierarchy
  (resolve via `gdsfactory` cell tree), free space N/S/E/W,
  same-net vs different-net neighbours, on-grid flag, isolated /
  array, layer enrichment.
* `packages/lithos-repair/lithos_repair/actions.py` — typed Python
  functions for the v1 verb list (start with polygon-level only):
  `widen`, `narrow`, `shift_n/s/e/w`, `extend`, `shrink`,
  `snap_to_grid`, `remove`, `redraw`. Each function:
  - signature: `(gds: gf.Component, target: PolygonRef, params: dict) -> gf.Component`
  - inverse declared as a metadata attribute
  - parameter Pydantic model
* `packages/lithos-repair/lithos_repair/registry.py` — Pydantic
  registry that names each action + exposes its JSON schema (this
  becomes the LLM's grammar later).
* `packages/lithos-repair/tests/` — round-trip test per verb
  (`apply(v) ∘ apply(v.inverse) == identity` within the mfg grid).
* **Acceptance**: every verb has a unit test; verbs can be invoked
  directly in a REPL on a real `SynthResult.component`.

### M3 — Context analyzer

**1 session.** Wraps M2's feature schema with the extraction logic.

* `packages/lithos-repair/lithos_repair/analyzer.py` —
  `analyze(violation: DRCViolation, comp: gf.Component, rules:
  BootstrapRules) -> ViolationContext`.
* Locates the violating polygon(s) using `(x, y)` proximity + layer.
* Walks the cell tree to find the parent device/instance.
* Looks up the `FixMetadata` for the rule via the rule DB if
  available (sky130 / TSMC180 don't have these populated yet — that's
  fine, the analyzer returns an empty hint).
* **Acceptance**: a single Python call returns a populated
  `ViolationContext` for any of the 1044 TSMC180 violations.

### M4 — Fix-log table

**0.5 session.** Required for both imitation (M6) and RL (later).

* Schema in `lithos_repair/fix_log.py`: SQLite table in the same DB
  as the rule data, columns `(timestamp, pdk, cell, rule_code,
  violation_xy, context_json, action_verb, params_json, outcome,
  pre_count, post_count, source)`.
* Source `enum` differentiates `manual`, `llm`, `policy`, `rl`.
* `record(...)` API for the repair loop to call after every applied
  action.
* **Acceptance**: a hand-driven repair session populates the table;
  rows are queryable via `JOIN rule_alias`.

### M5 — LLM repair agent (Anthropic + Ollama backends)

**2 sessions.** First closed loop.

* `lithos_repair/agent.py` — `LLMRepairAgent`. Reuses the
  `FixMetadataExtractor` backend abstraction from `lithos-ingest`.
* Single prompt: given `(rule_db_row, FixMetadata if any,
  ViolationContext, current cell parameters)`, emit a JSON
  `{verb, params}` from the M2 registry's grammar.
* `lithos_repair/loop.py` — `repair_cell(name, rules, max_iter=8,
  agent=...) -> RepairTrace`. Wraps `synthesize_cell` + DRC + agent +
  apply + re-DRC. Logs every step via M4.
* CLI: `python -m lithos_repair.cli inverter --pdk tsmc180 --max-iter 8`.
* **Acceptance**: a fresh `inverter` cell synthesises against
  TSMC180, the loop applies fixes, and TSMC180 violation count drops
  monotonically (no requirement to reach 0 — that's M7+).

### M6 — Imitation baseline (Phase B)

**3 sessions.** Once the fix-log has a few hundred LLM-driven traces,
we can BC-pretrain a per-category policy that doesn't need the LLM.

* `lithos_repair/policy.py` — small MLP / set-transformer over
  `ViolationContext` features → action distribution. One head per
  rule category (start: 6 categories from the TSMC180 top families).
* `lithos_rl/bc_trainer.py` — pulls `(context, action)` tuples from
  the fix-log, trains until BC validation loss converges.
* Decision logic in M5's loop: route violations whose category has a
  trained policy to the policy; everything else stays on the LLM.
* **Acceptance**: implant-enclosure category routes off the LLM
  entirely. Latency drops; coverage holds.

### M7 — RL fine-tune (Phase C)

**4+ sessions.** Port `layout_gen@drc-repair-engine` (commit
`23cb778` — see [PORTING_PLAN.md](PORTING_PLAN.md) §"RL phase").

* `lithos_rl/env.py`: state = `ViolationContext` + cell snapshot;
  action = (level, verb, params); reward = `pre_violations - post_violations
  - 0.3 * new_violations - 0.05 * step_cost`.
* `lithos_rl/ppo_train.py`: port from `layout_gen/rl/training/ppo_train.py`,
  with BC-init from M6.
* **Acceptance**: starting from the M6 policy, RL recovers an
  additional ≥30 % of unresolved violations on held-out cells.

### M8 — Polygon-diffusion (Phase A, optional)

**Open-ended.** Only revisit if M6+M7 plateau before clean cells.
See `docs/POLYGON_DIFFUSION.md` (TODO — write at the time, not now).

## File layout (target)

```
packages/lithos-repair/
├── lithos_repair/
│   ├── __init__.py
│   ├── features.py     # M2: Pydantic ViolationContext + co.
│   ├── actions.py      # M2: typed verb functions + inverses
│   ├── registry.py     # M2: action registry, LLM grammar source
│   ├── analyzer.py     # M3: violation → ViolationContext
│   ├── fix_log.py      # M4: SQLite recording
│   ├── agent.py        # M5: LLM-backed repair agent
│   ├── loop.py         # M5: repair_cell orchestration
│   ├── policy.py       # M6: per-category BC policy
│   └── cli.py          # M5: command-line entry point
├── tests/
└── pyproject.toml      # already exists (stub package)
```

## Notes

* M1–M4 don't need an LLM. They're pure refactor + plumbing. Useful
  even if M5+ never lands.
* The repair loop's **same input shape works for KLayout, Calibre,
  and Magic** — the runners already share `DRCViolation`. Means M5
  works against sky130 (KLayout) and TSMC180 (Calibre) with no
  per-tool branching.
* When a category's policy in M6 is mature enough, the LLM is
  retained as a fallback for low-confidence cases (per the doc's
  "low-confidence escalation" loop). Don't remove the LLM path even
  when policies cover most categories.
* The **action vocabulary IS the LLM grammar**. Whenever we add a
  new verb in M2, the LLM in M5 gets it for free.

## Quick reference — what's already in place

This plan builds on, doesn't replace:

* `BootstrapRules` (rule DB + bootstrap mapping bridge) — feeds M3.
* `SynthResult.violations` — feeds M3, M4, M5.
* `CalibreDRCRunner` / `KLayoutDRCRunner` — feed M5's re-DRC step.
* `FixMetadataExtractor` (LLM backend abstraction) — feeds M5's agent.
* `lithos_core.Rule` / `FixMetadata` IR — feeds M3, M5.

## Open questions for resume

If new evidence supersedes one of the locked decisions above, log it
here before changing course:

* (none yet)
