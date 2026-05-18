# lithos

PDK-agnostic layout generator. For a given PDK release, an LLM ingests the
foundry rule manual and DRC deck once into a structured rule DB. The
generator and the DRC-repair engine then read that DB at runtime — they
never re-read the PDF.

## Workspace layout

```
lithos/
├── packages/
│   ├── lithos-core/      typed IR, rule DB, PDK metadata. Bottom of the stack.
│   ├── lithos-ingest/    PDF chunker, deck parsers, LLM extractor, DB writer.
│   ├── lithos-drc/       tool-agnostic DRC runner + KLayout/Magic backends.
│   ├── lithos-layout/    cells, primitives, synth, generator, templates.
│   ├── lithos-repair/    heuristic repair primitives and fix-graph engine.
│   ├── lithos-rl/        RL env, policy, BC/PPO/IBRL training.
│   └── lithos-lvs/       LVS (netgen/magic) wrappers.
└── tests/                cross-package integration tests.
```

Each package is independently installable. `lithos-core` is the only required
dependency for any other package; the rest fan out from there.

## Quickstart

```bash
# One-time: install uv (https://docs.astral.sh/uv/) if you don't have it.
# Then, from the workspace root:

uv venv
uv sync                              # installs all workspace packages + dev deps

uv run pytest                        # runs every package's test suite
```
