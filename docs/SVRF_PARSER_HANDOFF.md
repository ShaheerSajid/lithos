# SVRF parser rewrite — handoff for delegated work

This document is a self-contained brief so another coding agent (Aider +
Groq, Continue + Ollama, Gemini + Cursor, etc.) can pick up the SVRF
parser rewrite without re-deriving context.

## Goal

Replace the current tolerant recursive-descent SVRF parser at
[packages/lithos-ingest/lithos_ingest/parsers/svrf.py](../packages/lithos-ingest/lithos_ingest/parsers/svrf.py)
with a **clean recursive-descent parser grounded in the Siemens SVRF
manual** (`~/Documents/pdk/svrf_ur.pdf`, v2022.3, 3574 pages).

**Hard constraints** (do not violate):

1. Keep the public API stable: `parse_svrf(src: str) -> list[ParsedRule]`.
   Callers (writer, joiner, tests) depend on `ParsedRule` from
   [packages/lithos-ingest/lithos_ingest/parsers/types.py](../packages/lithos-ingest/lithos_ingest/parsers/types.py).
2. All 406 existing tests must stay green throughout. Run
   `python -m pytest -q` after every meaningful change.
3. No new runtime dependencies (no lark, no pyparsing, no ANTLR). Pure
   stdlib. The project ships as an open-source uv workspace.
4. No vendor-specific references in checked-in code (no foundry names,
   no document numbers, no deck file names). The manual is fine to
   cite by section title and page number.

**Success metric**: structured-constraint coverage *on the major rule
sets the cell generator needs* (poly / diffusion / well / metal / via)
reaches ≥ 95%. Today: 180nm averages 91%, 65nm averages 95%. Edge-case
operators (BRANCH chains, RECTANGLE shape checks, etc.) can stay
no-branch and be resolved via LLM during DRC iteration — they are not
load-bearing for bootstrap GDS generation.

The validation corpora are local-only to the original author's machine
(foundry IP) — paths stored in agent memory, not in the repo. The
measurement script below expects them as environment variables:

```python
import os
from lithos_ingest.parsers.svrf import parse_svrf
for path in (os.environ["LITHOS_SVRF_180NM"], os.environ["LITHOS_SVRF_65NM"]):
    rules = parse_svrf(open(path).read())
    nb = sum(1 for r in rules if not r.constraint.branches)
    print(path, len(rules), nb, f"{100*nb/len(rules):.1f}% no-branch")
```

## Current state (as of this handoff)

- 406 tests passing across the workspace.
- 180nm: 708 rules parsed, **12.6%** no-branch overall (majors at
  90-100% structured).
- 65nm: 2314 rules parsed, **7.6%** no-branch overall (majors at
  90-100% structured).
- Parser is principled RD: one method per production, formal grammar
  documented at the top of the module.
- Already supported (see module docstring for the grammar):
  - Tokenizer with `IDENT`, `NUMBER`, `STRING` (single + double
    quote), comparators (incl. `==`), braces, brackets, math
    operators (`+ - * / ^ %`, `! ~`), `@` description lines,
    `#` directive lines, `//` + `/* */` comments. Digit-prefixed
    layer / variable names (manual page 67) handled by post-NUMBER
    glue.
  - `VARIABLE` / `#DEFINE` numeric symbol table for threshold
    resolution (e.g. `EXT a b < NW_S_5` resolves `NW_S_5` to its
    declared numeric value).
  - **Full numeric expression parser** (manual Table 2-3): operator
    precedence `() > unary > * / ^ % > + -`, math functions from
    Table 2-4 (`CEIL` / `FLOOR` / `TRUNC` / `SQRT` / `ABS` / `EXP` /
    `LOG` / `SIN` / `COS` / `TAN` / `MIN` / `MAX`). Variable
    identifiers resolve through the symbol table.
  - **Interval constraints** (manual Table 2-2): all 8 interval
    shapes (`> a < b`, `<= b >= a`, …) are parsed; for non-Density
    checks the lower bound becomes the structured threshold.
  - Rule blocks: `NAME { ... }` and `NAME:SUFFIX { ... }` (colon-glued
    multi-component names) and `RULECHECK "name" { ... }`.
  - Layer assignment: `IDENT = layer_expr`.
  - Check verbs: `EXT` / `INT` / `ENC` / `WIDTH` / `LENGTH` / `AREA`
    (single-threshold form), `ANGLE`, `OFFGRID`, `DENSITY` (simple
    form), `ENCLOSE`, `COPY`.
  - Layer expression operators: `AND`, `OR`, `NOT` (in expressions),
    `INSIDE`, `OUTSIDE`, `INTERACT`, `TOUCH`, `ENCLOSE`, `COVERS`,
    `SIZE BY <num>`.
  - Modifier soak-up: `ABUT < N`, `SINGULAR`, `REGION`, `OPPOSITE`,
    `PARALLEL`, `PROJECTING`, `NOTCH`, etc.
  - **Last-assigned-layer fallback**: when a rule body is a chain
    of layer assignments with no explicit check operator, the final
    derived layer is promoted to an `ExistenceCheck`. Recovers
    via-stack `BRANCH1` / `GoodBranch` / `BAD_REGION` style rules.
  - **Process-variant dup merge**: when the writer sees a duplicate
    rule code, it appends the new constraint's branches to the
    existing rule's `Constraint.branches` instead of dropping
    (predicate stays empty until Phase 5).

## What the manual says (and we don't do)

Each item below cites the manual page(s).

### Phase 1 — Tokenizer / lexer hardening (manual pages 65-77) — **partial**

Done in-thread: math operators (`+ - * / ^ %`, `! ~`), `==`, `[ ]`,
block comments (`/* */`), single-quoted strings collapsed onto STRING
token, digit-prefixed name handling. Still open:

- **Case-insensitive keywords** (page 67): verify every comparison
  is case-folded. Currently we mostly `.upper()` IDENTs at compare
  time; audit and consolidate.
- **Quoted string names override case-sensitivity rules** (page 74).
  `'523.4' { EXTERNAL < 'space' 'INSIDE LAYER' }` is valid — the
  quoted strings are names, not string literals. Parser still treats
  STRING tokens as string constants only; needs to accept them in
  name position as a `LayerRef`.

### Phase 2 — Numeric expression parser (manual pages 72-73) — **DONE**

Landed in-thread. `_parse_numeric_expression` implements the full
precedence climb (`() > unary > * / ^ % > + -`) and the math
function table. Verified on real-deck thresholds:
`< 3 * (GRID + 6)`, `< OD_W_1 * OD_L_2`, `< DCO_R_5 + 0.002`,
`< MAX(A, B)`.

### Phase 3 — Interval constraints (manual page 70, Table 2-2) — **partial**

`_parse_interval_constraint` exists and consumes the 8 interval
shapes correctly. Non-Density checks currently keep only the lower
bound's value as the structured threshold; richer IR (carrying both
bounds) is open work.

### Phase 4 — Flexible layer-operation syntax (manual page 68)

The manual is explicit: **syntax elements of an operation can appear
in any order**. The six AREA forms are all equivalent:

```
AREA < 4 contact         contact AREA < 4       AREA contact < 4
< 4 AREA contact         < 4 contact AREA       contact < 4 AREA
```

Currently our parser assumes a fixed positional order
(`KEYWORD layer cmp num modifiers`). Refactor each check parser into a
**token-bag** approach: collect the layer reference, comparator,
threshold, and modifiers regardless of order. Multi-word keywords
(`NOT INSIDE`, etc.) must NOT be reordered.

### Phase 5 — `#IFDEF` state tracking (manual pages 84-99)

Today we drop conditional-block info entirely. We've added a
"duplicate codes merged into variant branches" path in the writer
([packages/lithos-ingest/lithos_ingest/writer.py](../packages/lithos-ingest/lithos_ingest/writer.py))
but the `predicate` field on each ConstraintBranch stays empty.

Manual semantics:

- `#IFDEF NAME` / `#IFNDEF NAME` push a condition.
- `#ELSE` flips the top-of-stack.
- `#ENDIF` pops.
- `#DEFINE NAME` (no value) creates a boolean compile-flag.
- `#DEFINE NAME value` declares a (typed) macro.

Track the stack as the lexer/parser walks the file. Each `ParsedRule`
gets a `predicate: list[str]` (or stronger typed model) recording the
active conditions at its location. The writer's
`_merge_duplicate_rule` should populate `ConstraintBranch.predicate`
from these.

### Phase 6 — Macros and `#PRAGMA ENV` (manual pages 84-106)

Lower priority. Implement only if the coverage gap demands it.

- `#DEFINE NAME(args) body` — function-style macros.
- `#PRAGMA ENV NAME default` — env-variable defaults.
- `INCLUDE "path"` statements — resolve included files relative to
  the deck path.

### Phase 7 — Comprehensive operator coverage

The keyword-name-conflict list (Table 2-5, page 77) is the canonical
operator vocabulary — ~130 entries. We support roughly 20. Sample
what's missing on real decks (the 13%/17% no-branch tail), add
recognisers in priority order.

Top candidates from the conflict list:

- `Connect` / `Disconnect` — connectivity, page 183.
- `Convex Edge` — angle / shape — page 189.
- `Cut` — page 197.
- `Antenna` — page 174 (search the manual for the full operator).
- `Net Area Ratio (NAR)` — antenna-style.
- `Group` / `Rectangles` / `Path` — shape selectors.
- `With Width` / `With Length` — measurement filters.
- `Coincident Edge` / `Coincident Inside Edge` / `Coincident Outside
  Edge` — pages 180-182.

## Recommended sequence

Aim for one phase per session. After each phase: run tests, measure
no-branch coverage on the two corpora, commit.

| phase | manual chapter | est. session count | coverage gain est. |
|-------|----------------|--------------------|---------------------|
| 1 — tokenizer       | Ch. 2 (pp 65-77)  | 0.5 | small (correctness) |
| 2 — numeric exprs   | Ch. 2 (pp 72-73)  | 1   | medium-large (~3-5%) |
| 3 — interval consts | Ch. 2 (p 70)      | 0.5 | small-medium (~2%) |
| 4 — flexible order  | Ch. 2 (p 68)      | 1.5 | medium (~3%) |
| 5 — #IFDEF state    | Ch. 2 (pp 84-99)  | 1   | quality (variant branches) |
| 6 — macros          | Ch. 2 (pp 100-106)| 1   | small (real decks rare) |
| 7 — operators       | Ch. 3 + Ch. 4+    | many| largest (~5%+) |

## Repository pointers

- Parser: [packages/lithos-ingest/lithos_ingest/parsers/svrf.py](../packages/lithos-ingest/lithos_ingest/parsers/svrf.py)
- Parsed-rule schema: [packages/lithos-ingest/lithos_ingest/parsers/types.py](../packages/lithos-ingest/lithos_ingest/parsers/types.py)
- Writer / dup-merge: [packages/lithos-ingest/lithos_ingest/writer.py](../packages/lithos-ingest/lithos_ingest/writer.py)
- Constraint IR (target of parser output): [packages/lithos-core/lithos_core/ir.py](../packages/lithos-core/lithos_core/ir.py)
- Existing tests: [packages/lithos-ingest/tests/test_svrf.py](../packages/lithos-ingest/tests/test_svrf.py)
- Validation corpora: see memory file `validation_corpora.md` (paths
  local-only; not in repo).

## Working agreements

- Test-driven: every parser change adds at least one test pinning the
  new behaviour, plus the regression check on a real-deck rule when
  available.
- No new runtime deps; tooling deps (Aider/Continue/etc.) are fine.
- Open-source hygiene: no foundry-specific identifiers in code or
  tests. The validation corpora are local-only.
- Commit boundary: one phase per commit, descriptive title, before/after
  no-branch counts in the body.

## Communication back to the user

If you (the delegated agent) need a decision on scope, ambiguity, or
trade-off, dump the question to `docs/SVRF_PARSER_NOTES.md` with a
heading dated YYYY-MM-DD — the user reads that file between sessions.
Don't block on missing answers; pick the option more consistent with
the manual and note the assumption in the same file.
