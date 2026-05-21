"""lithos_repair.cli — module entry point for the closed-loop repair.

Invocable as::

    python -m lithos_repair inverter --pdk tsmc180 --max-iter 8

Loads the PDK bootstrap + metadata from ``pdks/<pdk>/``, opens (or
ingests) the rule DB, builds a DRC runner, wires up an
:class:`LLMRepairAgent`, and runs :func:`repair_cell`. Prints a per-step
trace and a one-line summary, then writes the final GDS.

Designed to mirror :file:`scripts/validate_*.py` — the inputs are the
same (PDK bootstrap + deck), the difference is the loop and the agent.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .agent   import AgentConfig, LLMRepairAgent
from .fix_log import FixLog, FixSource
from .loop    import repair_cell


# ── PDK + DB wiring ─────────────────────────────────────────────────────

def _build_rules(pdk_dir: Path, db_dir: Path):
    """Load metadata + bootstrap; ingest the SVRF deck into a fresh DB.

    Mirrors the validate scripts: each run starts from a freshly-ingested
    DB so changes to the deck or the parser flow into the next loop.
    """
    from lithos_core.metadata import load_metadata
    from lithos_core.db       import RuleDB
    from lithos_ingest        import svrf_to_db
    from lithos_layout        import BootstrapRules, load_bootstrap_mapping

    metadata = load_metadata(pdk_dir / "metadata.yaml")
    mapping  = load_bootstrap_mapping(pdk_dir / "bootstrap.yaml")

    deck_path = _deck_path(metadata)
    db_path = db_dir / f"rules.{metadata.name}.db"
    if db_path.exists():
        db_path.unlink()
    n = svrf_to_db(deck_path, db_path,
                   pdk_name=metadata.name, pdk_version=metadata.version)
    print(f"[lithos-repair] parsed {n} rules from {deck_path.name}")

    db = RuleDB(db_path)
    db.open()
    return BootstrapRules(metadata, db, mapping), db_path


def _deck_path(metadata):
    """Pick a deck path from metadata.drc_decks, preferring Calibre."""
    for tool in ("calibre", "klayout", "magic"):
        p = metadata.drc_decks.get(tool)
        if p is None:
            continue
        if "${" in str(p):
            continue
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No usable DRC deck in metadata.drc_decks. "
        "Set $TSMC180_DECK / $PDK_ROOT or point bootstrap to a real file."
    )


def _build_drc_runner(metadata, tool: str):
    """Construct the requested backend; raise if its binary is missing."""
    if tool == "calibre":
        from lithos_drc import CalibreDRCRunner
        runner = CalibreDRCRunner(metadata)
    elif tool == "klayout":
        from lithos_drc import KLayoutDRCRunner
        runner = KLayoutDRCRunner(metadata)
    elif tool == "magic":
        from lithos_drc import MagicDRCRunner
        runner = MagicDRCRunner(metadata)
    else:
        raise SystemExit(f"Unknown DRC tool: {tool}")
    if not runner.is_available():
        raise SystemExit(
            f"{tool} not available in this environment. "
            f"Install / configure the backend before running the loop."
        )
    return runner


def _build_agent(backend: str, model: Optional[str]) -> LLMRepairAgent:
    if backend == "anthropic":
        kwargs = {"model": model} if model else {}
        return LLMRepairAgent.from_anthropic(**kwargs)
    if backend == "ollama":
        kwargs = {"model": model} if model else {}
        return LLMRepairAgent.from_ollama(**kwargs)
    raise SystemExit(f"Unknown agent backend: {backend}")


# ── Trace pretty-printer ────────────────────────────────────────────────

def _print_step(step) -> None:
    arrow = "→"
    proposal_s = (
        f"{step.proposal.verb}({step.proposal.params.model_dump()})"
        if step.proposal is not None else "(no proposal)"
    )
    err = f" err={step.error!r}" if step.error else ""
    print(
        f"  step {step.iteration:>2d}: "
        f"{step.pre_count:>4d} {arrow} {step.post_count:<4d}  "
        f"{step.outcome.value:<8s}  rule={step.violation.rule!r:<12s}  "
        f"{proposal_s}{err}"
    )


# ── Main ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template", help="Template name (inverter, nand2, ...).")
    ap.add_argument("--pdk", required=True,
                    help="PDK directory under ./pdks/<pdk>/.")
    ap.add_argument("--drc-tool", choices=("calibre", "klayout", "magic"),
                    default="calibre",
                    help="Which DRC backend to use (default: calibre).")
    ap.add_argument("--agent",   choices=("anthropic", "ollama"),
                    default="anthropic")
    ap.add_argument("--model",   default=None,
                    help="Model identifier passed to the agent backend.")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--w", type=float, default=0.52)
    ap.add_argument("--l", type=float, default=0.18)
    ap.add_argument("--out", type=Path, default=Path("/tmp/lithos_repair"),
                    help="Output directory for the repaired GDS + fix log.")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    pdk_dir = Path("pdks") / args.pdk
    if not pdk_dir.is_dir():
        raise SystemExit(f"PDK directory not found: {pdk_dir}")

    rules, db_path = _build_rules(pdk_dir, args.out)
    runner = _build_drc_runner(rules.metadata, args.drc_tool)
    agent  = _build_agent(args.agent, args.model)

    print(f"lithos-repair: template={args.template}  pdk={rules.metadata.name}  "
          f"drc={args.drc_tool}  agent={args.agent}  max_iter={args.max_iter}")

    with FixLog(db_path) as fix_log:
        trace = repair_cell(
            args.template, rules,
            drc_runner = runner,
            agent      = agent,
            max_iter   = args.max_iter,
            params     = {"w": args.w, "l": args.l},
            fix_log    = fix_log,
            source     = FixSource.LLM,
        )

        for step in trace.steps:
            _print_step(step)

        gds_path = args.out / f"{args.template}.repaired.gds"
        if trace.component is not None:
            trace.component.write_gds(str(gds_path))
            print(f"\nrepaired GDS: {gds_path}")
        print(trace.summary())

    return 0 if (trace.converged or trace.initial_count > trace.final_count) else 1


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main(sys.argv[1:]))
