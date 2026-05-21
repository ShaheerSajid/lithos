"""lithos_repair.loop — closed-loop synthesize → DRC → repair driver.

Top-level entry point that ties M2 (actions) + M3 (analyzer) + M4
(fix-log) + M5 (agent) into one call::

    trace = repair_cell(
        "inverter", rules,
        drc_runner = CalibreDRCRunner(rules.metadata),
        agent      = LLMRepairAgent.from_anthropic(),
        max_iter   = 8,
        fix_log    = FixLog("rules.db"),
    )

Each iteration:

1. Synthesize the cell (first iter) **or** re-DRC the in-progress
   component (subsequent iters).
2. Pick the first violation. (M6+ will route by category; for v1 we
   take whichever the backend reported first.)
3. Run :func:`~lithos_repair.analyzer.analyze` to build a
   :class:`ViolationContext`.
4. Ask the agent for a proposal; apply it via the registry.
5. Re-DRC the modified component to measure the post-action count.
6. Log the step via :class:`FixLog`.

Loop terminates when:

* No violations remain (``converged=True``), or
* ``max_iter`` reached, or
* Two consecutive iterations failed to apply a proposal
  (``stagnant=True``).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .agent      import AgentProposalError, LLMRepairAgent, ProposedAction
from .analyzer   import analyze
from .features   import ViolationContext
from .fix_log    import FixLog, FixOutcome, FixSource
from .registry   import REGISTRY, ActionRegistry

if TYPE_CHECKING:                                    # pragma: no cover
    from lithos_drc    import DRCRunner, DRCViolation
    from lithos_layout import BootstrapRules


# ── Trace dataclasses ───────────────────────────────────────────────────

@dataclass
class RepairStep:
    """One iteration of the repair loop."""
    iteration:    int
    violation:    "DRCViolation"
    context:      ViolationContext
    proposal:     Optional[ProposedAction]
    outcome:      FixOutcome
    pre_count:    int
    post_count:   int
    error:        Optional[str] = None
    """When ``outcome`` is ``failed``, a short human-readable reason."""

    @property
    def delta(self) -> int:
        """Negative when the step reduced violations; positive when worse."""
        return self.post_count - self.pre_count


@dataclass
class RepairTrace:
    """End-to-end record of one :func:`repair_cell` invocation."""
    template:      str
    pdk:           str
    initial_count: int
    final_count:   int
    steps:         list[RepairStep] = field(default_factory=list)
    component:     Any = None
    converged:     bool = False
    stagnant:      bool = False
    """True when the loop exited because two consecutive applies failed."""

    @property
    def iterations(self) -> int:
        return len(self.steps)

    def summary(self) -> str:
        """Single-line human-readable summary, useful for the CLI / logs."""
        status = (
            "clean" if self.converged
            else "stagnant" if self.stagnant
            else f"capped at {self.iterations}"
        )
        return (
            f"{self.template} ({self.pdk}): "
            f"{self.initial_count} → {self.final_count} violations "
            f"after {self.iterations} steps [{status}]"
        )


# ── Loop ────────────────────────────────────────────────────────────────

def repair_cell(
    template_name: str,
    rules:         "BootstrapRules",
    *,
    drc_runner:    "DRCRunner",
    agent:         LLMRepairAgent,
    max_iter:      int = 8,
    params:        Optional[dict[str, Any]] = None,
    fix_log:       Optional[FixLog] = None,
    source:        FixSource = FixSource.LLM,
    registry:      ActionRegistry = REGISTRY,
    search_dirs:   Optional[list[Path]] = None,
) -> RepairTrace:
    """Synthesise ``template_name`` and run the repair loop end-to-end.

    Parameters
    ----------
    template_name
        Template name resolved via :func:`lithos_layout.load_template`.
    rules
        :class:`BootstrapRules` — supplies the PDK metadata and grid.
    drc_runner
        Real DRC backend (Calibre or KLayout). The loop calls it once
        per iteration; the M5 acceptance criterion is monotonic
        violation-count decrease against TSMC180 via Calibre.
    agent
        :class:`LLMRepairAgent` that proposes one fix per iteration.
    max_iter
        Hard cap on iterations. The plan's default is 8.
    params
        Device sizing forwarded to :func:`synthesize_cell`.
    fix_log
        Optional :class:`FixLog` — every step is recorded when supplied.
    source
        Stamped onto each fix-log row. Default ``FixSource.LLM`` since
        this is the LLM-driven loop; tests use ``FixSource.MANUAL`` for
        scripted runs.
    registry
        Action registry. Defaults to the package singleton.
    search_dirs
        Extra template search directories. Passed to
        :func:`synthesize_cell`.
    """
    from lithos_layout import synthesize_cell

    pdk_name = rules.metadata.name

    # ── Initial synthesis + DRC ──────────────────────────────────────────
    result = synthesize_cell(
        template_name, rules, params,
        search_dirs=search_dirs,
        drc_runner=drc_runner,
    )
    comp       = result.component
    violations = list(result.violations)
    initial_count = len(violations)

    trace = RepairTrace(
        template      = template_name,
        pdk           = pdk_name,
        initial_count = initial_count,
        final_count   = initial_count,
        component     = comp,
    )

    if initial_count == 0:
        trace.converged = True
        return trace

    # ── Iterate ──────────────────────────────────────────────────────────
    consecutive_failures = 0

    for i in range(max_iter):
        if not violations:
            trace.converged = True
            break

        v   = violations[0]
        ctx = analyze(v, comp, rules)
        pre_count = len(violations)

        # Ask the agent.
        proposal: Optional[ProposedAction] = None
        outcome  = FixOutcome.SKIPPED
        error    = None
        try:
            proposal = agent.propose(ctx)
        except AgentProposalError as exc:
            outcome = FixOutcome.FAILED
            error   = str(exc)

        # Apply the proposal.
        if proposal is not None:
            try:
                comp, _ = registry.apply(
                    proposal.verb, comp, ctx.primary_ref, proposal.params,
                )
                outcome = FixOutcome.APPLIED
            except Exception as exc:                 # noqa: BLE001 — action raises ValueError on bad params
                outcome = FixOutcome.FAILED
                error   = str(exc)

        # Re-DRC.
        violations = _run_drc(comp, drc_runner)
        post_count = len(violations)

        step = RepairStep(
            iteration  = i,
            violation  = v,
            context    = ctx,
            proposal   = proposal,
            outcome    = outcome,
            pre_count  = pre_count,
            post_count = post_count,
            error      = error,
        )
        trace.steps.append(step)

        if fix_log is not None:
            _record(fix_log, ctx, v, proposal, outcome, source,
                    pdk_name, template_name, pre_count, post_count)

        # Stagnation detection: two consecutive failed applies + no count change.
        if outcome == FixOutcome.FAILED:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                trace.stagnant = True
                break
        else:
            consecutive_failures = 0

    trace.component   = comp
    trace.final_count = len(violations)
    if not violations:
        trace.converged = True
    return trace


# ── helpers ─────────────────────────────────────────────────────────────

def _run_drc(comp: Any, runner: "DRCRunner") -> list["DRCViolation"]:
    """Write ``comp`` to a temp GDS and run ``runner``."""
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        tmp = Path(f.name)
    try:
        comp.write_gds(str(tmp))
        return list(runner.run(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _record(
    log:        FixLog,
    ctx:        ViolationContext,
    violation:  "DRCViolation",
    proposal:   Optional[ProposedAction],
    outcome:    FixOutcome,
    source:     FixSource,
    pdk:        str,
    cell:       str,
    pre_count:  int,
    post_count: int,
) -> None:
    log.record(
        pdk         = pdk,
        cell        = cell,
        rule_raw    = violation.rule,
        rule_code   = None,                 # resolution deferred to JOIN time
        violation_x = violation.x,
        violation_y = violation.y,
        context     = ctx,
        action_verb = proposal.verb if proposal is not None else "",
        params      = (proposal.params.model_dump()
                       if proposal is not None else {}),
        outcome     = outcome,
        pre_count   = pre_count,
        post_count  = post_count,
        source      = source,
    )
