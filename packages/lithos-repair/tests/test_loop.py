"""Tests for :func:`lithos_repair.repair_cell` and :class:`RepairTrace`.

We exercise the loop end-to-end with three doubles in place of real
infrastructure:

* :class:`ScriptedDRCRunner` — pre-loaded queue of violation lists.
  Each ``runner.run(gds)`` call pops the next list.
* :func:`scripted_agent` — wraps :class:`LLMRepairAgent` around a
  ``model_fn`` that pops scripted JSON responses.
* ``monkeypatch`` replaces :func:`lithos_layout.synthesize_cell` with a
  factory that returns a :class:`SynthResult` whose ``violations``
  field is the runner's first scripted list.

This bypasses the full PDK / SVRF / Calibre stack while still exercising
the real loop logic (analyze → propose → apply → re-DRC → log).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import pytest

from lithos_drc.base import DRCRunner, DRCViolation
from lithos_layout.synth.synthesizer import SynthResult
from lithos_repair import (
    FixLog,
    FixOutcome,
    FixSource,
    LLMRepairAgent,
    Polygon,
    rebuild_component,
    repair_cell,
)


# ── Doubles ─────────────────────────────────────────────────────────────

class ScriptedDRCRunner(DRCRunner):
    """A :class:`DRCRunner` that returns the next pre-loaded violation list.

    The constructor accepts a sequence of ``list[DRCViolation]``. Each
    call to :meth:`run` pops the head; if the queue is empty, it
    returns the last list forever (so the loop's post-DRC check after
    the synthesizer's initial run is also covered).
    """
    tool_name = "scripted"

    def __init__(self, metadata, queue):
        super().__init__(metadata)
        self._queue: deque[list[DRCViolation]] = deque(queue)
        self._last:  list[DRCViolation] = []

    def run(self, gds_path, cell_name: Optional[str] = None):
        if self._queue:
            self._last = self._queue.popleft()
        return list(self._last)

    def is_available(self) -> bool:
        return True


def _make_scripted_agent(responses: list[str]) -> LLMRepairAgent:
    """Wraps :class:`LLMRepairAgent` around a model_fn that pops responses."""
    q = deque(responses)
    def model_fn(**kwargs):
        msg = q.popleft() if q else q[-1]
        return {"choices": [{"message": {"role": "assistant", "content": msg}}]}
    return LLMRepairAgent(model_fn)


# ── Component + rules stubs ─────────────────────────────────────────────

def _comp_with_one_polygon():
    poly = Polygon(layer=(66, 20),
                   points=((0.0, 0.0), (0.5, 0.0), (0.5, 0.2), (0.0, 0.2)))
    return rebuild_component([poly]), poly


class _StubMetadata:
    """Minimal stand-in for :class:`PDKMetadata`."""
    name        = "stub"
    layers      = {"poly": (66, 20)}
    devices: dict = {}

    def layer(self, name): return self.layers[name]
    @property
    def mfg_grid(self): return 0.005


class _StubRules:
    """Minimal stand-in for :class:`BootstrapRules`.

    The loop only touches ``metadata.name``, ``metadata.layer``,
    ``mfg_grid``, and ``db`` (for analyzer fix-metadata hint, optional).
    """
    metadata = _StubMetadata()
    db       = None
    @property
    def mfg_grid(self): return self.metadata.mfg_grid


def _patch_synthesize(monkeypatch, comp, violations):
    """Replace :func:`synthesize_cell` with one that returns ``comp`` +
    ``violations``."""
    def fake_synth(name, rules, params=None, **kw):
        return SynthResult(
            component  = comp,
            placed     = {},
            params     = dict(params or {}),
            violations = list(violations),
        )
    monkeypatch.setattr("lithos_layout.synthesize_cell", fake_synth)


def _viol(rule="PO.W.1", layer="poly", x=0.25, y=0.1):
    return DRCViolation(rule=rule, description="", layer=layer,
                        severity="error", x=x, y=y, value=None)


# ── Tests ───────────────────────────────────────────────────────────────

WIDEN_RESPONSE = '{"verb": "widen", "params": {"axis": "x", "delta_um": 0.01}}'
SHIFT_RESPONSE = '{"verb": "shift_n", "params": {"delta_um": 0.05}}'
BAD_VERB_RESPONSE = '{"verb": "summon_pony", "params": {}}'


class TestRepairLoop:
    def test_zero_initial_violations_short_circuits(self, monkeypatch):
        comp, _ = _comp_with_one_polygon()
        _patch_synthesize(monkeypatch, comp, [])
        runner = ScriptedDRCRunner(_StubMetadata(), [[]])
        agent  = _make_scripted_agent([])    # never called

        trace = repair_cell("inverter", _StubRules(),
                            drc_runner=runner, agent=agent, max_iter=4)
        assert trace.initial_count == 0
        assert trace.final_count   == 0
        assert trace.converged is True
        assert trace.iterations == 0

    def test_converges_when_violations_drop_to_zero(self, monkeypatch):
        comp, _ = _comp_with_one_polygon()
        # Initial synth reports 2 violations; after one widen, 1; then 0.
        _patch_synthesize(monkeypatch, comp, [_viol(), _viol(x=0.5)])
        runner = ScriptedDRCRunner(_StubMetadata(), [
            [_viol(x=0.5)],   # after iter 0
            [],               # after iter 1
        ])
        agent  = _make_scripted_agent([WIDEN_RESPONSE, WIDEN_RESPONSE])

        trace = repair_cell("inverter", _StubRules(),
                            drc_runner=runner, agent=agent, max_iter=4)
        assert trace.initial_count == 2
        assert trace.final_count   == 0
        assert trace.converged is True
        assert trace.iterations == 2
        # All steps applied, monotonic decrease.
        assert all(s.outcome == FixOutcome.APPLIED for s in trace.steps)
        assert [s.delta for s in trace.steps] == [-1, -1]

    def test_capped_at_max_iter(self, monkeypatch):
        """Loop exits cleanly when max_iter is reached without convergence."""
        comp, _ = _comp_with_one_polygon()
        _patch_synthesize(monkeypatch, comp, [_viol(), _viol()])
        # Runner: returns same one-violation list forever.
        runner = ScriptedDRCRunner(_StubMetadata(), [[_viol()]])
        agent  = _make_scripted_agent([WIDEN_RESPONSE] * 5)

        trace = repair_cell("inverter", _StubRules(),
                            drc_runner=runner, agent=agent, max_iter=3)
        assert trace.iterations == 3
        assert trace.converged is False
        assert trace.stagnant  is False

    def test_stagnates_after_two_consecutive_failures(self, monkeypatch):
        comp, _ = _comp_with_one_polygon()
        _patch_synthesize(monkeypatch, comp, [_viol()])
        runner = ScriptedDRCRunner(_StubMetadata(), [[_viol()], [_viol()]])
        # Both responses pick an unknown verb → AgentProposalError.
        agent  = _make_scripted_agent([BAD_VERB_RESPONSE, BAD_VERB_RESPONSE,
                                       WIDEN_RESPONSE])

        trace = repair_cell("inverter", _StubRules(),
                            drc_runner=runner, agent=agent, max_iter=5)
        assert trace.stagnant   is True
        assert trace.converged  is False
        assert trace.iterations == 2
        assert all(s.outcome == FixOutcome.FAILED for s in trace.steps)
        assert all(s.error and "not in the registry" in s.error
                   for s in trace.steps)

    def test_fix_log_populated(self, monkeypatch, tmp_path: Path):
        comp, _ = _comp_with_one_polygon()
        _patch_synthesize(monkeypatch, comp, [_viol()])
        runner = ScriptedDRCRunner(_StubMetadata(), [[]])
        agent  = _make_scripted_agent([WIDEN_RESPONSE])

        with FixLog(tmp_path / "log.db") as log:
            trace = repair_cell(
                "inverter", _StubRules(),
                drc_runner=runner, agent=agent, max_iter=2,
                fix_log=log, source=FixSource.MANUAL,
            )
            assert trace.converged is True
            assert log.count() == 1
            row = log.all_rows()[0]
            assert row.pdk         == "stub"
            assert row.cell        == "inverter"
            assert row.rule_raw    == "PO.W.1"
            assert row.action_verb == "widen"
            assert row.outcome     == "applied"
            assert row.source      == "manual"
            assert row.pre_count   == 1
            assert row.post_count  == 0


class TestRepairTraceSummary:
    def test_summary_clean(self, monkeypatch):
        comp, _ = _comp_with_one_polygon()
        _patch_synthesize(monkeypatch, comp, [])
        runner = ScriptedDRCRunner(_StubMetadata(), [[]])
        agent  = _make_scripted_agent([])
        trace = repair_cell("inverter", _StubRules(),
                            drc_runner=runner, agent=agent, max_iter=2)
        s = trace.summary()
        assert "0 → 0" in s
        assert "clean" in s
