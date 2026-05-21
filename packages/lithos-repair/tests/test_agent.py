"""Tests for :class:`lithos_repair.LLMRepairAgent` and prompt builder."""
from __future__ import annotations

import json

import pytest

from lithos_repair import (
    AgentProposalError,
    LLMRepairAgent,
    Polygon,
    ViolationContext,
)
from lithos_repair.agent import (
    AgentConfig,
    _extract_json,
    build_messages,
)


# ── Fixtures ────────────────────────────────────────────────────────────

def _ctx(**overrides) -> ViolationContext:
    poly = Polygon(layer=(66, 20),
                   points=((0.0, 0.0), (0.5, 0.0), (0.5, 0.2), (0.0, 0.2)))
    base = dict(
        rule="PO.W.1", layer_name="poly", primary=poly, on_grid=True,
        description="poly too narrow", measured_um=0.12,
    )
    base.update(overrides)
    return ViolationContext(**base)


def _model_fn(content: str):
    """Return a model_fn that always replies with ``content``."""
    def fn(**kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return fn


# ── JSON extractor ─────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_object(self):
        assert _extract_json('{"verb": "widen", "params": {}}') == {
            "verb": "widen", "params": {}
        }

    def test_fenced_json(self):
        raw = '```json\n{"verb":"shift_n","params":{"delta_um":0.05}}\n```'
        assert _extract_json(raw)["verb"] == "shift_n"

    def test_object_with_trailing_prose(self):
        raw = 'Sure! Here is the fix: {"verb":"narrow","params":{"axis":"x","delta_um":0.02}} hope this helps.'
        out = _extract_json(raw)
        assert out["verb"] == "narrow"
        assert out["params"]["delta_um"] == 0.02

    def test_no_json_raises(self):
        with pytest.raises(AgentProposalError, match="No JSON"):
            _extract_json("I'm not sure what to do here.")


# ── Prompt assembly ────────────────────────────────────────────────────

class TestBuildMessages:
    def test_system_message_lists_every_verb(self):
        msgs = build_messages(_ctx())
        sys_content = msgs[0]["content"]
        for verb in ("widen", "narrow", "shift_n", "shift_s",
                     "shift_e", "shift_w", "extend", "shrink",
                     "snap_to_grid", "remove", "redraw"):
            assert verb in sys_content

    def test_user_message_carries_context(self):
        ctx = _ctx()
        msgs = build_messages(ctx)
        user = msgs[1]["content"]
        assert "PO.W.1"   in user            # rule
        assert "0.12"     in user            # measured
        assert "poly"     in user            # layer
        assert "on_grid"  in user

    def test_neighbor_block_truncates_to_top_k(self):
        # Build a context with 10 neighbours; only first 4 should appear.
        from lithos_repair.features import Neighbor
        nbrs = [
            Neighbor(
                polygon = Polygon(layer=(66, 20),
                                  points=((float(i), 0.0), (i+0.5, 0.0),
                                          (i+0.5, 0.2), (float(i), 0.2))),
                distance_um = float(i),
            )
            for i in range(10)
        ]
        ctx = _ctx()
        ctx = ctx.model_copy(update={"neighbors": nbrs})
        msgs = build_messages(ctx)
        user = msgs[1]["content"]
        # The literal "10 within" should appear, but only 4 detail lines.
        assert "10 within" in user
        assert user.count("dist=") == 4


# ── propose() ──────────────────────────────────────────────────────────

class TestPropose:
    def test_valid_widen_proposal(self):
        agent = LLMRepairAgent(_model_fn(
            '{"verb": "widen", "params": {"axis": "x", "delta_um": 0.02}}'
        ))
        action = agent.propose(_ctx())
        assert action.verb == "widen"
        assert action.params.model_dump() == {"axis": "x", "delta_um": 0.02}

    def test_fenced_response(self):
        agent = LLMRepairAgent(_model_fn(
            '```json\n{"verb":"shift_n","params":{"delta_um":0.05}}\n```'
        ))
        action = agent.propose(_ctx())
        assert action.verb == "shift_n"
        assert action.params.model_dump() == {"delta_um": 0.05}

    def test_unknown_verb_raises(self):
        agent = LLMRepairAgent(_model_fn(
            '{"verb": "make_smaller", "params": {}}'
        ))
        with pytest.raises(AgentProposalError, match="not in the registry"):
            agent.propose(_ctx())

    def test_bad_params_raises(self):
        agent = LLMRepairAgent(_model_fn(
            '{"verb": "widen", "params": {"axis": "z", "delta_um": 0.02}}'
        ))
        with pytest.raises(AgentProposalError, match="Params validation"):
            agent.propose(_ctx())

    def test_missing_content_raises(self):
        def bad_fn(**kw):
            return {"choices": []}
        with pytest.raises(AgentProposalError, match="missing choices"):
            LLMRepairAgent(bad_fn).propose(_ctx())

    def test_non_string_verb_raises(self):
        agent = LLMRepairAgent(_model_fn('{"verb": 42, "params": {}}'))
        with pytest.raises(AgentProposalError, match="must be a string"):
            agent.propose(_ctx())

    def test_config_forwarded_to_model_fn(self):
        seen: dict = {}
        def capturing(**kwargs):
            seen.update(kwargs)
            return {"choices": [{"message": {"content":
                '{"verb": "widen", "params": {"axis": "x", "delta_um": 0.02}}'
            }}]}
        agent = LLMRepairAgent(
            capturing,
            config=AgentConfig(temperature=0.5, max_tokens=128, seed=7),
        )
        agent.propose(_ctx())
        assert seen["temperature"] == 0.5
        assert seen["max_tokens"]  == 128
        assert seen["seed"]        == 7
