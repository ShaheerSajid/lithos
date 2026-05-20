"""Tests for ``lithos_layout.synth.constraints`` — symbolic expression evaluator."""
from __future__ import annotations

import pytest

from lithos_layout.synth.constraints import (
    _NS,
    build_namespace,
    eval_expr,
    resolve_named_constraints,
)
from lithos_layout.transistor import TransistorGeom


# ── _NS attribute-access wrapper ─────────────────────────────────────────────

class TestNamespaceWrapper:
    def test_flat_dict_returns_value(self):
        ns = _NS({"width_min_um": 0.15})
        assert ns.width_min_um == 0.15

    def test_nested_dict_wraps_recursively(self):
        ns = _NS({"poly": {"width_min_um": 0.15}})
        assert ns.poly.width_min_um == 0.15

    def test_missing_attribute_raises(self):
        ns = _NS({"a": 1})
        with pytest.raises(AttributeError, match="b"):
            _ = ns.b


# ── eval_expr ────────────────────────────────────────────────────────────────

class _StubRules:
    """Minimal BootstrapRules stand-in for expression-eval tests."""
    def __init__(self, mapping: dict[str, float]):
        self._mapping = mapping

    def get(self, key: str) -> float:
        return self._mapping[key]

    # _Section uses these for ``rules.poly.width_min_um`` access. We
    # mimic the lookup by routing through ``get``.
    def __getattr__(self, name: str):
        from lithos_layout.rules import _Section
        if name.startswith("_") or name in ("metadata", "db"):
            raise AttributeError(name)
        return _Section(self, name)


def _geom(**overrides) -> TransistorGeom:
    defaults = dict(
        w_um         = 0.42,
        l_um         = 0.15,
        device_type  = "nmos",
        n_fingers    = 1,
        w_finger_um  = 0.42,
        sd_length_um = 0.17,
        n_contacts_y = 1,
        total_x_um   = 0.50,
        total_y_um   = 0.42,
    )
    defaults.update(overrides)
    return TransistorGeom(**defaults)


class TestEvalExpr:
    def test_passthrough_for_numeric_inputs(self):
        rules = _StubRules({})
        assert eval_expr(0.5, rules) == 0.5
        assert eval_expr(1, rules) == 1.0

    def test_resolves_rule_reference(self):
        rules = _StubRules({"poly.width_min_um": 0.15})
        assert eval_expr("rules.poly.width_min_um", rules) == 0.15

    def test_resolves_arithmetic(self):
        rules = _StubRules({
            "diff.spacing_min_um":      0.30,
            "poly.endcap_over_diff_um": 0.05,
        })
        v = eval_expr(
            "rules.diff.spacing_min_um - 2*rules.poly.endcap_over_diff_um",
            rules,
        )
        assert v == pytest.approx(0.20)

    def test_resolves_device_geometry(self):
        rules = _StubRules({})
        geoms = {"N": _geom(total_y_um=0.42)}
        assert eval_expr("N.total_y + 0.01", rules, geoms=geoms) == pytest.approx(0.43)

    def test_named_scalars_in_scope(self):
        rules = _StubRules({})
        assert eval_expr("inter_cell_gap * 2", rules, named={"inter_cell_gap": 0.14}) \
            == pytest.approx(0.28)

    def test_eval_failure_wraps_in_value_error(self):
        rules = _StubRules({})
        with pytest.raises(ValueError, match="Failed to evaluate"):
            eval_expr("undefined_var * 2", rules)

    def test_math_module_available(self):
        rules = _StubRules({})
        assert eval_expr("math.sqrt(4)", rules) == pytest.approx(2.0)

    def test_builtins_locked_down(self):
        """``__builtins__`` is disabled so eval can't reach open()/etc."""
        rules = _StubRules({})
        with pytest.raises(ValueError, match="Failed to evaluate"):
            eval_expr("__import__('os')", rules)


# ── resolve_named_constraints ────────────────────────────────────────────────

class TestResolveNamedConstraints:
    def test_resolves_dict_min_form(self):
        rules = _StubRules({"diff.spacing_min_um": 0.30})
        out = resolve_named_constraints(
            {"inter_cell_gap": {"min": "rules.diff.spacing_min_um - 0.10"}},
            rules,
            geoms={},
        )
        assert out == {"inter_cell_gap": pytest.approx(0.20)}

    def test_resolves_bare_expression_form(self):
        rules = _StubRules({"poly.width_min_um": 0.15})
        out = resolve_named_constraints(
            {"gap": "rules.poly.width_min_um * 2"},
            rules,
            geoms={},
        )
        assert out == {"gap": pytest.approx(0.30)}

    def test_skips_non_min_dict_keys(self):
        rules = _StubRules({})
        out = resolve_named_constraints(
            {"foo": {"note": "informational"}},
            rules,
            geoms={},
        )
        assert out == {}
