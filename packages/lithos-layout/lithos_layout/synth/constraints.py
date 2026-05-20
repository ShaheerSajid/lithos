"""lithos_layout.synth.constraints — safe symbolic expression evaluator.

Expressions in cell topology templates reference PDK rules and device
geometry using dot notation::

    rules.diff.spacing_min_um - 2*rules.poly.endcap_over_diff_um
    N.total_y + inter_cell_gap

This module wraps :class:`BootstrapRules` and per-device
:class:`TransistorGeom` objects in attribute-access namespaces and
evaluates expressions with a restricted ``eval()``
(``__builtins__`` disabled).

Use cases:

* :func:`eval_expr` — evaluate one symbolic expression to a float.
* :func:`resolve_named_constraints` — bulk-evaluate the
  ``named_constraints`` dict on a :class:`CellTemplate`.
* :func:`build_namespace` — construct the eval namespace (mostly for
  callers that want to evaluate many expressions against the same
  rules / device geometry).
"""
from __future__ import annotations

import math
from typing import Any

from lithos_layout.rules      import BootstrapRules
from lithos_layout.transistor import TransistorGeom


# ── Recursive attribute namespace for non-rule namespaces ────────────────────

class _NS:
    """Wraps a dict so keys are accessible as attributes, recursively.

    Used for the per-device geometry namespace; :class:`BootstrapRules`
    provides its own attribute-access path (via :class:`_Section`).
    """
    __slots__ = ("_d",)

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str):
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(
                f"No attribute {name!r} in namespace {list(d)}"
            )
        val = d[name]
        return _NS(val) if isinstance(val, dict) else val

    def __repr__(self) -> str:
        d = object.__getattribute__(self, "_d")
        return f"_NS({list(d)})"


# ── Namespace construction ──────────────────────────────────────────────────

def build_namespace(
    rules: BootstrapRules,
    geoms: dict[str, TransistorGeom] | None = None,
    named: dict[str, float]          | None = None,
) -> dict[str, Any]:
    """Build an eval namespace for constraint expressions.

    Parameters
    ----------
    rules :
        Bootstrap rules. Exposed as ``rules.poly.width_min_um`` etc.
        Sections available are whatever the active bootstrap mapping
        declares (e.g. ``poly`` / ``diff`` / ``contact`` / ``m0`` /
        ``m1`` / ``via_m0_m1`` / ``nwell`` / ``nimplant`` / ``pimplant``).
    geoms :
        Device name → :class:`TransistorGeom` map. Each device is
        exposed by name so ``N.total_y`` resolves to
        ``geoms["N"].total_y_um``. Short aliases (``total_y``,
        ``total_x``, ``sd``, ``l``, ``w``, ``n``) are added alongside
        the canonical ``_um`` names.
    named :
        Pre-computed scalar constraints (e.g. ``inter_cell_gap = 0.14``).
        Added as plain float names in the namespace.

    Returns
    -------
    dict
        Namespace for ``eval(expr, {"__builtins__": {}}, ns)``.
    """
    ns: dict[str, Any] = {
        "__builtins__": {},
        "rules":        rules,
        "math":         math,
        "max":          max,
        "min":          min,
        "abs":          abs,
    }

    if geoms:
        for dev_name, g in geoms.items():
            # Expose both canonical (_um-suffix) and short-alias names.
            geom_dict = {
                **g.__dict__,
                "total_y": g.total_y_um,
                "total_x": g.total_x_um,
                "sd":      g.sd_length_um,
                "l":       g.l_um,
                "w":       g.w_um,
                "n":       g.n_fingers,
            }
            ns[dev_name] = _NS(geom_dict)

    if named:
        ns.update(named)

    return ns


# ── Expression evaluator ────────────────────────────────────────────────────

def eval_expr(
    expr:  Any,
    rules: BootstrapRules,
    geoms: dict[str, TransistorGeom] | None = None,
    named: dict[str, float]          | None = None,
) -> float:
    """Evaluate a symbolic constraint expression.

    Parameters
    ----------
    expr :
        The expression. If already a number, returned as-is. If a
        string, evaluated in the namespace built by
        :func:`build_namespace`.
    rules, geoms, named :
        See :func:`build_namespace`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the expression cannot be evaluated.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    ns = build_namespace(rules, geoms, named)
    try:
        return float(eval(str(expr), ns))
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate constraint expression {expr!r}: {exc}"
        ) from exc


# ── Named constraint resolution ─────────────────────────────────────────────

def resolve_named_constraints(
    named_constraints: dict[str, Any],
    rules:             BootstrapRules,
    geoms:             dict[str, TransistorGeom],
) -> dict[str, float]:
    """Evaluate all named constraints from a template's
    ``named_constraints`` dict.

    Named constraints may reference ``rules.*`` and device geometry —
    but they intentionally do *not* reference each other (the loader
    does not order them, so cross-references would be ambiguous).

    Parameters
    ----------
    named_constraints :
        From :attr:`CellTemplate.named_constraints`.
    rules :
        Bootstrap rules.
    geoms :
        Device geometry objects.

    Returns
    -------
    dict[str, float]
        Resolved scalar values, e.g. ``{"inter_cell_gap": 0.14}``.
    """
    resolved: dict[str, float] = {}
    for name, spec in named_constraints.items():
        if isinstance(spec, dict):
            if "min" in spec:
                resolved[name] = eval_expr(spec["min"], rules, geoms, named={})
            # other sub-keys (like "note") are ignored
        else:
            resolved[name] = eval_expr(spec, rules, geoms, named={})
    return resolved
