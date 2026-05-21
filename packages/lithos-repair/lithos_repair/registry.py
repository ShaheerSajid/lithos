"""lithos_repair.registry — typed action registry.

The registry serves three audiences:

1. **The repair loop** — call :meth:`ActionRegistry.apply` with a verb
   name + raw params dict (typically straight from an LLM) and get back
   ``(comp, ref)``.
2. **The agent / policy grammar** — :meth:`ActionRegistry.grammar`
   returns a JSON-schema-style dict listing every verb's param shape.
   This is what the M5 agent will hand to the LLM as the allowed action
   vocabulary.
3. **The fix-log** — :meth:`ActionRegistry.inverse_of` returns the
   inverse ``(verb, params)`` pair for any applied action, which the M4
   fix-log uses to record what would undo a fix.

Verbs are :class:`ActionDef` entries bundling the function, its Pydantic
param model, its inverse function, and a short description. The module
exports a populated singleton :data:`REGISTRY` so callers can do::

    from lithos_repair import REGISTRY
    new_comp, new_ref = REGISTRY.apply("widen", comp, ref, {"axis": "x",
                                                            "delta_um": 0.05})

without having to import each verb individually.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ConfigDict

from .actions import (
    EdgeParams,
    NarrowParams,
    RedrawParams,
    RemoveParams,
    ShiftParams,
    SnapParams,
    WidenParams,
    extend,
    extend_inverse,
    narrow,
    narrow_inverse,
    redraw,
    redraw_inverse,
    remove,
    remove_inverse,
    shift_e,
    shift_e_inverse,
    shift_n,
    shift_n_inverse,
    shift_s,
    shift_s_inverse,
    shift_w,
    shift_w_inverse,
    shrink,
    shrink_inverse,
    snap_to_grid,
    snap_to_grid_inverse,
    widen,
    widen_inverse,
)
from .features import Polygon, PolygonRef


# ── Registry entry ──────────────────────────────────────────────────────

class ActionDef(BaseModel):
    """One verb's entry in the registry.

    Stored as a Pydantic model so the registry as a whole can be
    serialised / introspected. The callables themselves are stashed as
    ``Any`` because Pydantic refuses to validate raw functions; the
    registry's apply / inverse methods do the actual call-site checks.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name:         str
    description:  str
    func:         Any
    params_model: Any
    inverse_func: Any


# ── Registry ─────────────────────────────────────────────────────────────

class ActionRegistry:
    """Name → :class:`ActionDef` lookup with apply / inverse / grammar helpers."""

    def __init__(self):
        self._actions: dict[str, ActionDef] = {}

    # ── registration ─────────────────────────────────────────────────────

    def register(
        self,
        name:         str,
        func:         Callable,
        params_model: Type[BaseModel],
        inverse_func: Callable,
        description:  str = "",
    ) -> None:
        """Add a verb to the registry.

        Raises ``ValueError`` if ``name`` is already registered — verb
        names form the LLM's grammar enum, so silent overrides would be
        a footgun.
        """
        if name in self._actions:
            raise ValueError(f"Action {name!r} already registered.")
        self._actions[name] = ActionDef(
            name         = name,
            description  = description,
            func         = func,
            params_model = params_model,
            inverse_func = inverse_func,
        )

    # ── queries ──────────────────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def names(self) -> list[str]:
        """Sorted list of registered verb names."""
        return sorted(self._actions)

    def get(self, name: str) -> ActionDef:
        """Return the :class:`ActionDef` for ``name`` (raises ``KeyError``)."""
        try:
            return self._actions[name]
        except KeyError as exc:
            raise KeyError(
                f"No action named {name!r}. Registered: {self.names()}"
            ) from exc

    # ── invocation ───────────────────────────────────────────────────────

    def apply(
        self,
        name:   str,
        comp:   Any,
        ref:    PolygonRef,
        params: dict | BaseModel,
    ) -> tuple[Any, PolygonRef]:
        """Validate ``params`` and apply verb ``name`` to ``comp``.

        Accepts either a raw ``dict`` (validated against the verb's
        Pydantic model) or an already-built params model. Returns the
        updated ``(component, ref)`` pair.
        """
        action = self.get(name)
        if isinstance(params, action.params_model):
            p = params
        else:
            p = action.params_model.model_validate(params)
        return action.func(comp, ref, p)

    def inverse_of(
        self,
        name:    str,
        params:  dict | BaseModel,
        *,
        polygon: Optional[Polygon] = None,
    ) -> tuple[str, BaseModel]:
        """Return ``(inverse_verb_name, inverse_params_model)`` for an action.

        ``polygon`` carries the targeted polygon's pre-action state.
        Most inverses ignore it; :func:`~lithos_repair.actions.remove`
        needs it because the deleted geometry is otherwise lost.
        """
        action = self.get(name)
        if isinstance(params, action.params_model):
            p = params
        else:
            p = action.params_model.model_validate(params)
        inv_name, inv_param_dict = action.inverse_func(p, polygon)
        inv_action = self.get(inv_name)
        return inv_name, inv_action.params_model.model_validate(inv_param_dict)

    # ── grammar export ───────────────────────────────────────────────────

    def grammar(self) -> dict:
        """Return a JSON-schema-style description of all verbs.

        Shape::

            {
              "verbs": {
                "widen": {
                  "description": "...",
                  "params":      <pydantic JSON schema>,
                },
                ...
              },
              "verb_names": ["extend", "narrow", ...],
            }

        The dict is JSON-serialisable; M5's agent uses it as the
        ``allowed_action_classes`` grammar for the LLM.
        """
        return {
            "verbs": {
                name: {
                    "description": entry.description,
                    "params":      entry.params_model.model_json_schema(),
                }
                for name, entry in sorted(self._actions.items())
            },
            "verb_names": self.names(),
        }


# ── Populated default registry ──────────────────────────────────────────

REGISTRY = ActionRegistry()

REGISTRY.register(
    "widen", widen, WidenParams, widen_inverse,
    description="Symmetric expand of the polygon along an axis (x or y).",
)
REGISTRY.register(
    "narrow", narrow, NarrowParams, narrow_inverse,
    description="Symmetric contract of the polygon along an axis (x or y).",
)
REGISTRY.register(
    "shift_n", shift_n, ShiftParams, shift_n_inverse,
    description="Translate the polygon north (+Y) by delta_um.",
)
REGISTRY.register(
    "shift_s", shift_s, ShiftParams, shift_s_inverse,
    description="Translate the polygon south (-Y) by delta_um.",
)
REGISTRY.register(
    "shift_e", shift_e, ShiftParams, shift_e_inverse,
    description="Translate the polygon east (+X) by delta_um.",
)
REGISTRY.register(
    "shift_w", shift_w, ShiftParams, shift_w_inverse,
    description="Translate the polygon west (-X) by delta_um.",
)
REGISTRY.register(
    "extend", extend, EdgeParams, extend_inverse,
    description="Push a single edge (n/s/e/w) outward by delta_um.",
)
REGISTRY.register(
    "shrink", shrink, EdgeParams, shrink_inverse,
    description="Push a single edge (n/s/e/w) inward by delta_um.",
)
REGISTRY.register(
    "snap_to_grid", snap_to_grid, SnapParams, snap_to_grid_inverse,
    description="Round each vertex to the nearest grid_um multiple.",
)
REGISTRY.register(
    "remove", remove, RemoveParams, remove_inverse,
    description="Delete the polygon. Inverse needs the polygon's data.",
)
REGISTRY.register(
    "redraw", redraw, RedrawParams, redraw_inverse,
    description="Add a polygon at (layer, points). Inverse is remove.",
)
