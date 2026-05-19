"""lithos_layout.rules — the bridge between the rule DB and cell generation.

The cell generator code wants semantic accessors: "poly width minimum",
"contact size", "m0 enclosure of the contact cut on adjacent edges". Those
are *per-PDK names that the foundry calls something else* — e.g. SkyWater
sky130's "PO.W.1" is what the cell code refers to as ``poly.width_min_um``.

lithos uses a PDK-agnostic metal stack: ``m0``, ``m1``, ``m2``, … with
``contact`` for poly/diff → m0 cuts and ``via_mX_mY`` for inter-metal
cuts. The bootstrap mapping (per PDK) translates these abstract names
to the foundry's rule codes, while :class:`PDKMetadata` translates the
abstract layer names to (gds_layer, datatype) pairs.

Each PDK ships a small ``bootstrap.yaml`` mapping semantic dotted-keys
to canonical rule codes::

    mapping:
      poly:
        width_min_um:           PO.W.1
        spacing_min_um:         PO.S.1
        endcap_over_diff_um:    PO.E.1
      contact:
        size_um:                CO.W.1
        spacing_um:             CO.S.1
        enclosure_in_diff_um:   CO.E.1
        enclosure_in_m0_um:     CO.E.2
      diff:
        width_min_um:           DI.W.1
        extension_past_poly_um: DI.E.1
      m0:
        width_min_um:           LI.W.1
        spacing_min_um:         LI.S.1

:class:`BootstrapRules` wraps this mapping + a :class:`lithos_core.RuleDB`
+ a :class:`lithos_core.PDKMetadata` and exposes two surfaces:

* :meth:`BootstrapRules.get` — modern flat API: ``rules.get("poly.width_min_um")``.
* ``rules.poly["width_min_um"]`` — dict-section compatibility for code
  ported from the original ``PDKRules`` shape.

It also forwards :meth:`layer` / :meth:`device` to the underlying
metadata, and exposes :meth:`enclosure` for asymmetric enclosure rules
that return ``(adj2, opp)`` pairs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lithos_core.db import Rule, RuleDB
from lithos_core.ir import (
    AreaCheck,
    EnclosureCheck,
    SpacingCheck,
    WidthCheck,
)
from lithos_core.metadata import PDKMetadata


# ── Config schema ───────────────────────────────────────────────────────────

class BootstrapMapping(BaseModel):
    """Per-PDK semantic-name → rule-code mapping.

    ``mapping`` is conceptually flat (e.g. ``"poly.width_min_um"``) but
    written nested in YAML for readability. The loader flattens it.
    """
    model_config = ConfigDict(frozen=True)

    mapping: dict[str, str] = Field(default_factory=dict)


def load_bootstrap_mapping(path: Path | str) -> BootstrapMapping:
    """Load a bootstrap mapping YAML, flattening any nested form.

    Accepts either flat::

        mapping:
          poly.width_min_um: PO.W.1
          contacts.size_um: CO.W.1

    Or nested::

        mapping:
          poly:
            width_min_um: PO.W.1
          contacts:
            size_um: CO.W.1
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("mapping") or {}
    flat: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = str(sub_v)
        else:
            flat[str(k)] = str(v)
    return BootstrapMapping(mapping=flat)


# ── Bridge ───────────────────────────────────────────────────────────────────

class _Section:
    """Dict-flavoured proxy for ``rules.poly["width_min_um"]`` access."""
    __slots__ = ("_rules", "_prefix")

    def __init__(self, rules: "BootstrapRules", prefix: str):
        self._rules  = rules
        self._prefix = prefix

    def __getitem__(self, key: str) -> float:
        return self._rules.get(f"{self._prefix}.{key}")

    def get(self, key: str, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return f"{self._prefix}.{key}" in self._rules._mapping.mapping


class BootstrapRules:
    """Semantic-rule accessor over a rule DB + PDK metadata + bootstrap mapping.

    Constructed from three pieces:

      * a :class:`PDKMetadata` (provides layer map, grid, device defs),
      * an open :class:`RuleDB` (the parsed deck constraints),
      * a :class:`BootstrapMapping` (semantic → foundry-code translation).

    Cell-generator code can read rule values either via the modern
    :meth:`get` API or via the ``rules.<section>["<key>"]`` dict idiom
    (which makes ports from the upstream prototype near-mechanical).
    """

    def __init__(
        self,
        metadata: PDKMetadata,
        db:       RuleDB,
        mapping:  BootstrapMapping,
    ):
        self.metadata  = metadata
        self.db        = db
        self._mapping  = mapping
        self._cache: dict[str, float] = {}

    # ── Flat semantic accessor (canonical API) ──────────────────────────

    def get(self, semantic_name: str) -> float:
        """Return the numeric threshold for ``semantic_name``.

        Looks up the rule code in the mapping, fetches the rule from the
        DB, and extracts the first branch's check threshold. Cached for
        repeated lookups inside one cell-draw call.
        """
        if semantic_name in self._cache:
            return self._cache[semantic_name]
        code = self._resolve_code(semantic_name)
        rule = self.db.get_rule(code)
        if rule is None:
            raise KeyError(
                f"Bootstrap mapping says {semantic_name!r} → {code!r} "
                f"but no such rule is in the DB."
            )
        val = _threshold_from_rule(rule)
        self._cache[semantic_name] = val
        return val

    def has(self, semantic_name: str) -> bool:
        """Return True if ``semantic_name`` is mapped (does not check the DB)."""
        return semantic_name in self._mapping.mapping

    # ── Convenience: per-section dict proxy for ported code ─────────────

    def section(self, prefix: str) -> _Section:
        """Return a dict-flavoured proxy for ``prefix.*`` keys."""
        return _Section(self, prefix)

    def __getattr__(self, name: str) -> _Section:
        """Fall-through so ``rules.poly["width_min_um"]`` works for any prefix
        that the mapping uses."""
        # Only triggered when normal attribute lookup fails (e.g. for
        # ``rules.poly``). Guard against dunder access so pickle/etc. keep
        # working.
        if name.startswith("_") or name in ("metadata", "db"):
            raise AttributeError(name)
        return _Section(self, name)

    # ── Asymmetric rules (enclosure) ────────────────────────────────────

    def enclosure(
        self,
        section: str,
        key_prefix: str,
    ) -> tuple[float, float]:
        """Return ``(adj2, opp)`` for an asymmetric enclosure rule.

        Asymmetric enclosure mapping convention::

            mapping:
              contacts:
                <prefix>_2adj_um:    <code for the 2-adj-edge minimum>
                <prefix>_um:         <code for the all-sides (opposite) minimum>

        If only ``<prefix>_um`` is mapped, the rule is symmetric and
        both returned values equal that single threshold.
        """
        adj_key = f"{section}.{key_prefix}_2adj_um"
        all_key = f"{section}.{key_prefix}_um"
        adj_val = self.get(adj_key) if self.has(adj_key) else None
        all_val = self.get(all_key) if self.has(all_key) else 0.0
        if adj_val is not None:
            return (float(adj_val), float(all_val))
        return (float(all_val), float(all_val))

    # ── Forwards to PDKMetadata ─────────────────────────────────────────

    def layer(self, name: str) -> tuple[int, int]:
        return self.metadata.layer(name)

    def device(self, name: str) -> dict:
        return self.metadata.device(name)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def mfg_grid(self) -> float:
        return self.metadata.mfg_grid

    @property
    def routing_grid(self) -> float:
        return self.metadata.routing_grid

    @property
    def m0_is_m1(self) -> bool:
        """True when m0 and m1 share a GDS (layer, datatype) pair.

        Some foundries collapse the local-interconnect layer (m0) into
        the first metal layer (m1), which means the m0→m1 cut is a no-op
        and via cells should only draw the m1 landing pad.
        """
        try:
            return self.layer("m0") == self.layer("m1")
        except KeyError:
            return False

    # ── Internals ───────────────────────────────────────────────────────

    def _resolve_code(self, semantic_name: str) -> str:
        try:
            return self._mapping.mapping[semantic_name]
        except KeyError as exc:
            raise KeyError(
                f"No bootstrap mapping for semantic name {semantic_name!r} "
                f"in PDK {self.metadata.name!r}. Available: "
                f"{sorted(self._mapping.mapping)}"
            ) from exc


# ── Threshold extraction ────────────────────────────────────────────────────

def _threshold_from_rule(rule: Rule) -> float:
    """Pull a single numeric threshold out of a rule's first branch.

    Handles the four geometry-primitive check kinds. Rules with empty
    constraints or non-numeric shapes raise so the caller knows the
    bootstrap mapping is pointing at the wrong rule.
    """
    constraint = rule.constraint
    if constraint is None or not constraint.branches:
        raise ValueError(
            f"Rule {rule.code!r} has no constraint branches; can't extract "
            f"a numeric threshold for bootstrap access."
        )
    check = constraint.branches[0].check
    if isinstance(check, (WidthCheck, SpacingCheck, EnclosureCheck)):
        return float(check.threshold_um)
    if isinstance(check, AreaCheck):
        return float(check.threshold_um2)
    raise ValueError(
        f"Rule {rule.code!r}: don't know how to extract a numeric threshold "
        f"from a {type(check).__name__}."
    )
