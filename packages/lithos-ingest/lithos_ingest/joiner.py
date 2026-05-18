"""lithos_ingest.joiner — merge deck-derived ParsedRule + LLM-derived FixMetadata.

The joiner is where the two halves of the ingestion pipeline meet:

  *  :class:`lithos_ingest.parsers.types.ParsedRule` carries the structured
     :class:`lithos_core.ir.Constraint` extracted deterministically from the
     deck (SVRF / PVS / KLayout-DRC).
  *  :class:`lithos_core.fix.FixMetadata` carries the LLM-extracted intent
     and allowed/forbidden action classes from the PDF/HTML/RST manual.

For each rule the joiner produces a fully-populated :class:`lithos_core.db.Rule`
with per-field provenance + confidence + a ``needs_review`` flag that fires
when cross-validation surfaces a disagreement between the two sources.

Cross-validation
----------------
The deck is treated as authoritative; the LLM is treated as advisory. So
cross-validation never *overrides* deck-derived fields — it only raises
the review flag and records mismatches for the user to triage.

Current checks:

* **Layer mismatch.** Layer names appearing in the constraint AST should
  overlap with ``FixMetadata.affected_layers``. Zero overlap → mismatch.
* **Empty intent.** ``FixMetadata.intent`` is empty after stripping → mismatch.
* **No action classes.** Neither allowed nor forbidden classes set → mismatch.

Add new checks here as you discover failure modes during real-PDK ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from lithos_core.categories import CategoryConfig
from lithos_core.db import Rule
from lithos_core.fix import FixMetadata
from lithos_core.ir import (
    AntennaCheck,
    AreaCheck,
    Constraint,
    DensityCheck,
    EnclosureCheck,
    ExistenceCheck,
    LayerBool,
    LayerConnect,
    LayerEdges,
    LayerHoles,
    LayerRef,
    LayerSelect,
    LayerSize,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.chunker import Chunk
from lithos_ingest.parsers.types import ParsedRule


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JoinResult:
    """One joined rule + diagnostics from the merge.

    ``rule`` is what gets written to the DB. ``mismatches`` lists the human-
    readable reasons (if any) the joiner flagged the rule for review;
    when non-empty, ``rule.needs_review`` is True.

    ``rule_source_chunk`` is the PDF chunk text the joiner harvested for
    storage in the ``rule_source`` table — surfaced separately so the
    writer can route it cleanly. ``None`` when no chunk was supplied.
    """
    rule:              Rule
    mismatches:        list[str] = field(default_factory=list)
    rule_source_chunk: Optional[Chunk] = None


# ── Heuristics ───────────────────────────────────────────────────────────────

def usage_class_from_constraint(constraint: Optional[Constraint]) -> str:
    """Coarse usage-class heuristic from constraint check kinds.

    * ``geometry_primitive`` — width / spacing / enclosure / area
    * ``density``           — DensityCheck present
    * ``antenna``           — AntennaCheck present
    * ``unknown``           — empty / unrecognised
    """
    if constraint is None or not constraint.branches:
        return "unknown"
    has_density   = False
    has_antenna   = False
    has_geom      = False
    has_existence = False
    for br in constraint.branches:
        chk = br.check
        if isinstance(chk, DensityCheck):
            has_density = True
        elif isinstance(chk, AntennaCheck):
            has_antenna = True
        elif isinstance(chk, (WidthCheck, SpacingCheck, EnclosureCheck, AreaCheck)):
            has_geom = True
        elif isinstance(chk, ExistenceCheck):
            has_existence = True
    if has_density:
        return "density"
    if has_antenna:
        return "antenna"
    if has_geom:
        return "geometry_primitive"
    if has_existence:
        # "this set must be empty" rules are boolean process-integrity
        # checks — categorise them with the geometry primitives so they
        # show up in the bootstrap / repair stream until the LLM
        # extraction refines them further.
        return "geometry_primitive"
    return "unknown"


def layers_in_constraint(constraint: Optional[Constraint]) -> set[str]:
    """Walk a Constraint AST and return every LayerRef name encountered.

    Used by cross-validation to compare against
    ``FixMetadata.affected_layers``. Derived intermediate names in
    ``derived_layers`` are included; structural keys (the keys of the
    ``derived_layers`` dict) are not — they're internal aliases, not
    physical layers.
    """
    if constraint is None:
        return set()
    names: set[str] = set()
    for layer_expr in constraint.derived_layers.values():
        _collect_layer_refs(layer_expr, names)
    for branch in constraint.branches:
        _collect_check_layers(branch.check, names)
    return names


def _collect_layer_refs(node, out: set[str]) -> None:
    if isinstance(node, LayerRef):
        out.add(node.name)
        return
    if isinstance(node, (LayerBool, LayerConnect)):
        for op in node.operands if isinstance(node, LayerBool) else node.layers:
            _collect_layer_refs(op, out)
        if isinstance(node, LayerConnect):
            for op in node.via_layers:
                _collect_layer_refs(op, out)
        return
    if isinstance(node, (LayerSize, LayerEdges, LayerHoles)):
        _collect_layer_refs(node.operand, out)
        return
    if isinstance(node, LayerSelect):
        _collect_layer_refs(node.subject, out)
        _collect_layer_refs(node.reference, out)
        return


def _collect_check_layers(check, out: set[str]) -> None:
    if isinstance(check, WidthCheck):
        _collect_layer_refs(check.target, out)
    elif isinstance(check, SpacingCheck):
        _collect_layer_refs(check.layer_a, out)
        if check.layer_b is not None:
            _collect_layer_refs(check.layer_b, out)
    elif isinstance(check, EnclosureCheck):
        _collect_layer_refs(check.inner, out)
        _collect_layer_refs(check.outer, out)
    elif isinstance(check, AreaCheck):
        _collect_layer_refs(check.target, out)
    elif isinstance(check, DensityCheck):
        _collect_layer_refs(check.target, out)
    elif isinstance(check, AntennaCheck):
        _collect_layer_refs(check.metal_area, out)
        _collect_layer_refs(check.gate_area,  out)


def cross_validate(
    parsed: ParsedRule,
    fix: Optional[FixMetadata],
) -> list[str]:
    """Return a list of mismatch messages between deck and LLM data.

    Empty list = no flagged issues. Order is stable so the writer can
    reproduce identical needs_review rows on re-ingest.
    """
    mismatches: list[str] = []
    if fix is None:
        return mismatches

    deck_layers = layers_in_constraint(parsed.constraint)
    fix_layers  = {layer for layer in fix.affected_layers if layer}
    if deck_layers and fix_layers and not (deck_layers & fix_layers):
        mismatches.append(
            "layer mismatch: deck constraint references "
            f"{sorted(deck_layers)} but FixMetadata.affected_layers is "
            f"{sorted(fix_layers)}"
        )

    if not fix.intent.strip():
        mismatches.append("FixMetadata.intent is empty")

    if not fix.allowed_action_classes and not fix.forbidden_action_classes:
        mismatches.append(
            "FixMetadata has neither allowed nor forbidden action classes"
        )

    return mismatches


# ── Joiner ───────────────────────────────────────────────────────────────────

def join_rule(
    parsed: ParsedRule,
    fix: Optional[FixMetadata] = None,
    *,
    category:       str = "unknown",
    chunk:          Optional[Chunk] = None,
    fix_confidence: float = 0.85,
) -> JoinResult:
    """Merge one :class:`ParsedRule` with optional :class:`FixMetadata`.

    Provenance / confidence policy:

    * ``constraint`` always comes from the deck → provenance ``"deck"``,
      confidence ``1.0``.
    * ``fix_metadata`` comes from the LLM → provenance ``"llm"``,
      confidence ``fix_confidence`` (default 0.85). Reduced to
      ``fix_confidence / 2`` when cross-validation flags issues, so the
      runtime engine can downgrade the rule's recommendations.
    * ``category`` and ``usage_class`` are deterministic projections; no
      provenance entry — they're effectively deck-derived.
    """
    mismatches = cross_validate(parsed, fix)

    provenance: dict = {"constraint": "deck"}
    confidence: dict = {"constraint": 1.0}
    if fix is not None:
        provenance["fix_metadata"] = "llm"
        adjusted = fix_confidence / 2 if mismatches else fix_confidence
        confidence["fix_metadata"] = round(adjusted, 3)

    rule = Rule(
        code         = parsed.code,
        category     = category,
        usage_class  = usage_class_from_constraint(parsed.constraint),
        short_desc   = parsed.title,
        constraint   = parsed.constraint,
        fix_metadata = fix,
        provenance   = provenance,
        confidence   = confidence,
        needs_review = bool(mismatches),
    )
    return JoinResult(rule=rule, mismatches=mismatches, rule_source_chunk=chunk)


def join_rules(
    parsed_rules: Iterable[ParsedRule],
    fix_metadata: Optional[dict[str, FixMetadata]] = None,
    *,
    categories:     Optional[CategoryConfig] = None,
    chunks:         Optional[dict[str, list[Chunk]]] = None,
    fix_confidence: float = 0.85,
) -> Iterator[JoinResult]:
    """Join a stream of ParsedRules against per-code FixMetadata + chunks.

    For each code:

      * The category is resolved via ``categories.category_for(code)`` if
        given, else ``"unknown"``.
      * The first chunk in ``chunks.get(code, [])`` (if any) is attached
        for the writer to record in ``rule_source``.
      * Missing FixMetadata is fine — produces a join with no fix data,
        equivalent to today's deck-only path.
    """
    fix_metadata = fix_metadata or {}
    chunks       = chunks or {}
    for parsed in parsed_rules:
        category = (
            categories.category_for(parsed.code)
            if categories is not None else "unknown"
        )
        chunk = chunks.get(parsed.code, [None])[0] if chunks.get(parsed.code) else None
        fix = fix_metadata.get(parsed.code)
        yield join_rule(
            parsed,
            fix,
            category       = category,
            chunk          = chunk,
            fix_confidence = fix_confidence,
        )
