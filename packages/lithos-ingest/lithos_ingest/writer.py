"""lithos_ingest.writer — turn ingestion outputs into RuleDB rows.

Thin glue layer between :mod:`lithos_ingest.parsers`,
:mod:`lithos_ingest.extractor`, and :class:`lithos_core.RuleDB`. The
joining logic itself lives in :mod:`lithos_ingest.joiner`; this module
just iterates the joined results and writes them.

Public entry points:

* :func:`parsed_rules_to_db` — universal: pass parsed deck rules, optionally
  with extracted FixMetadata and PDF chunks; writes joined rows.
* :func:`svrf_to_db` — convenience that parses an SVRF file then calls
  :func:`parsed_rules_to_db` with no LLM data (deck-only ingest).

When ``fix_metadata`` is supplied, cross-validation in the joiner flags
disagreements between the deck constraint and the LLM's claims, sets
``needs_review = 1`` on the affected rows, and records the human-readable
mismatch messages under ``provenance['review_mismatches']``.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

from lithos_core.categories import CategoryConfig
from lithos_core.db import Rule, RuleDB
from lithos_core.fix import FixMetadata
from lithos_core.ir import Constraint, ConstraintBranch

from lithos_ingest.chunker import Chunk
from lithos_ingest.joiner import JoinResult, join_rules, usage_class_from_constraint
from lithos_ingest.parsers.svrf import parse_svrf
from lithos_ingest.parsers.types import ParsedRule


# Re-export for backward-compat callers that imported from writer.
__all__ = [
    "parsed_rules_to_db",
    "svrf_to_db",
    "usage_class_from_constraint",
]


def _merge_duplicate_rule(
    db: RuleDB,
    new_rule: Rule,
    parsed: ParsedRule,
    stats: dict[str, int],
) -> None:
    """Merge a duplicate-code arrival into the existing DB row.

    Appends ``new_rule.constraint.branches`` onto the existing rule's
    constraint (deduplicating identical branches) and concatenates the
    deck-block text in :class:`rule_source` so all variant bodies are
    preserved verbatim for human review.

    When the existing rule has no constraint (no-branch), the new
    rule's constraint replaces it. When both are no-branch, nothing
    changes except the deck-block concatenation.
    """
    existing = db.get_rule(new_rule.code)
    if existing is None:
        # Shouldn't happen — caller said the code was seen — but be safe.
        stats["duplicate_codes_skipped"] += 1
        return

    merged_branches: list[ConstraintBranch] = []
    if existing.constraint is not None:
        merged_branches.extend(existing.constraint.branches)
    if new_rule.constraint is not None:
        for b in new_rule.constraint.branches:
            if b not in merged_branches:
                merged_branches.append(b)

    derived = {}
    raw_text_parts: list[str] = []
    deck_dialect = "svrf"
    if existing.constraint is not None:
        derived.update(existing.constraint.derived_layers)
        deck_dialect = existing.constraint.deck_dialect or deck_dialect
        if existing.constraint.raw_deck_text:
            raw_text_parts.append(existing.constraint.raw_deck_text)
    if new_rule.constraint is not None:
        for k, v in new_rule.constraint.derived_layers.items():
            derived.setdefault(k, v)
        if new_rule.constraint.raw_deck_text:
            raw_text_parts.append(new_rule.constraint.raw_deck_text)

    merged_constraint = Constraint(
        derived_layers = derived,
        branches       = merged_branches,
        deck_dialect   = deck_dialect,
        raw_deck_text  = "\n\n".join(raw_text_parts) if raw_text_parts else "",
    ) if (merged_branches or derived or raw_text_parts) else existing.constraint

    merged = Rule(
        code         = existing.code,
        category     = existing.category,
        usage_class  = existing.usage_class,
        short_desc   = existing.short_desc or new_rule.short_desc,
        constraint   = merged_constraint,
        fix_metadata = existing.fix_metadata or new_rule.fix_metadata,
        provenance   = {**existing.provenance, "variant_count":
                        existing.provenance.get("variant_count", 1) + 1},
        confidence   = existing.confidence,
        needs_review = existing.needs_review or new_rule.needs_review,
    )
    db.upsert_rule(merged)

    # Concatenate deck-block text so every variant body is reachable.
    new_deck_block = parsed.deck_block or ""
    if new_deck_block:
        # Re-fetch the existing rule_source to merge deck_block.
        # Cheap because the DB is local SQLite.
        existing_block = ""
        row = db._c().execute(
            "SELECT deck_block FROM rule_source WHERE code = ?",
            (existing.code,),
        ).fetchone()
        if row and row[0]:
            existing_block = row[0]
        combined = f"{existing_block}\n\n{new_deck_block}" if existing_block else new_deck_block
        db.set_source(
            code       = existing.code,
            deck_block = combined,
            deck_title = parsed.title or None,
        )
    stats["duplicate_codes_merged"] += 1


def _write_one(
    db: RuleDB,
    result: JoinResult,
    *,
    parsed: ParsedRule,
    code_seen: set[str],
    stats:     dict[str, int],
) -> None:
    """Write a single JoinResult to the DB (rule + aliases + source).

    Real foundry decks contain two kinds of duplication that we have to
    tolerate during ingestion:

    1. **Duplicate rule codes** — typically the same code defined in
       multiple ``#IFDEF`` branches (process-variant overrides such as
       GP / LP / ULP). The branches' bodies are *all* meaningful — they
       describe alternative constraint shapes the runtime picks
       between based on PDK options. We merge subsequent rules' check
       expressions into the existing rule's :attr:`Constraint.branches`
       (deduplicating identical branches) instead of dropping them.
       Predicate tracking — *which* ``#IFDEF`` each branch came from —
       is a follow-up; for now the predicates stay empty and the
       branches are stored as alternatives.
    2. **Duplicate title-as-alias** — the human description is shared
       across copy-pasted per-layer rules (e.g. one ``"Wide Metal ..."``
       for ``AMS.1.M1`` through ``AMS.1.M5``). The foundry code itself
       remains a unique alias; only the ambiguous title is dropped, with
       a counter increment.
    """
    rule = result.rule

    if rule.code in code_seen:
        _merge_duplicate_rule(db, rule, parsed, stats)
        return
    code_seen.add(rule.code)

    if result.mismatches:
        rule = Rule(
            code         = rule.code,
            category     = rule.category,
            usage_class  = rule.usage_class,
            short_desc   = rule.short_desc,
            constraint   = rule.constraint,
            fix_metadata = rule.fix_metadata,
            provenance   = {**rule.provenance, "review_mismatches": result.mismatches},
            confidence   = rule.confidence,
            needs_review = rule.needs_review,
        )
    db.upsert_rule(rule)
    for alias, source in parsed.aliases:
        existing = db.resolve_alias(alias)
        if existing is None:
            db.add_alias(alias, code=parsed.code, source=source)
        elif existing == parsed.code:
            # Idempotent re-ingest of the same alias — silently skip.
            pass
        else:
            # Title collides with another rule's title. The foundry-code
            # alias for this rule (always emitted first by the parser) is
            # already in place; dropping the ambiguous human-readable
            # alias keeps runtime resolution clean.
            stats["ambiguous_aliases_skipped"] += 1

    pdf_chunk_text = result.rule_source_chunk.text if result.rule_source_chunk else None
    pdf_page       = result.rule_source_chunk.page if result.rule_source_chunk else None
    if parsed.deck_block or pdf_chunk_text:
        db.set_source(
            code       = parsed.code,
            deck_block = parsed.deck_block or None,
            deck_title = parsed.title or None,
            pdf_chunk  = pdf_chunk_text,
            pdf_page   = pdf_page,
        )


def parsed_rules_to_db(
    parsed: Iterable[ParsedRule],
    db: RuleDB,
    categories:     Optional[CategoryConfig] = None,
    *,
    fix_metadata:   Optional[dict[str, FixMetadata]] = None,
    chunks:         Optional[dict[str, list[Chunk]]] = None,
    fix_confidence: float = 0.85,
) -> int:
    """Write parsed deck rules to ``db``, optionally merged with extracted
    FixMetadata and PDF chunks.

    Parameters
    ----------
    parsed
        Stream of ParsedRule from a deck parser.
    db
        An open :class:`RuleDB`. The caller is responsible for setting
        the PDK identity row before invoking this.
    categories
        Optional :class:`CategoryConfig`. When provided, each rule's
        ``category`` field is populated via the matcher; otherwise
        ``"unknown"``.
    fix_metadata
        Per-code FixMetadata from the LLM extractor. Missing codes
        produce deck-only rows.
    chunks
        Per-code PDF chunks. The first chunk per code is recorded in the
        ``rule_source`` table as ``pdf_chunk`` + ``pdf_page``.
    fix_confidence
        Default confidence for LLM-derived ``fix_metadata`` fields.
        Halved automatically when cross-validation surfaces a mismatch.

    Returns
    -------
    int
        Number of rules written.
    """
    parsed_list = list(parsed)             # need a stable sequence to pair with results
    results = join_rules(
        parsed_list,
        fix_metadata,
        categories     = categories,
        chunks         = chunks,
        fix_confidence = fix_confidence,
    )
    code_seen: set[str] = set()
    stats: dict[str, int] = {
        "duplicate_codes_skipped":   0,
        "duplicate_codes_merged":    0,
        "ambiguous_aliases_skipped": 0,
    }
    count = 0
    for parsed_rule, result in zip(parsed_list, results):
        before = len(code_seen)
        _write_one(db, result, parsed=parsed_rule,
                   code_seen=code_seen, stats=stats)
        if len(code_seen) > before:
            count += 1

    if stats["duplicate_codes_skipped"] or stats["duplicate_codes_merged"] \
            or stats["ambiguous_aliases_skipped"]:
        import sys as _sys
        msg = []
        if stats["duplicate_codes_merged"]:
            msg.append(
                f"{stats['duplicate_codes_merged']} duplicate rule codes "
                f"merged into variant branches"
            )
        if stats["duplicate_codes_skipped"]:
            msg.append(
                f"{stats['duplicate_codes_skipped']} duplicate rule codes "
                f"skipped (first-wins)"
            )
        if stats["ambiguous_aliases_skipped"]:
            msg.append(
                f"{stats['ambiguous_aliases_skipped']} ambiguous "
                f"title-aliases dropped (shared across multiple rules)"
            )
        print("lithos-ingest: " + "; ".join(msg), file=_sys.stderr)
    return count


def svrf_to_db(
    svrf_path:     Path | str,
    db_path:       Path | str,
    pdk_name:      str,
    pdk_version:   str,
    categories:    Optional[CategoryConfig] = None,
    *,
    fix_metadata:  Optional[dict[str, FixMetadata]] = None,
    chunks:        Optional[dict[str, list[Chunk]]] = None,
    fix_confidence: float = 0.85,
) -> int:
    """End-to-end: parse an SVRF deck file and write to a fresh RuleDB.

    Sets the single ``pdk`` identity row to ``(pdk_name, pdk_version)``
    with the SVRF path recorded under ``deck_files``. ``fix_metadata`` and
    ``chunks`` are forwarded to :func:`parsed_rules_to_db` so the same
    function handles the deck-only and full-pipeline paths.
    """
    svrf_path = Path(svrf_path)
    src = svrf_path.read_text()
    parsed = parse_svrf(src)

    with RuleDB(db_path) as db:
        db.set_pdk(
            name        = pdk_name,
            version     = pdk_version,
            ingested_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            deck_files  = [str(svrf_path)],
        )
        return parsed_rules_to_db(
            parsed, db, categories,
            fix_metadata   = fix_metadata,
            chunks         = chunks,
            fix_confidence = fix_confidence,
        )
