"""lithos_ingest.parsers.types — shared dataclass for parser output.

Every deck-dialect parser emits ``ParsedRule`` instances. The ingestion
joiner turns these into :class:`lithos_core.db.Rule` rows after enriching
them with PDF-derived fix metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lithos_core.ir import Constraint


@dataclass(frozen=True)
class ParsedRule:
    """One rule extracted from a deck.

    Attributes
    ----------
    code
        Canonical foundry code if extractable from the rule's title
        (e.g. ``"M2.S.1"`` from a title like ``"M2.S.1: metal2 spacing"``).
        Falls back to the deck-internal rule name when the title carries
        no code prefix.
    title
        Verbatim rule-block title string (e.g. the SVRF ``RULECHECK`` title).
        Stored on the DB row as ``deck_title``.
    aliases
        Every string a DRC tool might emit for this rule — typically
        includes ``code``, ``title``, and any nested sub-check names. Each
        element is ``(alias, source)`` where ``source`` is one of
        ``"foundry_code"``, ``"deck_rulecheck"``, ``"deck_subcheck"``.
    constraint
        The structured constraint IR.
    deck_block
        Raw deck text for this rule, stored verbatim for human review.
    """
    code:        str
    title:       str
    aliases:     list[tuple[str, str]] = field(default_factory=list)
    constraint:  Optional[Constraint] = None
    deck_block:  str = ""
