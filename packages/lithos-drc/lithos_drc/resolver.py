"""lithos_drc.resolver — map tool-emitted check names to canonical rule codes.

A DRC backend emits :class:`DRCViolation` instances whose ``rule`` field is
whatever string the underlying tool wrote into its report. The resolver
joins each violation against :class:`lithos_core.RuleDB` to attach:

* the canonical foundry rule code (via the ``rule_alias`` table),
* the rule's user-configured category,
* the rule's heuristic usage class.

Unknown aliases produce a :class:`ResolvedViolation` with ``code = None`` and
``unresolved = True``. The caller decides whether to ignore, warn, or fail
on unresolved violations — at signoff time you want to fail, but during
ingestion-iteration you typically want to warn so you can add the missing
alias to the DB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from lithos_core.db import Rule, RuleDB

from lithos_drc.base import DRCViolation


@dataclass(frozen=True)
class ResolvedViolation:
    """A :class:`DRCViolation` enriched with DB-derived metadata.

    Attributes
    ----------
    violation
        The original :class:`DRCViolation`, unchanged.
    code
        Canonical foundry rule code, or ``None`` if no alias matched.
    rule
        Full :class:`Rule` row, or ``None`` if no alias matched (or the
        alias resolved but the rule row is missing, which would indicate
        a DB-integrity bug).
    unresolved
        Convenience flag — ``True`` iff ``code is None``.
    """
    violation:  DRCViolation
    code:       Optional[str] = None
    rule:       Optional[Rule] = None

    @property
    def unresolved(self) -> bool:
        return self.code is None

    @property
    def category(self) -> Optional[str]:
        return self.rule.category if self.rule is not None else None

    @property
    def usage_class(self) -> Optional[str]:
        return self.rule.usage_class if self.rule is not None else None


def resolve_violations(
    violations: Iterable[DRCViolation],
    db: RuleDB,
) -> list[ResolvedViolation]:
    """Resolve every violation against ``db``.

    ``db`` must already be opened (call ``db.open()`` or use it as a
    context manager). One DB query per distinct alias — repeated aliases
    in the violation stream share a single lookup via the internal cache.
    """
    code_cache: dict[str, Optional[str]] = {}
    rule_cache: dict[str, Optional[Rule]] = {}
    resolved: list[ResolvedViolation] = []
    for v in violations:
        alias = v.rule
        if alias not in code_cache:
            code_cache[alias] = db.resolve_alias(alias)
        code = code_cache[alias]
        rule: Optional[Rule] = None
        if code is not None:
            if code not in rule_cache:
                rule_cache[code] = db.get_rule(code)
            rule = rule_cache[code]
        resolved.append(ResolvedViolation(violation=v, code=code, rule=rule))
    return resolved


def partition_unresolved(
    resolved: Iterable[ResolvedViolation],
) -> tuple[list[ResolvedViolation], list[ResolvedViolation]]:
    """Split a stream into ``(known, unknown)`` based on resolver outcome.

    ``unknown`` is what you'd surface as "extend the DB / add aliases";
    ``known`` is what the repair engine actually acts on.
    """
    known:   list[ResolvedViolation] = []
    unknown: list[ResolvedViolation] = []
    for r in resolved:
        (unknown if r.unresolved else known).append(r)
    return known, unknown
