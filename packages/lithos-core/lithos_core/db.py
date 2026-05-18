"""lithos_core.db — SQLite-backed rule DB for one PDK release.

One DB file per (PDK name, PDK version). The DB schema is defined as constant
DDL; migration is intentionally not a feature — new PDK releases ingest into
fresh DB files and diffing across versions is a separate concern.

Schema (see :data:`SCHEMA_DDL`):

* ``pdk``           — single row identifying the (name, version) this DB carries.
* ``rule``          — one row per canonical foundry rule code.
* ``rule_alias``    — every tool-emitted string that resolves to a rule.
                     Strict PK on ``alias``: deck author wording collisions are
                     surfaced at ingest time, not at runtime.
* ``rule_relation`` — directed cross-reference graph (``see_also``,
                     ``fix_may_trigger``, ``depends_on``, ``supersedes``).
* ``rule_source``   — raw source text (deck block, PDF chunk) for QA, kept
                     off the hot lookup path.

JSON columns on ``rule`` are serialised through Pydantic models from
:mod:`lithos_core.ir` and :mod:`lithos_core.fix`.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from lithos_core.fix import FixMetadata
from lithos_core.ir import Constraint


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS pdk (
    name                  TEXT PRIMARY KEY,
    version               TEXT NOT NULL,
    ingested_at           TEXT NOT NULL,
    ingest_tool_versions  TEXT,
    deck_files            TEXT,
    pdf_files             TEXT
);

CREATE TABLE IF NOT EXISTS rule (
    code               TEXT PRIMARY KEY,
    category           TEXT NOT NULL DEFAULT 'unknown',
    usage_class        TEXT NOT NULL,
    short_desc         TEXT,
    constraint_json    TEXT,
    fix_metadata_json  TEXT,
    provenance_json    TEXT,
    confidence_json    TEXT,
    needs_review       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rule_alias (
    alias    TEXT PRIMARY KEY,
    code     TEXT NOT NULL,
    source   TEXT NOT NULL,
    FOREIGN KEY (code) REFERENCES rule(code)
);

CREATE TABLE IF NOT EXISTS rule_relation (
    rule_from  TEXT NOT NULL,
    rule_to    TEXT NOT NULL,
    relation   TEXT NOT NULL,
    PRIMARY KEY (rule_from, rule_to, relation)
);

CREATE TABLE IF NOT EXISTS rule_source (
    code        TEXT PRIMARY KEY,
    deck_block  TEXT,
    deck_title  TEXT,
    pdf_chunk   TEXT,
    pdf_page    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_rule_usage_class  ON rule(usage_class);
CREATE INDEX IF NOT EXISTS idx_rule_category     ON rule(category);
CREATE INDEX IF NOT EXISTS idx_rule_needs_review ON rule(needs_review);
"""


UsageClass = str
"""One of: ``geometry_primitive``, ``density``, ``antenna``, ``electrical``,
``recommendation``, ``unknown``. Kept as ``str`` so new categories can be
added without an enum migration; ingestion validates against the known set."""

AliasSource = str
"""One of: ``deck_rulecheck``, ``deck_subcheck``, ``foundry_code``, ``manual``."""

RelationKind = str
"""One of: ``see_also``, ``fix_may_trigger``, ``depends_on``, ``supersedes``."""


@dataclass(frozen=True)
class Rule:
    """One rule as it lives in the DB.

    ``category`` is the foundry-topical grouping a rule belongs to (e.g.
    ``"poly"``, ``"metal_low"``, ``"via"``, ``"antenna"``). It is set by the
    ingestion pipeline from the user-configured category definitions in
    :mod:`lithos_core.categories` and is orthogonal to ``usage_class``
    (which describes *how* the rule is consumed at runtime). ``"unknown"``
    is the fallback when no enabled category claims the code.
    """
    code:          str
    usage_class:   UsageClass
    category:      str = "unknown"
    short_desc:    Optional[str] = None
    constraint:    Optional[Constraint] = None
    fix_metadata:  Optional[FixMetadata] = None
    provenance:    dict = field(default_factory=dict)   # per-field source tags
    confidence:    dict = field(default_factory=dict)   # per-field 0..1
    needs_review:  bool = False


class RuleDB:
    """Read/write API for one (PDK, version) rule DB.

    Use as a context manager::

        with RuleDB(path) as db:
            db.set_pdk(name="sky130A", version="1.0.5", ingested_at=...)
            db.upsert_rule(...)
            ...

    Or open/close manually with :meth:`open` / :meth:`close`.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "RuleDB":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "RuleDB not open. Use `with RuleDB(path):` or call .open() first."
            )
        return self._conn

    # ── PDK identity ────────────────────────────────────────────────────

    def set_pdk(
        self,
        name: str,
        version: str,
        ingested_at: str,
        ingest_tool_versions: Optional[dict] = None,
        deck_files: Optional[list[str]] = None,
        pdf_files: Optional[list[str]] = None,
    ) -> None:
        """Set or replace the single ``pdk`` row identifying this DB."""
        self._c().execute(
            "DELETE FROM pdk WHERE name <> ?", (name,),
        )
        self._c().execute(
            "INSERT OR REPLACE INTO pdk("
            "  name, version, ingested_at, ingest_tool_versions, "
            "  deck_files, pdf_files"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                version,
                ingested_at,
                json.dumps(ingest_tool_versions or {}),
                json.dumps(deck_files or []),
                json.dumps(pdf_files or []),
            ),
        )
        self._c().commit()

    def pdk_identity(self) -> Optional[tuple[str, str]]:
        row = self._c().execute("SELECT name, version FROM pdk LIMIT 1").fetchone()
        return (row[0], row[1]) if row else None

    # ── Rules ───────────────────────────────────────────────────────────

    def upsert_rule(self, rule: Rule) -> None:
        self._c().execute(
            "INSERT OR REPLACE INTO rule("
            "  code, category, usage_class, short_desc, constraint_json, "
            "  fix_metadata_json, provenance_json, confidence_json, needs_review"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rule.code,
                rule.category,
                rule.usage_class,
                rule.short_desc,
                rule.constraint.model_dump_json() if rule.constraint is not None else None,
                rule.fix_metadata.model_dump_json() if rule.fix_metadata is not None else None,
                json.dumps(rule.provenance),
                json.dumps(rule.confidence),
                1 if rule.needs_review else 0,
            ),
        )
        self._c().commit()

    _RULE_COLS = (
        "code, category, usage_class, short_desc, constraint_json, "
        "fix_metadata_json, provenance_json, confidence_json, needs_review"
    )

    def get_rule(self, code: str) -> Optional[Rule]:
        row = self._c().execute(
            f"SELECT {self._RULE_COLS} FROM rule WHERE code = ?", (code,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_rule(row)

    def all_rules(
        self,
        usage_class: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Iterator[Rule]:
        """Iterate every rule, optionally filtered by usage_class and/or category.

        The two filters AND together — both must match when both are set.
        Either may be ``None`` to drop that filter.
        """
        where, params = _build_filter(usage_class=usage_class, category=category)
        cur = self._c().execute(
            f"SELECT {self._RULE_COLS} FROM rule{where}", params,
        )
        for row in cur:
            yield _row_to_rule(row)

    def count_rules(
        self,
        usage_class: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        where, params = _build_filter(usage_class=usage_class, category=category)
        return self._c().execute(
            f"SELECT COUNT(*) FROM rule{where}", params,
        ).fetchone()[0]

    def categories(self) -> list[tuple[str, int]]:
        """Return ``[(category, count), ...]`` for every distinct category present.

        Useful for coverage reports: "geometry_primitive: 18, antenna: 0".
        """
        cur = self._c().execute(
            "SELECT category, COUNT(*) FROM rule GROUP BY category ORDER BY category"
        )
        return list(cur)

    # ── Aliases ─────────────────────────────────────────────────────────

    def add_alias(self, alias: str, code: str, source: AliasSource) -> None:
        """Register an alias. Raises ``sqlite3.IntegrityError`` on collision —
        every alias must map to exactly one rule per PDK."""
        self._c().execute(
            "INSERT INTO rule_alias(alias, code, source) VALUES (?, ?, ?)",
            (alias, code, source),
        )
        self._c().commit()

    def resolve_alias(self, alias: str) -> Optional[str]:
        """Tool-emitted check name → canonical rule code. Returns ``None`` if
        the alias is unknown (the matcher should treat that as an error to
        investigate at ingest time, not a runtime fallback path)."""
        row = self._c().execute(
            "SELECT code FROM rule_alias WHERE alias = ?", (alias,),
        ).fetchone()
        return row[0] if row else None

    def aliases_for(self, code: str) -> list[tuple[str, str]]:
        cur = self._c().execute(
            "SELECT alias, source FROM rule_alias WHERE code = ?", (code,),
        )
        return list(cur)

    # ── Relations ───────────────────────────────────────────────────────

    def add_relation(self, rule_from: str, rule_to: str, relation: RelationKind) -> None:
        self._c().execute(
            "INSERT OR IGNORE INTO rule_relation(rule_from, rule_to, relation) "
            "VALUES (?, ?, ?)",
            (rule_from, rule_to, relation),
        )
        self._c().commit()

    def relations_from(
        self, code: str, relation: Optional[RelationKind] = None,
    ) -> list[tuple[str, str]]:
        """Return ``[(rule_to, relation), ...]`` edges leaving ``code``."""
        if relation is None:
            cur = self._c().execute(
                "SELECT rule_to, relation FROM rule_relation WHERE rule_from = ?",
                (code,),
            )
        else:
            cur = self._c().execute(
                "SELECT rule_to, relation FROM rule_relation "
                "WHERE rule_from = ? AND relation = ?",
                (code, relation),
            )
        return list(cur)

    # ── Source text ─────────────────────────────────────────────────────

    def set_source(
        self,
        code: str,
        deck_block: Optional[str] = None,
        deck_title: Optional[str] = None,
        pdf_chunk: Optional[str] = None,
        pdf_page: Optional[int] = None,
    ) -> None:
        self._c().execute(
            "INSERT OR REPLACE INTO rule_source("
            "  code, deck_block, deck_title, pdf_chunk, pdf_page"
            ") VALUES (?, ?, ?, ?, ?)",
            (code, deck_block, deck_title, pdf_chunk, pdf_page),
        )
        self._c().commit()

    def get_source(self, code: str) -> Optional[dict]:
        row = self._c().execute(
            "SELECT deck_block, deck_title, pdf_chunk, pdf_page "
            "FROM rule_source WHERE code = ?",
            (code,),
        ).fetchone()
        if row is None:
            return None
        return {
            "deck_block": row[0],
            "deck_title": row[1],
            "pdf_chunk":  row[2],
            "pdf_page":   row[3],
        }


def _row_to_rule(row: tuple) -> Rule:
    (
        code, category, usage_class, short_desc, constraint_json,
        fix_metadata_json, provenance_json, confidence_json, needs_review,
    ) = row
    return Rule(
        code         = code,
        category     = category,
        usage_class  = usage_class,
        short_desc   = short_desc,
        constraint   = Constraint.model_validate_json(constraint_json) if constraint_json else None,
        fix_metadata = FixMetadata.model_validate_json(fix_metadata_json) if fix_metadata_json else None,
        provenance   = json.loads(provenance_json) if provenance_json else {},
        confidence   = json.loads(confidence_json) if confidence_json else {},
        needs_review = bool(needs_review),
    )


def _build_filter(
    usage_class: Optional[str] = None,
    category: Optional[str] = None,
) -> tuple[str, tuple]:
    clauses: list[str] = []
    params: list = []
    if usage_class is not None:
        clauses.append("usage_class = ?")
        params.append(usage_class)
    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if not clauses:
        return "", ()
    return " WHERE " + " AND ".join(clauses), tuple(params)
