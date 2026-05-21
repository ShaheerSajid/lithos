"""lithos_repair.fix_log — SQLite recording for every applied repair action.

Lives alongside :class:`lithos_core.RuleDB` in the same database file, so
``JOIN rule_alias`` can recover canonical foundry codes from whatever
raw rule string the DRC tool emitted. The fix log is the substrate for:

* Imitation-baseline training (M6) — ``(context, action)`` tuples become
  the BC dataset.
* RL fine-tune (M7) — the same rows plus outcome (pre/post violation
  counts) provide the reward signal.
* Debugging / auditing — every action a human or agent took is queryable
  after the fact.

Schema lives entirely in this module — :class:`RuleDB` doesn't need to
know about ``fix_log``. We open our own SQLite connection to the same
file; SQLite handles concurrent readers fine (the repair loop is
single-writer anyway).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from pydantic import BaseModel

if TYPE_CHECKING:                                    # pragma: no cover
    from .features import ViolationContext


# ── Schema ──────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS fix_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    pdk           TEXT NOT NULL,
    cell          TEXT NOT NULL,
    rule_code     TEXT NOT NULL,
    rule_raw      TEXT NOT NULL,
    violation_x   REAL NOT NULL,
    violation_y   REAL NOT NULL,
    context_json  TEXT NOT NULL,
    action_verb   TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    pre_count     INTEGER NOT NULL,
    post_count    INTEGER NOT NULL,
    source        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fix_log_pdk    ON fix_log(pdk);
CREATE INDEX IF NOT EXISTS idx_fix_log_rule   ON fix_log(rule_code);
CREATE INDEX IF NOT EXISTS idx_fix_log_source ON fix_log(source);
CREATE INDEX IF NOT EXISTS idx_fix_log_cell   ON fix_log(cell);
"""


_ROW_COLUMNS = (
    "id, timestamp, pdk, cell, rule_code, rule_raw, "
    "violation_x, violation_y, context_json, "
    "action_verb, params_json, outcome, "
    "pre_count, post_count, source"
)


# ── Enums (string-valued so they round-trip through SQLite verbatim) ───

class FixSource(str, Enum):
    """Who chose the action."""
    MANUAL = "manual"
    LLM    = "llm"
    POLICY = "policy"
    RL     = "rl"


class FixOutcome(str, Enum):
    """What happened when the action ran."""
    APPLIED = "applied"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ── Row dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class FixRow:
    """One row from the ``fix_log`` table."""
    id:            int
    timestamp:     str
    pdk:           str
    cell:          str
    rule_code:     str
    rule_raw:      str
    violation_x:   float
    violation_y:   float
    context_json:  str
    action_verb:   str
    params_json:   str
    outcome:       str
    pre_count:     int
    post_count:    int
    source:        str

    @property
    def delta_violations(self) -> int:
        """Negative = fix reduced violations; positive = fix made things worse."""
        return self.post_count - self.pre_count


def _row(t: tuple) -> FixRow:
    return FixRow(
        id            = int(t[0]),
        timestamp     = str(t[1]),
        pdk           = str(t[2]),
        cell          = str(t[3]),
        rule_code     = str(t[4]),
        rule_raw      = str(t[5]),
        violation_x   = float(t[6]),
        violation_y   = float(t[7]),
        context_json  = str(t[8]),
        action_verb   = str(t[9]),
        params_json   = str(t[10]),
        outcome       = str(t[11]),
        pre_count     = int(t[12]),
        post_count    = int(t[13]),
        source        = str(t[14]),
    )


# ── FixLog ──────────────────────────────────────────────────────────────

class FixLog:
    """SQLite-backed recorder for every applied repair action.

    Open against the same file as a :class:`lithos_core.RuleDB`; the two
    keep separate connections but share the underlying database, so
    ``JOIN rule_alias`` queries work across the boundary.

    Use as a context manager or call :meth:`open` / :meth:`close` manually.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def open(self) -> None:
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FixLog":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "FixLog not open. Use `with FixLog(path):` or call .open() first."
            )
        return self._conn

    # ── insert ───────────────────────────────────────────────────────────

    def record(
        self,
        *,
        pdk:         str,
        cell:        str,
        rule_raw:    str,
        violation_x: float,
        violation_y: float,
        context:     "ViolationContext | dict | str",
        action_verb: str,
        params:      "BaseModel | dict | str",
        outcome:     "FixOutcome | str",
        pre_count:   int,
        post_count:  int,
        source:      "FixSource | str",
        rule_code:   Optional[str] = None,
        timestamp:   Optional[str] = None,
    ) -> int:
        """Insert one row; return its ``id``.

        Parameters
        ----------
        rule_raw
            The raw tool-emitted rule string (the alias).
        rule_code
            Canonical code, when known. If ``None`` falls back to
            ``rule_raw`` so the column stays NOT NULL and ``JOIN
            rule_alias`` still works for the rows where the canonical
            code is known.
        context, params
            Accepted as Pydantic models, dicts, or pre-serialised JSON
            strings. Stored as JSON text.
        outcome, source
            Accepted as :class:`FixOutcome` / :class:`FixSource` or
            equivalent string. The runtime doesn't enforce the enum
            (SQLite has no enum type); the constructor just stringifies.
        timestamp
            Override for tests. Default is ``datetime.now(UTC).isoformat()``.
        """
        ctx_json    = _to_json(context)
        params_json = _to_json(params)
        outcome_s   = outcome.value if isinstance(outcome, FixOutcome) else str(outcome)
        source_s    = source.value  if isinstance(source,  FixSource)  else str(source)
        ts          = timestamp or datetime.now(timezone.utc).isoformat()
        canonical   = rule_code or rule_raw

        cur = self._c().execute(
            "INSERT INTO fix_log("
            "  timestamp, pdk, cell, rule_code, rule_raw, "
            "  violation_x, violation_y, context_json, "
            "  action_verb, params_json, outcome, "
            "  pre_count, post_count, source"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts, pdk, cell, canonical, rule_raw,
                float(violation_x), float(violation_y), ctx_json,
                action_verb, params_json, outcome_s,
                int(pre_count), int(post_count), source_s,
            ),
        )
        self._c().commit()
        return int(cur.lastrowid)

    # ── queries ──────────────────────────────────────────────────────────

    def all_rows(self) -> list[FixRow]:
        """Return every row in insertion order."""
        cur = self._c().execute(
            f"SELECT {_ROW_COLUMNS} FROM fix_log ORDER BY id",
        )
        return [_row(t) for t in cur.fetchall()]

    def count(self) -> int:
        return int(
            self._c().execute("SELECT COUNT(*) FROM fix_log").fetchone()[0]
        )

    def query(
        self,
        *,
        pdk:         Optional[str] = None,
        cell:        Optional[str] = None,
        rule_code:   Optional[str] = None,
        action_verb: Optional[str] = None,
        source:      Optional[str] = None,
        outcome:     Optional[str] = None,
    ) -> list[FixRow]:
        """Filter rows by any combination of column equality predicates."""
        clauses: list[str] = []
        args:    list[Any] = []
        for col, val in (
            ("pdk",         pdk),
            ("cell",        cell),
            ("rule_code",   rule_code),
            ("action_verb", action_verb),
            ("source",      source),
            ("outcome",     outcome),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                args.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._c().execute(
            f"SELECT {_ROW_COLUMNS} FROM fix_log{where} ORDER BY id",
            args,
        )
        return [_row(t) for t in cur.fetchall()]

    def aliases_for(self, rule_code: str) -> list[str]:
        """Return all aliases that map to ``rule_code`` via ``rule_alias``.

        Demonstrates the cross-table JOIN promised by the M4 acceptance
        criterion. Returns an empty list if no aliases exist or the
        ``rule_alias`` table isn't populated.
        """
        cur = self._c().execute(
            "SELECT DISTINCT ra.alias FROM rule_alias ra "
            "WHERE ra.code = ?",
            (rule_code,),
        )
        return [row[0] for row in cur.fetchall()]

    def join_rule_alias(
        self,
        *,
        pdk: Optional[str] = None,
    ) -> list[dict]:
        """``fix_log ⋈ rule_alias`` keyed on canonical code.

        Returns a list of dicts, one per (fix_log row, matching alias)
        pair. Rows with no matching alias still appear once with
        ``alias = None`` (LEFT JOIN). Useful for end-to-end test of the
        acceptance criterion: every recorded fix should be reachable
        from every alias the DRC tool might emit for the same rule.
        """
        clauses: list[str] = []
        args:    list[Any] = []
        if pdk is not None:
            clauses.append("fl.pdk = ?")
            args.append(pdk)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._c().execute(
            "SELECT fl.id, fl.pdk, fl.cell, fl.rule_code, fl.rule_raw, "
            "       fl.action_verb, fl.source, ra.alias "
            "FROM fix_log fl "
            f"LEFT JOIN rule_alias ra ON ra.code = fl.rule_code{where} "
            "ORDER BY fl.id, ra.alias",
            args,
        )
        return [
            {
                "id":          row[0],
                "pdk":         row[1],
                "cell":        row[2],
                "rule_code":   row[3],
                "rule_raw":    row[4],
                "action_verb": row[5],
                "source":      row[6],
                "alias":       row[7],
            }
            for row in cur.fetchall()
        ]

    # ── maintenance ──────────────────────────────────────────────────────

    def clear(self) -> int:
        """Delete every fix-log row; return the number removed."""
        cur = self._c().execute("DELETE FROM fix_log")
        self._c().commit()
        return int(cur.rowcount)


# ── helpers ─────────────────────────────────────────────────────────────

def _to_json(value: Any) -> str:
    """Serialise a Pydantic model, dict, or pre-JSONed string to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=_json_default)


def _json_default(o: Any) -> Any:
    """Fallback serialiser for objects ``json.dumps`` can't natively handle."""
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)
