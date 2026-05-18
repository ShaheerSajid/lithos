"""lithos_ingest.loaders.csv — structured tabular rule lists → Chunks directly.

Some foundries ship rule values as a CSV alongside the prose manual:

::

    rule_code, layer, metric, value_um, description
    M2.W.1,    met2,  width,  0.14,     metal2 minimum width
    M2.S.1,    met2,  space,  0.14,     metal2 minimum spacing
    ...

CSV is qualitatively different from prose: each row already corresponds
to one rule, so the code-anchored chunker would just be moving text
around. This loader emits one :class:`lithos_ingest.chunker.Chunk` per
row directly, with a human-readable ``"col: val | col: val"`` rendering
in :attr:`Chunk.text` that the LLM extractor can consume the same way it
consumes a prose chunk.

The caller specifies which column carries the rule code via ``code_column``.
Rows with empty or missing codes are skipped.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Iterable, Optional

from lithos_ingest.chunker import Chunk


def csv_to_chunks(
    path: Path | str,
    *,
    code_column: str,
    encoding:    str = "utf-8",
    delimiter:   str = ",",
    columns:     Optional[Iterable[str]] = None,
) -> dict[str, list[Chunk]]:
    """Load a CSV rule table and return ``{code: [chunk]}``.

    Parameters
    ----------
    code_column
        Name of the column holding the canonical rule code. Required.
    columns
        Optional whitelist of columns to include in each chunk's rendered
        text. When ``None`` every non-empty column is included.
    """
    path = Path(path)
    out: dict[str, list[Chunk]] = {}
    with open(path, newline="", encoding=encoding) as f:
        reader = _csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or code_column not in reader.fieldnames:
            raise ValueError(
                f"CSV at {path} has no column {code_column!r}. "
                f"Found columns: {reader.fieldnames}"
            )
        whitelist = set(columns) if columns is not None else None
        for row_index, row in enumerate(reader, start=2):  # +1 header, +1 to be 1-indexed
            code = (row.get(code_column) or "").strip()
            if not code:
                continue
            parts = []
            for col, val in row.items():
                if col is None or col == code_column:
                    continue
                if whitelist is not None and col not in whitelist:
                    continue
                v = (val or "").strip()
                if v:
                    parts.append(f"{col}: {v}")
            text = " | ".join(parts)
            chunk = Chunk(
                code   = code,
                text   = text,
                page   = row_index,
                span   = (0, len(text)),
                anchor = 0,
                section = "csv",
            )
            out.setdefault(code, []).append(chunk)
    return out
