"""lithos_ingest.parsers — DRC deck parsers, one module per dialect.

Each parser projects its dialect (SVRF / KLayout-DRC / PVS / Magic) into
the canonical :class:`lithos_ingest.parsers.types.ParsedRule` dataclass.
From the ingestion pipeline's point of view, the dialects are
interchangeable — everything downstream of the parser sees the same IR.
"""

from lithos_ingest.parsers.types import ParsedRule
from lithos_ingest.parsers.svrf import SVRFParseError, parse_svrf
from lithos_ingest.parsers.klayout_drc import KLayoutDRCParseError, parse_klayout_drc

__all__ = [
    "ParsedRule",
    "SVRFParseError",
    "parse_svrf",
    "KLayoutDRCParseError",
    "parse_klayout_drc",
]
