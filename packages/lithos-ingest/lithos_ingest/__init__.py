"""lithos_ingest — PDK rule-DB builder.

Pipeline (one pass per PDK release):

1. **Deck parser** (per dialect: SVRF / PVS / KLayout-DRC / Magic) extracts the
   canonical rule list plus the formal constraint AST per rule.
   See :mod:`lithos_ingest.parsers`.
2. **PDF chunker** uses the deck-derived rule_id list as the index — for each
   code, locate it in the rule manual and extract a context window.
   See :mod:`lithos_ingest.chunker`.
3. **LLM extractor** reads each chunk under a constrained grammar and emits
   ``FixMetadata`` + a short description, scoped to the user's enabled
   categories (see :mod:`lithos_core.categories`).
   See :mod:`lithos_ingest.extractor`.
4. **Joiner** cross-validates LLM output against deck-derived constraints,
   flags disagreements, and writes the merged result through
   :class:`lithos_core.RuleDB`.
   See :mod:`lithos_ingest.joiner`.
"""

from lithos_ingest.parsers.types import ParsedRule
from lithos_ingest.parsers.svrf import SVRFParseError, parse_svrf
from lithos_ingest.parsers.klayout_drc import KLayoutDRCParseError, parse_klayout_drc
from lithos_ingest.writer import parsed_rules_to_db, svrf_to_db
from lithos_ingest.joiner import (
    JoinResult,
    cross_validate,
    join_rule,
    join_rules,
    layers_in_constraint,
    usage_class_from_constraint,
)
from lithos_ingest.chunker import (
    Chunk,
    Document,
    chunk_for_categories,
    chunk_for_codes,
    extract_window,
    find_code_occurrences,
)
from lithos_ingest.loaders import (
    csv_to_chunks,
    load_html,
    load_pdf,
    load_text,
)
from lithos_ingest.extractor import (
    ExtractionError,
    ExtractorConfig,
    FixMetadataExtractor,
    build_messages,
    fix_metadata_json_schema,
    parse_response,
)

__all__ = [
    "ParsedRule",
    "SVRFParseError",
    "parse_svrf",
    "KLayoutDRCParseError",
    "parse_klayout_drc",
    "parsed_rules_to_db",
    "svrf_to_db",
    "JoinResult",
    "cross_validate",
    "join_rule",
    "join_rules",
    "layers_in_constraint",
    "usage_class_from_constraint",
    "Chunk",
    "Document",
    "chunk_for_categories",
    "chunk_for_codes",
    "extract_window",
    "find_code_occurrences",
    "csv_to_chunks",
    "load_html",
    "load_pdf",
    "load_text",
    "ExtractionError",
    "ExtractorConfig",
    "FixMetadataExtractor",
    "build_messages",
    "fix_metadata_json_schema",
    "parse_response",
]
