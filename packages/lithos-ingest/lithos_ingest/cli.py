"""lithos_ingest.cli — command-line driver for the ingestion pipeline.

Subcommands::

    lithos-ingest svrf   DECK  --db PATH --pdk-name NAME --pdk-version VER \\
                              [--categories YAML]
        Parse an SVRF deck and write a deck-only rule DB. Fast, deterministic,
        no LLM. Useful for first-pass verification.

    lithos-ingest full   --svrf DECK --doc DOC --db PATH \\
                              --pdk-name NAME --pdk-version VER \\
                              [--categories YAML] [--model GGUF] [--no-llm] \\
                              [--csv-code-column COL]
        Full pipeline: parse the deck, load the doc (PDF/HTML/RST/MD/CSV
        — auto-detected by extension), chunk by rule code, optionally
        extract FixMetadata with a local LLM, join, and write.

    lithos-ingest stats  DB
        Print rule-coverage summary by category and review state.

    lithos-ingest review DB [--limit N]
        List rules flagged needs_review=1, with the mismatch messages
        recorded by the joiner.

Format auto-detection (``--doc``):

    .pdf            → load_pdf (pdfplumber)
    .html / .htm    → load_html (stdlib)
    .rst / .md      → load_text (with heading-based section offsets)
    .txt            → load_text (flat)
    .csv            → csv_to_chunks (bypasses the chunker; needs --csv-code-column)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

from lithos_core import CategoryConfig, RuleDB, load_categories

from lithos_ingest.chunker import Chunk, Document, chunk_for_categories, chunk_for_codes
from lithos_ingest.loaders import csv_to_chunks, load_html, load_pdf, load_text
from lithos_ingest.parsers.svrf import parse_svrf
from lithos_ingest.parsers.types import ParsedRule
from lithos_ingest.writer import parsed_rules_to_db


# ── Doc loading by extension ────────────────────────────────────────────────

_TEXT_SUFFIXES = {".txt", ".rst", ".md", ".markdown"}


def _detect_loader(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".pdf":
        return "pdf"
    if s in {".html", ".htm"}:
        return "html"
    if s in _TEXT_SUFFIXES:
        return "text"
    if s == ".csv":
        return "csv"
    raise ValueError(
        f"Cannot detect loader for {path}: unknown extension {s!r}. "
        f"Supported: .pdf .html .htm .rst .md .markdown .txt .csv"
    )


def _load_doc_or_csv(
    path: Path, *, csv_code_column: Optional[str] = None,
) -> tuple[Optional[Document], Optional[dict[str, list[Chunk]]]]:
    """Return ``(document, csv_chunks)``. Exactly one is non-None.

    CSVs short-circuit the chunker (they're already row-structured); other
    formats produce a Document for code-anchored chunking.
    """
    kind = _detect_loader(path)
    if kind == "pdf":
        return load_pdf(path), None
    if kind == "html":
        return load_html(path), None
    if kind == "text":
        return load_text(path), None
    if kind == "csv":
        if not csv_code_column:
            raise ValueError(
                "CSV input requires --csv-code-column NAME (the column "
                "holding the canonical rule code)."
            )
        return None, csv_to_chunks(path, code_column=csv_code_column)
    raise AssertionError(f"unreachable: {kind}")


# ── Subcommand: svrf ────────────────────────────────────────────────────────

def cmd_svrf(args: argparse.Namespace) -> int:
    deck_path = Path(args.deck)
    db_path   = Path(args.db)
    categories = (
        load_categories(args.categories) if args.categories else None
    )
    src    = deck_path.read_text()
    parsed = parse_svrf(src)
    with RuleDB(db_path) as db:
        db.set_pdk(
            name        = args.pdk_name,
            version     = args.pdk_version,
            ingested_at = _now(),
            deck_files  = [str(deck_path)],
        )
        n = parsed_rules_to_db(parsed, db, categories=categories)
    print(f"Wrote {n} rules to {db_path} (deck-only).", file=sys.stderr)
    return 0


# ── Subcommand: full ────────────────────────────────────────────────────────

def cmd_full(args: argparse.Namespace) -> int:
    deck_path = Path(args.svrf)
    doc_path  = Path(args.doc)
    db_path   = Path(args.db)
    categories = (
        load_categories(args.categories) if args.categories else None
    )

    parsed: list[ParsedRule] = parse_svrf(deck_path.read_text())
    codes = [p.code for p in parsed]

    doc, csv_chunks = _load_doc_or_csv(doc_path, csv_code_column=args.csv_code_column)
    if csv_chunks is not None:
        chunks = {c: cs for c, cs in csv_chunks.items() if c in set(codes)}
    elif doc is not None:
        if categories is not None:
            chunks = chunk_for_categories(doc, codes, categories)
        else:
            chunks = chunk_for_codes(doc, codes)
    else:
        chunks = {}

    fix_metadata = None
    if args.no_llm or args.model is None:
        if args.model is None and not args.no_llm:
            print(
                "warning: --model not provided; skipping LLM extraction. "
                "Pass --no-llm to silence this warning.",
                file=sys.stderr,
            )
    else:
        from lithos_ingest.extractor import FixMetadataExtractor
        extractor = FixMetadataExtractor.from_gguf(args.model)
        # Pass only chunks for codes we actually parsed.
        scoped = {c: cs for c, cs in chunks.items() if c in set(codes)}
        fix_metadata = extractor.extract_many(scoped)

    with RuleDB(db_path) as db:
        db.set_pdk(
            name        = args.pdk_name,
            version     = args.pdk_version,
            ingested_at = _now(),
            deck_files  = [str(deck_path)],
            pdf_files   = [str(doc_path)],
            ingest_tool_versions = {
                "model": str(args.model) if args.model else None,
            },
        )
        n = parsed_rules_to_db(
            parsed, db, categories,
            fix_metadata = fix_metadata,
            chunks       = chunks,
        )

    have_llm = "yes" if fix_metadata else "no"
    print(
        f"Wrote {n} rules to {db_path} (chunks={sum(len(v) for v in chunks.values())}, "
        f"llm_extracted={have_llm}).",
        file=sys.stderr,
    )
    return 0


# ── Subcommand: stats ──────────────────────────────────────────────────────

def cmd_stats(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    with RuleDB(db_path) as db:
        ident = db.pdk_identity()
        if ident is None:
            print(f"{db_path}: no PDK identity row.", file=sys.stderr)
            return 1
        total  = db.count_rules()
        review = db.count_rules() and sum(
            1 for r in db.all_rules() if r.needs_review
        )
        print(f"PDK:       {ident[0]} v{ident[1]}")
        print(f"Rules:     {total}")
        print(f"Review:    {review}  (needs_review = 1)")
        print()
        print("By category:")
        for cat, cnt in db.categories():
            print(f"  {cat:<24} {cnt:>5}")
    return 0


# ── Subcommand: review ──────────────────────────────────────────────────────

def cmd_review(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    limit = args.limit
    shown = 0
    with RuleDB(db_path) as db:
        for rule in db.all_rules():
            if not rule.needs_review:
                continue
            if limit is not None and shown >= limit:
                break
            print(f"[{rule.code}] {rule.short_desc or ''}")
            mismatches = rule.provenance.get("review_mismatches", [])
            for m in mismatches:
                print(f"    - {m}")
            print()
            shown += 1
        if shown == 0:
            print("No rules flagged for review.", file=sys.stderr)
    return 0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ── Main ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lithos-ingest",
        description="Build and inspect lithos rule DBs from foundry deck + doc.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # svrf
    p = sub.add_parser("svrf", help="Parse an SVRF deck → deck-only rule DB.")
    p.add_argument("deck", help="SVRF deck file")
    p.add_argument("--db",          required=True, help="Output rule DB path")
    p.add_argument("--pdk-name",    required=True)
    p.add_argument("--pdk-version", required=True)
    p.add_argument("--categories",  help="Category config YAML (optional)")
    p.set_defaults(func=cmd_svrf)

    # full
    p = sub.add_parser("full", help="Deck + doc (+ optional LLM) → rule DB.")
    p.add_argument("--svrf",            required=True, help="SVRF deck path")
    p.add_argument("--doc",             required=True, help="PDF/HTML/RST/MD/CSV rule manual")
    p.add_argument("--db",              required=True)
    p.add_argument("--pdk-name",        required=True)
    p.add_argument("--pdk-version",     required=True)
    p.add_argument("--categories",      help="Category config YAML")
    p.add_argument("--model",           help="GGUF model path for LLM extraction")
    p.add_argument("--no-llm",          action="store_true",
                                       help="Skip LLM extraction (chunks still recorded)")
    p.add_argument("--csv-code-column", help="For CSV docs: the code column name")
    p.set_defaults(func=cmd_full)

    # stats
    p = sub.add_parser("stats", help="Print rule coverage by category.")
    p.add_argument("db")
    p.set_defaults(func=cmd_stats)

    # review
    p = sub.add_parser("review", help="List rules flagged for review.")
    p.add_argument("db")
    p.add_argument("--limit", type=int, default=None,
                  help="Maximum number of rules to display")
    p.set_defaults(func=cmd_review)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())
