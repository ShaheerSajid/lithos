"""lithos_ingest.loaders — turn source files of various formats into Documents.

Each loader is a thin adapter; the structural / search logic lives in
:mod:`lithos_ingest.chunker`. Loaders fall into two groups:

Text-shaped formats (PDF, HTML, RST, Markdown, plain text)
    Produce a :class:`lithos_ingest.chunker.Document`. The chunker then
    finds rule-code anchors and extracts windows uniformly.

Structured tabular formats (CSV)
    Bypass the chunker — each row already corresponds to a rule, so they
    emit :class:`lithos_ingest.chunker.Chunk` directly. See
    :func:`csv_to_chunks`.
"""

from lithos_ingest.loaders.pdf  import load_pdf
from lithos_ingest.loaders.html import load_html
from lithos_ingest.loaders.text import load_text
from lithos_ingest.loaders.csv  import csv_to_chunks

__all__ = ["load_pdf", "load_html", "load_text", "csv_to_chunks"]
