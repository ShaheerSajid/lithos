"""lithos_ingest.loaders.text — plain text / RST / Markdown → Document.

Rule manuals shipped as ``.rst`` / ``.md`` / ``.txt`` are treated as flat
text: the chunker's code-anchored search doesn't care about markup syntax,
and RST/Markdown control characters (``==``, ``**``, ``\\.\\. directive::``,
etc.) sit outside the rule-code character class and don't interfere with
anchor matching.

For RST and Markdown, section breaks are derived from underline-style
RST headings (``====``, ``----``, ``~~~~`` etc.) and ATX-style Markdown
headings (``#``, ``##``, …) so :meth:`Document.page_at` returns a useful
section index.
"""
from __future__ import annotations

import re
from pathlib import Path

from lithos_ingest.chunker import Document


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+\S.*$", re.MULTILINE)
"""ATX Markdown heading: ``#`` to ``######`` followed by content."""

_RST_HEADING_RE = re.compile(
    r"^(?P<title>\S.{0,200})\n(?P<underline>([=\-~`'\"^*+#<>]){3,})\s*$",
    re.MULTILINE,
)
"""RST underline heading: title line followed by a line of dashes (or other
allowed punctuation) at least 3 chars long."""


def load_text(path: Path | str, *, encoding: str = "utf-8") -> Document:
    """Load a plain-text / RST / Markdown file into a :class:`Document`.

    Section offsets are inferred from heading syntax of both formats;
    files without recognised headings get a single section starting at 0.
    """
    path = Path(path)
    text = path.read_text(encoding=encoding)
    offsets = sorted({
        *(m.start() for m in _MD_HEADING_RE.finditer(text)),
        *(m.start() for m in _RST_HEADING_RE.finditer(text)),
    })
    return Document(
        text        = text,
        page_breaks = offsets or [0],
        path        = path,
    )
