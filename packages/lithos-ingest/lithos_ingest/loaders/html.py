"""lithos_ingest.loaders.html — HTML → Document using stdlib only.

Many open-source PDKs publish design rule docs as a tree of HTML pages.
This loader strips tags, decodes entities, and inserts newlines after
block-level elements so that rule codes embedded in tables, list items,
or paragraphs remain on lexically distinct lines.

Stdlib-only by design: no ``beautifulsoup4`` / ``lxml`` dependency.
Foundry rule HTML is typically clean enough that ``html.parser`` handles
it well; for unusually gnarly docs, the user can post-process with a
heavier tool and feed the result through
:func:`lithos_ingest.loaders.text.load_text`.

Section boundaries are recorded at every ``<h1>`` / ``<h2>`` / ``<h3>``,
so :meth:`Document.page_at` resolves to a 1-indexed heading number
instead of an actual page.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from lithos_ingest.chunker import Document


_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer",
}
_HEADING_TAGS = {"h1", "h2", "h3"}
_SKIP_CONTENT_TAGS = {"script", "style", "head"}


class _TextExtractor(HTMLParser):
    """Linearises HTML into plain text, recording section-break offsets.

    Section breaks fire at the *start* of every ``<h1>``/``<h2>``/``<h3>``
    so the resulting Document's ``page_breaks`` align with chapter / section
    boundaries — usable as the natural "page" idiom for HTML sources.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.section_offsets: list[int] = []
        self._suppress_depth = 0
        self._cursor = 0

    def _emit(self, s: str) -> None:
        self.parts.append(s)
        self._cursor += len(s)

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_CONTENT_TAGS:
            self._suppress_depth += 1
        if tag in _HEADING_TAGS:
            self.section_offsets.append(self._cursor)
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_CONTENT_TAGS:
            self._suppress_depth = max(0, self._suppress_depth - 1)
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data):
        if self._suppress_depth:
            return
        self._emit(data)


def load_html(path: Path | str, *, encoding: str = "utf-8") -> Document:
    """Load an HTML file from disk into a :class:`Document`."""
    path = Path(path)
    raw = path.read_text(encoding=encoding)
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    return Document(
        text        = "".join(parser.parts),
        page_breaks = parser.section_offsets or [0],
        path        = path,
    )
