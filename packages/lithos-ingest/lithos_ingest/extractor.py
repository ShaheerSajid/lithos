"""lithos_ingest.extractor — LLM-driven FixMetadata extraction.

Given a :class:`lithos_ingest.chunker.Chunk` (a context window of text
surrounding a rule code in some rule manual — PDF, HTML, RST, CSV),
produce a :class:`lithos_core.fix.FixMetadata` describing the intent of
the rule and which classes of fix-actions are appropriate.

Design
------
* The extractor is **dependency-injected** with a ``model_fn`` callable
  that mirrors ``llama_cpp.Llama.create_chat_completion``'s signature.
  This makes the extractor unit-testable without loading a multi-GB GGUF —
  pass in a mock that returns canned chat-completion responses.
* For real use, instantiate via :meth:`FixMetadataExtractor.from_gguf`,
  which is the thin adapter over ``llama-cpp-python``.
* Decoding is **schema-constrained**: the generated JSON is forced to
  match :class:`FixMetadata`'s Pydantic JSON schema by llama.cpp's
  GBNF / response_format mechanism. The model can only emit valid output.

Why local + small
-----------------
The task is translation from natural-language rule text into a small typed
schema — a setting where constrained decoding makes 3B–7B models match
much larger ones. No GPU required (llama.cpp runs on CPU). PDK rule
manuals are confidential to varying degrees, so on-prem inference matters.

Install (for real model usage)
------------------------------
The runtime dep is declared optional::

    pip install 'lithos-ingest[llm]'

Then download a GGUF quant. Models that have worked well in similar
structured-extraction tasks:

* ``Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`` (≈ 2 GB)
* ``Phi-3.5-mini-instruct-Q4_K_M.gguf``     (≈ 2.4 GB)
* ``Llama-3.2-3B-Instruct-Q4_K_M.gguf``     (≈ 2 GB)

The extractor doesn't care which — they all work through the same API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from lithos_core.fix import FixMetadata

from lithos_ingest.chunker import Chunk


# ── Prompt construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You extract structured fix metadata for semiconductor design-rule (DRC) violations.

You will be given one DRC rule from a foundry PDK. The input contains:

  * the rule's foundry code (e.g. "M2.S.1");
  * its prose description as it appears in the rule manual;
  * optionally, the raw deck source for the rule (SVRF / KLayout-DRC /
    PVS) — use this to ground your understanding of which layers are
    involved and what kind of check is being performed.

Return ONE JSON object describing the rule:

  - intent: a single sentence stating the physical or electrical reason
    the rule exists (e.g. "prevents litho bridging between adjacent
    metal-2 lines"). Plain factual language; do NOT speculate.
  - allowed_action_classes: list of opaque action-class tags naming the
    kinds of fix-actions a layout-repair engine could use to resolve a
    violation. Tags are short snake_case strings drawn from the foundry-
    facing vocabulary, e.g. "widen", "shift_orthogonal", "drop_via",
    "add_fill", "split_polygon", "increase_enclosure", "remove_overlap".
  - forbidden_action_classes: action classes explicitly ruled out by the
    rule text (rare; usually empty).
  - affected_layers: every physical / derived layer name mentioned in
    the rule, verbatim as written (e.g. "MET2", "NWEL", "HV_PGATE_W").
  - branches: optional conditional alternatives. Populate only when the
    rule text states "if <condition> then <different action>".
  - notes: free-text guidance worth surfacing for human review. Quote
    foundry phrasing verbatim where it's load-bearing.

Empty-set / forbidden-overlap rules (e.g. "PP AND NP not allowed to
overlap") get intent="prevent overlap between <A> and <B>",
allowed_action_classes=["remove_overlap", "shift_orthogonal"], and the
two layers in affected_layers.

Do NOT invent details that aren't in the text or deck body. Use empty
lists / strings when information is absent. Output the JSON object
ONLY — no prose, no preamble, no code fences.

────────────────────────────────────────────────────────────────────────
Example input → output (one shot to anchor the JSON shape):

INPUT:
  Rule code: M2.S.1
  Description: Min. metal-2 space < 0.28
  Deck body:
      M2.S.1 { @ Min. metal-2 space < 0.28
        EXT MET2 < 0.28 ABUT < 90 SINGULAR REGION
      }

OUTPUT:
  {
    "intent": "prevents litho bridging between adjacent metal-2 polygons during fabrication",
    "allowed_action_classes": ["widen", "shift_orthogonal"],
    "forbidden_action_classes": [],
    "affected_layers": ["MET2"],
    "branches": [],
    "notes": "Min. metal-2 space < 0.28",
    "pdf_page": null
  }
────────────────────────────────────────────────────────────────────────
"""

USER_TEMPLATE = """\
Rule code: {code}

Description:
\"\"\"
{chunk_text}
\"\"\"
{deck_block_section}
Return the FixMetadata JSON.
"""

_DECK_BLOCK_TEMPLATE = """
Deck body ({dialect}):
```
{deck_block}
```
"""


def build_messages(
    chunk: Chunk,
    *,
    code:       Optional[str] = None,
    deck_block: Optional[str] = None,
    deck_dialect: Optional[str] = None,
) -> list[dict]:
    """Build chat messages for one chunk.

    Parameters
    ----------
    chunk
        The PDF / source-doc context window for this rule.
    code
        Override the rule code (defaults to ``chunk.code``). Use when the
        chunk represents a sub-section of a different rule.
    deck_block
        Optional raw deck body for the rule (SVRF / KLayout-DRC source).
        Strongly recommended: it lets the LLM ground its layer-name
        guesses against the actual code. Pulled from
        :class:`lithos_core.db.Rule`'s ``rule_source.deck_block``.
    deck_dialect
        Label for the deck block ("svrf" / "klayout" / "pvs") — used
        purely for the prompt's code-block fence.
    """
    deck_section = ""
    if deck_block:
        deck_section = _DECK_BLOCK_TEMPLATE.format(
            dialect    = deck_dialect or "deck",
            deck_block = deck_block.strip(),
        )
    user = USER_TEMPLATE.format(
        code               = code or chunk.code,
        chunk_text         = chunk.text.strip(),
        deck_block_section = deck_section,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


def fix_metadata_json_schema() -> dict:
    """The JSON schema FixMetadata serialises to. Used to constrain decoding."""
    return FixMetadata.model_json_schema()


# ── Response parsing ─────────────────────────────────────────────────────────

class ExtractionError(RuntimeError):
    """Raised when the LLM response can't be parsed into a FixMetadata.

    Usually means the model returned malformed JSON despite the grammar
    constraint, or returned JSON that doesn't validate against
    FixMetadata's schema (e.g. a required field missing). Includes the raw
    response in :attr:`raw` for debugging."""

    def __init__(self, message: str, *, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def parse_response(response: dict) -> FixMetadata:
    """Parse a llama.cpp ``create_chat_completion`` response into FixMetadata.

    Expects the standard response shape::

        {"choices": [{"message": {"content": "<json>"}}]}

    Raises :class:`ExtractionError` with the raw content on parse failure.
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtractionError(f"unexpected response shape: {exc}") from exc

    text = (content or "").strip()
    # Some models still wrap JSON in ```json fences despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        text = text.strip("`").strip()

    try:
        return FixMetadata.model_validate_json(text)
    except Exception as exc:
        raise ExtractionError(
            f"response not valid FixMetadata JSON: {exc}",
            raw=content,
        ) from exc


# ── Extractor ────────────────────────────────────────────────────────────────

# Mirrors llama_cpp.Llama.create_chat_completion: takes messages + kwargs,
# returns a dict in the OpenAI-style chat-completion format.
ModelFn = Callable[..., dict]


@dataclass(frozen=True)
class ExtractorConfig:
    """Tunable knobs for the LLM extractor.

    Defaults aim at deterministic structured output, not creative prose:
    low temperature, modest max_tokens (FixMetadata serialises in well
    under 512 tokens for any realistic rule), fixed seed for repeatability.
    """
    temperature: float = 0.1
    top_p:       float = 0.9
    max_tokens:  int   = 512
    seed:        Optional[int] = 0


class FixMetadataExtractor:
    """Extract :class:`FixMetadata` from chunks via an injected model callable.

    Construct directly with a ``model_fn`` for unit tests, or via
    :meth:`from_gguf` for real model loading::

        # unit-test path
        extractor = FixMetadataExtractor(model_fn=mock_create_chat_completion)
        meta = extractor.extract(chunk)

        # real-model path
        extractor = FixMetadataExtractor.from_gguf(
            "models/qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
            n_ctx=4096, n_threads=8,
        )
        meta = extractor.extract(chunk)
    """

    def __init__(
        self,
        model_fn: ModelFn,
        *,
        config: Optional[ExtractorConfig] = None,
    ):
        self._model_fn = model_fn
        self.config    = config or ExtractorConfig()
        self._schema   = fix_metadata_json_schema()

    @classmethod
    def from_ollama(
        cls,
        model:   str = "qwen2.5-coder:3b",
        *,
        host:    str = "http://localhost:11434",
        timeout: float = 180.0,
        config:  Optional[ExtractorConfig] = None,
    ) -> "FixMetadataExtractor":
        """Construct an extractor backed by a local Ollama server.

        Ollama is the friction-free path for running an LLM locally with no
        API keys, no cloud cost, no GPU. Install with::

            curl -fsSL https://ollama.com/install.sh | sh    # Linux
            # or download from https://ollama.com for macOS/Windows
            ollama pull qwen2.5-coder:3b                      # ~2 GB

        Then in Python::

            from lithos_ingest import FixMetadataExtractor
            extractor = FixMetadataExtractor.from_ollama("qwen2.5-coder:3b")
            meta = extractor.extract(chunk, deck_block=deck_src,
                                    deck_dialect="svrf")

        Implementation notes:

        * Uses stdlib ``urllib`` — no extra dependency on the ``openai``,
          ``ollama``, or ``requests`` packages.
        * Sets Ollama's ``format: "json"`` flag, which constrains the
          model to emit valid JSON. The system prompt's few-shot example
          anchors the *shape*; ``format: json`` enforces the *syntax*.
        * System message is concatenated with the first user message
          (Ollama's chat endpoint accepts ``role: system`` natively, so
          we forward as-is).

        Recommended models for this task: ``qwen2.5-coder:3b`` (small,
        strong on structured outputs), ``llama3.2:3b`` (general-purpose),
        ``mistral:7b-instruct`` (more capable, slower on CPU).
        """
        import json as _json
        import urllib.error
        import urllib.request

        url = f"{host.rstrip('/')}/api/chat"

        def _model_fn(**kwargs):
            messages = kwargs.get("messages", [])
            body = {
                "model":    model,
                "messages": [{"role": m["role"], "content": m["content"]}
                             for m in messages],
                "format":   "json",
                "stream":   False,
                "options":  {
                    "temperature": kwargs.get("temperature") or 0.1,
                    "num_predict": kwargs.get("max_tokens")  or 1024,
                },
            }
            req = urllib.request.Request(
                url,
                data=_json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:                # pragma: no cover
                raise RuntimeError(
                    f"Could not reach Ollama at {url}: {exc}. "
                    f"Is `ollama serve` running and the model pulled?"
                ) from exc

            content = (data.get("message") or {}).get("content", "")
            return {"choices": [{"message": {
                "role":    "assistant",
                "content": content,
            }}]}

        return cls(model_fn=_model_fn, config=config)

    @classmethod
    def from_anthropic(
        cls,
        api_key: Optional[str] = None,
        *,
        model:   str = "claude-sonnet-4-5",
        config:  Optional[ExtractorConfig] = None,
        **client_kwargs: Any,
    ) -> "FixMetadataExtractor":
        """Construct an extractor backed by the Anthropic Messages API.

        Falls back to the ``ANTHROPIC_API_KEY`` environment variable when
        ``api_key`` is not given. The adapter:

        * pulls the ``system`` message out of the chat list (Anthropic
          takes it as a top-level parameter, not as a message);
        * forwards the remaining messages through ``client.messages.create``;
        * wraps the response in the OpenAI-flavoured
          ``{"choices": [{"message": {"content": ...}}]}`` shape that
          :func:`parse_response` expects.

        The ``response_format`` schema argument from the llama.cpp path is
        ignored (Anthropic uses prompt-side conventions for JSON). The
        few-shot example baked into :data:`SYSTEM_PROMPT` is what keeps
        the model output JSON-shaped.
        """
        try:
            import anthropic                              # type: ignore[import-not-found]
        except ImportError as exc:                        # pragma: no cover
            raise ImportError(
                "FixMetadataExtractor.from_anthropic requires the anthropic "
                "SDK. Install with: pip install 'lithos-ingest[llm]' "
                "(or pip install anthropic)."
            ) from exc

        client = anthropic.Anthropic(api_key=api_key, **client_kwargs)

        def _model_fn(**kwargs):
            messages = kwargs.get("messages", [])
            system_text = ""
            chat_messages: list[dict] = []
            for m in messages:
                if m.get("role") == "system":
                    system_text = m.get("content", "")
                else:
                    chat_messages.append({
                        "role":    m["role"],
                        "content": m["content"],
                    })
            resp = client.messages.create(
                model       = model,
                max_tokens  = kwargs.get("max_tokens") or 1024,
                system      = system_text,
                messages    = chat_messages,
                temperature = kwargs.get("temperature") or 0.1,
            )
            # Concatenate any text-shaped content blocks (Anthropic responses
            # may include thinking / tool_use blocks; we keep only text).
            text_parts = [
                getattr(b, "text", "") for b in (resp.content or [])
                if getattr(b, "type", "text") == "text"
            ]
            return {"choices": [{"message": {
                "role":    "assistant",
                "content": "".join(text_parts),
            }}]}

        return cls(model_fn=_model_fn, config=config)

    @classmethod
    def from_gguf(
        cls,
        model_path: Path | str,
        *,
        n_ctx:     int = 4096,
        n_threads: Optional[int] = None,
        verbose:   bool = False,
        config:    Optional[ExtractorConfig] = None,
        **llama_kwargs: Any,
    ) -> "FixMetadataExtractor":
        """Construct an extractor backed by a real ``llama_cpp.Llama``.

        Forwards ``n_ctx`` / ``n_threads`` / any extra kwargs to ``Llama``.
        Raises :class:`ImportError` with an install hint when
        ``llama-cpp-python`` isn't available.
        """
        try:
            from llama_cpp import Llama                  # type: ignore[import-not-found]
        except ImportError as exc:                       # pragma: no cover - install hint
            raise ImportError(
                "FixMetadataExtractor.from_gguf requires llama-cpp-python. "
                "Install with: pip install 'lithos-ingest[llm]'"
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {model_path}")

        llama = Llama(
            model_path = str(model_path),
            n_ctx      = n_ctx,
            n_threads  = n_threads,
            verbose    = verbose,
            **llama_kwargs,
        )
        return cls(model_fn=llama.create_chat_completion, config=config)

    def extract(
        self,
        chunk: Chunk,
        *,
        code:         Optional[str] = None,
        deck_block:   Optional[str] = None,
        deck_dialect: Optional[str] = None,
    ) -> FixMetadata:
        """Extract FixMetadata for one chunk.

        Pass ``deck_block`` (the raw SVRF / KLayout-DRC source for the
        rule, from ``RuleDB.get_source(code)['deck_block']``) to give the
        model a second grounding source alongside the PDF text. Strongly
        recommended for unstructured rules where the deck syntax carries
        the semantic intent that the prose description glosses over.
        """
        messages = build_messages(
            chunk,
            code         = code,
            deck_block   = deck_block,
            deck_dialect = deck_dialect,
        )
        response = self._model_fn(
            messages    = messages,
            temperature = self.config.temperature,
            top_p       = self.config.top_p,
            max_tokens  = self.config.max_tokens,
            seed        = self.config.seed,
            response_format = {
                "type":   "json_object",
                "schema": self._schema,
            },
        )
        return parse_response(response)

    def extract_many(
        self,
        chunks: dict[str, list[Chunk]],
        *,
        deck_blocks: Optional[dict[str, str]] = None,
        deck_dialect: Optional[str] = None,
    ) -> dict[str, FixMetadata]:
        """Convenience: extract one FixMetadata per code from a chunk map.

        Uses the **first** chunk per code. Optional ``deck_blocks`` maps
        ``code → raw deck source`` so each rule's prompt includes its
        deck body for grounding (recommended: pull from
        ``RuleDB.get_source``).
        """
        out: dict[str, FixMetadata] = {}
        for code, code_chunks in chunks.items():
            if not code_chunks:
                continue
            deck = (deck_blocks or {}).get(code)
            out[code] = self.extract(
                code_chunks[0],
                code=code,
                deck_block=deck,
                deck_dialect=deck_dialect,
            )
        return out
