"""lithos_repair.agent — LLM-backed proposer for one repair step.

Mirrors the :class:`lithos_ingest.FixMetadataExtractor` pattern: an
injected ``model_fn`` lets unit tests stub out the LLM, while the
:meth:`from_ollama` / :meth:`from_anthropic` factories wire up real
backends with no per-call dependency on the SDK.

A single call to :meth:`LLMRepairAgent.propose`:

1. Builds a system message describing the registered verb vocabulary
   (the M2 grammar) and a user message containing the
   :class:`~lithos_repair.features.ViolationContext`.
2. Asks the model for ``{"verb": ..., "params": ...}`` JSON.
3. Validates the response against the registry's params model for the
   chosen verb and returns a :class:`ProposedAction`.

Parsing is defensive — models occasionally wrap the JSON in markdown
fences or trailing prose; :func:`_extract_json` recovers the first
balanced object. On total failure the agent raises
:class:`AgentProposalError` so the loop can record an outcome of
``failed`` without crashing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from .features import ViolationContext
from .registry import REGISTRY, ActionRegistry


# Mirrors the FixMetadataExtractor ModelFn contract — OpenAI-style chat
# completion that returns a dict with ``choices[0].message.content``.
ModelFn = Callable[..., dict]


# ── Exceptions ──────────────────────────────────────────────────────────

class AgentProposalError(RuntimeError):
    """Raised when the model output can't be turned into a valid action."""


# ── Result type ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProposedAction:
    """A single (verb, params) the agent picked for one violation.

    ``params`` is the verb's Pydantic params model — already validated
    against the registry, so the caller can pass it straight to
    :meth:`ActionRegistry.apply`.
    """
    verb:        str
    params:      BaseModel
    raw_response: str
    """Original model text — kept for the fix-log and debugging."""


# ── Prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a DRC repair agent for a semiconductor layout generator. You
inspect ONE design-rule violation and propose ONE polygon-level fix.

Reply with NOTHING but a single JSON object of the form:

    {{"verb": "<verb_name>", "params": {{...}}}}

The verb must be drawn from the vocabulary below. Each verb's params
schema is shown; supply every required field. Distance units are µm.

Available verbs:
{grammar_block}

Heuristics:
* Width/spacing violations are usually fixed by widen / narrow / extend
  / shrink on the violating polygon.
* When ``free_space`` in a direction is non-zero, that direction is the
  safest to push the polygon into via shift_*. Negative or zero free
  space means another polygon is touching — pushing that way creates a
  new violation.
* Vias and contacts that look like array members (is_array_member=true)
  usually need shift_* in lockstep with their siblings rather than a
  single-polygon edit. Prefer shift_* over widen for those.
* Never call ``remove`` unless the rule_hint or description explicitly
  authorises it.
"""


USER_PROMPT_TEMPLATE = """\
Violation:
  rule         : {rule}
  description  : {description}
  measured (µm): {measured}
  layer        : {layer_name}
  cell         : {cell_name}
  rule_hint    : {rule_hint}

Primary polygon:
  bbox (µm)    : ({px0:.4f}, {py0:.4f}) – ({px1:.4f}, {py1:.4f})
  size (µm)    : w={pw:.4f} h={ph:.4f}
  on_grid      : {on_grid}
  array_member : {is_array_member}

Free space to nearest neighbour (µm; -1 = none in that direction):
  N={fs_n:.4f}  S={fs_s:.4f}  E={fs_e:.4f}  W={fs_w:.4f}

Nearest neighbours ({n_neighbors} within search radius):
{neighbor_block}

Propose ONE fix as JSON.
"""


def _format_grammar(registry: ActionRegistry, *, indent: int = 2) -> str:
    """Render the registry's grammar as a readable block for the prompt."""
    g = registry.grammar()
    out: list[str] = []
    for name in g["verb_names"]:
        entry = g["verbs"][name]
        out.append(f"- {name}: {entry['description']}")
        params_schema = entry["params"].get("properties", {})
        for field, spec in params_schema.items():
            typ = spec.get("type", spec.get("anyOf", "?"))
            choices = spec.get("enum")
            tail = f" (one of {choices})" if choices else ""
            out.append(f"    {field}: {typ}{tail}")
    return "\n".join(out)


def _format_neighbor_block(ctx: ViolationContext, *, top_k: int = 4) -> str:
    if not ctx.neighbors:
        return "  (none)"
    lines = []
    for nb in ctx.neighbors[:top_k]:
        cx, cy = nb.polygon.centroid
        lines.append(
            f"  - layer={nb.polygon.layer} centroid=({cx:.3f},{cy:.3f}) "
            f"dist={nb.distance_um:.4f}"
        )
    return "\n".join(lines)


def build_messages(
    ctx:      ViolationContext,
    *,
    registry: ActionRegistry = REGISTRY,
) -> list[dict]:
    """Build the (system, user) chat messages for one violation."""
    grammar_block = _format_grammar(registry)
    px0, py0, px1, py1 = ctx.primary.bbox

    return [
        {
            "role":    "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(grammar_block=grammar_block),
        },
        {
            "role":    "user",
            "content": USER_PROMPT_TEMPLATE.format(
                rule            = ctx.rule,
                description     = ctx.description or "(none)",
                measured        = ctx.measured_um if ctx.measured_um is not None else "n/a",
                layer_name      = ctx.layer_name or "(unknown)",
                cell_name       = ctx.cell_name  or "(unknown)",
                rule_hint       = ctx.rule_hint  or "(none)",
                px0=px0, py0=py0, px1=px1, py1=py1,
                pw  = ctx.primary.width_um,
                ph  = ctx.primary.height_um,
                on_grid         = ctx.on_grid,
                is_array_member = ctx.is_array_member,
                fs_n=ctx.free_space.n, fs_s=ctx.free_space.s,
                fs_e=ctx.free_space.e, fs_w=ctx.free_space.w,
                n_neighbors     = len(ctx.neighbors),
                neighbor_block  = _format_neighbor_block(ctx),
            ),
        },
    ]


# ── Response parsing ────────────────────────────────────────────────────

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```",
                             re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """Best-effort: parse the first balanced JSON object in ``text``.

    Tries (in order):

    1. JSON-fenced block ``\\`\\`\\`json {...} \\`\\`\\```.
    2. The full text as JSON.
    3. The first balanced ``{ ... }`` substring (greedy brace match).
    """
    text = text.strip()

    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group("body"))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan: find the first '{' and walk to its matching '}'.
    start = text.find("{")
    if start < 0:
        raise AgentProposalError(f"No JSON object found in response: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise AgentProposalError(
                        f"Could not parse JSON object in response: {exc}"
                    ) from exc
    raise AgentProposalError(
        f"Unterminated JSON object in response: {text!r}"
    )


# ── Config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentConfig:
    """Tunables for the LLM repair agent.

    Defaults aim for deterministic single-shot proposals: low temperature
    (the response is structured JSON, not creative text), modest token
    budget (one verb + params fits in <256 tokens), fixed seed for
    reproducibility.
    """
    temperature: float = 0.1
    top_p:       float = 0.9
    max_tokens:  int   = 256
    seed:        Optional[int] = 0


# ── Agent ───────────────────────────────────────────────────────────────

class LLMRepairAgent:
    """Propose a polygon-level repair action for one DRC violation.

    Construct directly with a ``model_fn`` for unit tests, or via
    :meth:`from_ollama` / :meth:`from_anthropic` for real models. The
    chat-completion contract is identical to
    :class:`lithos_ingest.FixMetadataExtractor`, so the existing
    backend adapters carry over without modification.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        *,
        registry: ActionRegistry = REGISTRY,
        config:   Optional[AgentConfig] = None,
    ):
        self._model_fn = model_fn
        self.registry  = registry
        self.config    = config or AgentConfig()

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def from_ollama(
        cls,
        model:    str = "qwen2.5-coder:3b",
        *,
        host:     str = "http://localhost:11434",
        timeout:  float = 180.0,
        config:   Optional[AgentConfig] = None,
        registry: ActionRegistry = REGISTRY,
    ) -> "LLMRepairAgent":
        """Reuse the Ollama adapter from :class:`FixMetadataExtractor`."""
        from lithos_ingest.extractor import FixMetadataExtractor
        # Borrow the existing model_fn — same chat completion contract.
        extractor = FixMetadataExtractor.from_ollama(
            model=model, host=host, timeout=timeout,
        )
        return cls(model_fn=extractor._model_fn,    # type: ignore[attr-defined]
                   registry=registry, config=config)

    @classmethod
    def from_anthropic(
        cls,
        api_key:  Optional[str] = None,
        *,
        model:    str = "claude-sonnet-4-5",
        config:   Optional[AgentConfig] = None,
        registry: ActionRegistry = REGISTRY,
        **client_kwargs: Any,
    ) -> "LLMRepairAgent":
        """Reuse the Anthropic adapter from :class:`FixMetadataExtractor`."""
        from lithos_ingest.extractor import FixMetadataExtractor
        extractor = FixMetadataExtractor.from_anthropic(
            api_key=api_key, model=model, **client_kwargs,
        )
        return cls(model_fn=extractor._model_fn,    # type: ignore[attr-defined]
                   registry=registry, config=config)

    # ── propose ──────────────────────────────────────────────────────────

    def propose(self, ctx: ViolationContext) -> ProposedAction:
        """Ask the model for one ``(verb, params)`` proposal.

        Raises :class:`AgentProposalError` when the response can't be
        parsed, the verb isn't in the registry, or params validation
        fails. The repair loop should record these as ``outcome=failed``.
        """
        messages = build_messages(ctx, registry=self.registry)
        response = self._model_fn(
            messages    = messages,
            temperature = self.config.temperature,
            top_p       = self.config.top_p,
            max_tokens  = self.config.max_tokens,
            seed        = self.config.seed,
        )

        try:
            raw = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentProposalError(
                f"Model response missing choices[0].message.content: {response!r}"
            ) from exc

        payload = _extract_json(raw)
        verb    = payload.get("verb")
        params  = payload.get("params", {})

        if not isinstance(verb, str):
            raise AgentProposalError(
                f"Response 'verb' must be a string; got {verb!r}"
            )
        if verb not in self.registry:
            raise AgentProposalError(
                f"Response 'verb' {verb!r} is not in the registry. "
                f"Known: {self.registry.names()}"
            )
        if not isinstance(params, dict):
            raise AgentProposalError(
                f"Response 'params' must be an object; got {type(params).__name__}"
            )

        action = self.registry.get(verb)
        try:
            params_model = action.params_model.model_validate(params)
        except ValidationError as exc:
            raise AgentProposalError(
                f"Params validation failed for verb {verb!r}: {exc}"
            ) from exc

        return ProposedAction(verb=verb, params=params_model, raw_response=raw)
