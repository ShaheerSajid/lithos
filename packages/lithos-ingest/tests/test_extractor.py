"""LLM extractor tests using an injected mock model.

The extractor's design separates the pure logic (prompt construction,
response parsing) from the model invocation. These tests cover both
without loading a real GGUF.
"""
from __future__ import annotations

import json

import pytest

from lithos_core import FixMetadata

from lithos_ingest.chunker import Chunk
from lithos_ingest.extractor import (
    ExtractionError,
    FixMetadataExtractor,
    build_messages,
    fix_metadata_json_schema,
    parse_response,
)


def _chunk(text: str, code: str = "M2.S.1") -> Chunk:
    return Chunk(
        code=code, text=text, page=1, span=(0, len(text)), anchor=0,
    )


def _completion(content: str) -> dict:
    """Mimic the llama.cpp chat-completion response shape."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ── Prompt construction (pure) ──────────────────────────────────────────────

def test_build_messages_includes_code_and_text():
    chunk = _chunk("Minimum spacing of met2 shall be 0.14 micrometres.")
    msgs = build_messages(chunk)
    assert msgs[0]["role"] == "system"
    user_content = msgs[1]["content"]
    assert "M2.S.1" in user_content
    assert "Minimum spacing of met2" in user_content


def test_build_messages_explicit_code_overrides_chunk():
    chunk = _chunk("text", code="orig")
    msgs = build_messages(chunk, code="override")
    assert "override" in msgs[1]["content"]
    assert "orig" not in msgs[1]["content"]


# ── JSON schema ─────────────────────────────────────────────────────────────

def test_fix_metadata_json_schema_shape():
    schema = fix_metadata_json_schema()
    assert schema["type"] == "object"
    assert "intent" in schema["properties"]
    assert "allowed_action_classes" in schema["properties"]


# ── Response parsing ────────────────────────────────────────────────────────

def test_parse_response_valid_json():
    payload = json.dumps({
        "intent": "prevents litho bridging",
        "allowed_action_classes": ["widen", "shift_orthogonal"],
        "forbidden_action_classes": [],
        "affected_layers": ["met2"],
        "branches": [],
        "notes": "",
    })
    meta = parse_response(_completion(payload))
    assert isinstance(meta, FixMetadata)
    assert meta.intent == "prevents litho bridging"
    assert meta.allowed_action_classes == ["widen", "shift_orthogonal"]
    assert meta.affected_layers == ["met2"]


def test_parse_response_strips_code_fences():
    """Some models wrap JSON in ```json fences despite the instruction."""
    payload = (
        "```json\n"
        '{"intent": "x", "allowed_action_classes": [], '
        '"forbidden_action_classes": [], "affected_layers": [], '
        '"branches": [], "notes": ""}\n'
        "```"
    )
    meta = parse_response(_completion(payload))
    assert meta.intent == "x"


def test_parse_response_missing_choices_raises():
    with pytest.raises(ExtractionError, match="unexpected response shape"):
        parse_response({"choices": []})


def test_parse_response_invalid_json_raises():
    bad = _completion("this is not json")
    with pytest.raises(ExtractionError) as exc:
        parse_response(bad)
    assert "not valid FixMetadata JSON" in str(exc.value)
    assert exc.value.raw == "this is not json"


# ── Extractor end-to-end with mock model ────────────────────────────────────

def test_extractor_invokes_model_with_messages_and_schema():
    calls: list[dict] = []

    def model_fn(**kwargs):
        calls.append(kwargs)
        return _completion(json.dumps({
            "intent": "ok",
            "allowed_action_classes": ["widen"],
            "forbidden_action_classes": [],
            "affected_layers": ["met2"],
            "branches": [],
            "notes": "",
        }))

    extractor = FixMetadataExtractor(model_fn=model_fn)
    chunk = _chunk("metal2 minimum spacing shall be 0.14 um.")
    meta = extractor.extract(chunk)

    assert isinstance(meta, FixMetadata)
    assert meta.intent == "ok"
    [call] = calls
    # Messages were built and forwarded.
    assert len(call["messages"]) == 2
    assert call["messages"][1]["content"].count("metal2 minimum spacing") == 1
    # Decoding is schema-constrained.
    assert call["response_format"]["type"] == "json_object"
    assert "intent" in call["response_format"]["schema"]["properties"]


def test_extract_many_iterates_first_chunk_per_code():
    calls: list[str] = []

    def model_fn(messages, **_):
        calls.append(messages[1]["content"])
        return _completion(json.dumps({
            "intent": "ok", "allowed_action_classes": [],
            "forbidden_action_classes": [], "affected_layers": [],
            "branches": [], "notes": "",
        }))

    chunks = {
        "M2.S.1": [_chunk("first M2.S.1 chunk", code="M2.S.1"),
                   _chunk("second M2.S.1 chunk", code="M2.S.1")],
        "M2.W.1": [_chunk("M2.W.1 chunk", code="M2.W.1")],
    }
    extractor = FixMetadataExtractor(model_fn=model_fn)
    result = extractor.extract_many(chunks)

    assert set(result) == {"M2.S.1", "M2.W.1"}
    # Only the first chunk per code was sent.
    assert any("first M2.S.1 chunk" in c for c in calls)
    assert not any("second M2.S.1 chunk" in c for c in calls)


def test_extractor_propagates_validation_errors():
    def model_fn(**_):
        return _completion("{not valid json")

    extractor = FixMetadataExtractor(model_fn=model_fn)
    with pytest.raises(ExtractionError):
        extractor.extract(_chunk("test"))


def test_build_messages_includes_deck_block_when_provided():
    """When deck_block is passed, the user prompt embeds it in a fenced code block."""
    chunk = _chunk("Min. metal-2 space < 0.28")
    msgs = build_messages(
        chunk,
        deck_block=(
            "M2.S.1 { @ Min. metal-2 space < 0.28\n"
            "  EXT MET2 < 0.28 ABUT < 90 SINGULAR REGION\n"
            "}"
        ),
        deck_dialect="svrf",
    )
    user_text = msgs[1]["content"]
    assert "Deck body (svrf):" in user_text
    assert "EXT MET2 < 0.28" in user_text
    assert "```" in user_text          # code-fence boundaries


def test_build_messages_omits_deck_block_section_when_not_provided():
    msgs = build_messages(_chunk("desc"))
    assert "Deck body" not in msgs[1]["content"]


def test_extract_forwards_deck_block_into_messages():
    """End-to-end: extract() passes deck_block through to build_messages and
    on into the model call."""
    captured: dict = {}
    def model_fn(**kwargs):
        captured.update(kwargs)
        return _completion(json.dumps({
            "intent": "x", "allowed_action_classes": [],
            "forbidden_action_classes": [], "affected_layers": [],
            "branches": [], "notes": "",
        }))
    extractor = FixMetadataExtractor(model_fn=model_fn)
    extractor.extract(_chunk("desc"), deck_block="DECK_BODY_SENTINEL", deck_dialect="svrf")
    assert "DECK_BODY_SENTINEL" in captured["messages"][1]["content"]


def test_from_ollama_posts_chat_request_and_parses_response(monkeypatch):
    """Verify the Ollama adapter formats the HTTP body correctly and
    surfaces a chat-completion-shaped response back through the extractor.

    We monkey-patch ``urllib.request.urlopen`` so the test runs offline.
    """
    import io
    import json as _json
    import urllib.request

    captured: dict = {}

    class _FakeResponse:
        def __init__(self, body: bytes): self._body = body
        def __enter__(self):  return self
        def __exit__(self, *exc): return False
        def read(self): return self._body

    def fake_urlopen(req, timeout=None):
        captured["url"]    = req.full_url
        captured["method"] = req.get_method()
        captured["body"]   = _json.loads(req.data.decode("utf-8"))
        # Mimic Ollama /api/chat response shape.
        payload = {
            "model":   "qwen2.5-coder:3b",
            "message": {"role": "assistant", "content": _json.dumps({
                "intent": "from ollama",
                "allowed_action_classes": ["widen"],
                "forbidden_action_classes": [],
                "affected_layers": ["met2"],
                "branches": [],
                "notes": "",
            })},
        }
        return _FakeResponse(_json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    extractor = FixMetadataExtractor.from_ollama(
        "qwen2.5-coder:3b", host="http://localhost:11434",
    )
    meta = extractor.extract(_chunk("Min. met2 space < 0.14"))

    assert captured["url"].endswith("/api/chat")
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "qwen2.5-coder:3b"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    # System message is forwarded as a chat message (Ollama accepts role=system).
    assert any(m["role"] == "system" for m in captured["body"]["messages"])
    # And our extracted FixMetadata round-tripped from the mock response.
    assert isinstance(meta, FixMetadata)
    assert meta.intent == "from ollama"
    assert meta.allowed_action_classes == ["widen"]


def test_extract_many_forwards_per_code_deck_blocks():
    """extract_many should pull each code's deck_block from the optional map."""
    seen_bodies: list[str] = []
    def model_fn(messages, **_):
        seen_bodies.append(messages[1]["content"])
        return _completion(json.dumps({
            "intent": "ok", "allowed_action_classes": [],
            "forbidden_action_classes": [], "affected_layers": [],
            "branches": [], "notes": "",
        }))
    extractor = FixMetadataExtractor(model_fn=model_fn)
    chunks = {
        "A.1": [_chunk("desc A", code="A.1")],
        "B.1": [_chunk("desc B", code="B.1")],
    }
    extractor.extract_many(chunks, deck_blocks={
        "A.1": "BODY_FOR_A", "B.1": "BODY_FOR_B",
    }, deck_dialect="svrf")
    blob = "\n".join(seen_bodies)
    assert "BODY_FOR_A" in blob
    assert "BODY_FOR_B" in blob


def test_extractor_config_forwarded_to_model():
    captured: dict = {}

    def model_fn(**kwargs):
        captured.update(kwargs)
        return _completion(json.dumps({
            "intent": "", "allowed_action_classes": [],
            "forbidden_action_classes": [], "affected_layers": [],
            "branches": [], "notes": "",
        }))

    from lithos_ingest.extractor import ExtractorConfig
    cfg = ExtractorConfig(temperature=0.3, max_tokens=128, seed=42)
    extractor = FixMetadataExtractor(model_fn=model_fn, config=cfg)
    extractor.extract(_chunk("x"))
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"]  == 128
    assert captured["seed"]        == 42
