"""Layer 3 verifier tests — mock LLMClient, no real Gemini calls.

The cascade depends on Layer 3 returning EITHER a valid `VerifierResponse`
or `None` on any failure. Tests pin every failure path so a degraded
Gemini upstream can't crash the cascade in production.

Scenarios:
  1. Happy AUTO_CORRECT — Gemini returns a clean AUTO_CORRECT JSON
  2. Happy SUGGEST — Gemini returns SUGGEST with top-3 surfaced
  3. Happy BLOCK — Gemini decides no plausible match
  4. Transport failure (timeout / network) -> None
  5. Malformed JSON -> None (cascade falls back to L2)
  6. Schema-violating verdict (e.g. confidence > 1.0) -> None
  7. Missing required field in JSON -> None
  8. StubVerifier returns its configured response
  9. StubVerifier returns None when constructed with no response
 10. get_verifier() falls back to stub when GEMINI_API_KEY missing
 11. get_verifier() returns stub when provider == "stub"
 12. get_verifier() is singleton
 13. Prompt builder includes registry + candidates + reasoning
 14. layer3() wrapper returns (response, elapsed_ms) tuple
"""

from __future__ import annotations

import json

import pytest

from sentinel.layer3 import (
    GeminiFlashVerifier,
    LLMClient,
    StubVerifier,
    Verifier,
    VerifierResponse,
    VerifierSuggestion,
    _normalize_gemini_data,
    _strip_fences,
    get_verifier,
    layer3,
    reset_verifier_cache,
)
from sentinel.schemas import DetectRequest, Tool, ToolRegistry


# ============================================================================
# Test scaffolding — mock LLMClient with configurable behavior
# ============================================================================


class MockLLMClient:
    """LLMClient that returns canned responses or raises canned errors.

    `responses` is a list — each `generate()` call pops the head and either
    returns it (if str) or raises it (if Exception)."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("MockLLMClient ran out of canned responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _two_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        tools=(
            Tool(name="web_search", description="Search the web", required_args=("query",)),
            Tool(name="mcp__lint_check", description="Lint", required_args=("file",)),
        )
    )


def _phantom_req() -> DetectRequest:
    return DetectRequest(
        tool_name="search_the_internet",
        tool_input={"query": "phantom tool calls in LLM agents"},
        agent_reasoning="I want to find recent research on agent reliability.",
    )


def _top_candidates(registry: ToolRegistry) -> list[tuple[float, Tool]]:
    return [(0.81, registry.tools[0]), (0.34, registry.tools[1])]


# ============================================================================
# Protocol conformance
# ============================================================================


def test_mock_llm_client_satisfies_protocol() -> None:
    assert isinstance(MockLLMClient([]), LLMClient)


def test_stub_verifier_satisfies_verifier_protocol() -> None:
    assert isinstance(StubVerifier(), Verifier)


def test_gemini_verifier_satisfies_verifier_protocol() -> None:
    client = MockLLMClient([])
    assert isinstance(GeminiFlashVerifier(client=client), Verifier)


# ============================================================================
# Happy path scenarios
# ============================================================================


def test_happy_auto_correct() -> None:
    """Gemini returns a clean AUTO_CORRECT verdict."""
    canned = json.dumps(
        {
            "verdict": "AUTO_CORRECT",
            "confidence": 0.91,
            "reason": "Tool 'search_the_internet' not in registry. Use 'web_search' (sim 0.81).",
            "suggestion": {
                "tool_name": "web_search",
                "rationale": "top-1 by cosine, schema-compatible with required_args=[query]",
            },
        }
    )
    client = MockLLMClient([canned])
    verifier = GeminiFlashVerifier(client=client)
    registry = _two_tool_registry()
    req = _phantom_req()

    response = verifier.verify(req, registry, _top_candidates(registry))

    assert response is not None
    assert response.verdict == "AUTO_CORRECT"
    assert response.confidence == 0.91
    assert response.suggestion is not None
    assert response.suggestion.tool_name == "web_search"


def test_happy_suggest() -> None:
    canned = json.dumps(
        {
            "verdict": "SUGGEST",
            "confidence": 0.72,
            "reason": "Ambiguous; closest is 'web_search' (sim 0.72). Confirm or revise.",
            "suggestion": {
                "tool_name": "web_search",
                "rationale": "top-1 of 3 candidates, sim 0.72",
            },
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([canned]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is not None
    assert response.verdict == "SUGGEST"
    assert 0.6 <= response.confidence < 0.85


def test_happy_block() -> None:
    canned = json.dumps(
        {
            "verdict": "BLOCK",
            "confidence": 0.85,
            "reason": "Tool 'phantom_x' not in registry; no plausible match. Revise plan.",
            "suggestion": None,
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([canned]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is not None
    assert response.verdict == "BLOCK"
    assert response.suggestion is None


# ============================================================================
# Failure paths — every one MUST return None (caller degrades to L2)
# ============================================================================


def test_transport_failure_returns_none() -> None:
    """Network timeout / connection error -> None."""
    client = MockLLMClient([TimeoutError("Gemini API timeout after 2.0s")])
    verifier = GeminiFlashVerifier(client=client)
    registry = _two_tool_registry()

    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is None
    assert len(client.calls) == 1  # the call WAS attempted


def test_malformed_json_returns_none() -> None:
    """Gemini returns non-JSON garbage -> None."""
    verifier = GeminiFlashVerifier(client=MockLLMClient(["this is not json {{{"]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


def test_schema_violating_confidence_returns_none() -> None:
    """confidence > 1.0 violates Pydantic schema -> None."""
    bad = json.dumps(
        {
            "verdict": "AUTO_CORRECT",
            "confidence": 1.5,  # out of [0, 1]
            "reason": "should be rejected",
            "suggestion": {"tool_name": "web_search", "rationale": "nope"},
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([bad]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


def test_unknown_verdict_literal_returns_none() -> None:
    """A verdict outside the 4-value literal is rejected."""
    bad = json.dumps(
        {
            "verdict": "MAYBE",
            "confidence": 0.7,
            "reason": "unknown verdict literal",
            "suggestion": None,
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([bad]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


def test_missing_required_field_returns_none() -> None:
    """Gemini omits the `reason` field — VerifierResponse.reason is required."""
    bad = json.dumps(
        {
            "verdict": "AUTO_CORRECT",
            "confidence": 0.9,
            "suggestion": {"tool_name": "web_search", "rationale": "no reason field"},
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([bad]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


def test_reason_over_240_chars_returns_none() -> None:
    """Schema caps reason at 240 chars — over-long reasons rejected."""
    bad = json.dumps(
        {
            "verdict": "BLOCK",
            "confidence": 0.9,
            "reason": "x" * 241,
            "suggestion": None,
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([bad]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


# ============================================================================
# Gemini quirk guards — extra fields + markdown fences + empty suggestion
# ============================================================================


def test_extra_fields_are_tolerated() -> None:
    """gemini-2.5-flash appends 'explanation' / 'chain_of_thought' — must not fail."""
    canned = json.dumps(
        {
            "verdict": "AUTO_CORRECT",
            "confidence": 0.92,
            "reason": "Tool 'search_the_internet' not in registry. Use 'web_search'.",
            "suggestion": {"tool_name": "web_search", "rationale": "top-1 semantic match"},
            "explanation": "I chose web_search because it is the closest semantic match.",
            "chain_of_thought": "Step 1: ...",
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([canned]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is not None
    assert response.verdict == "AUTO_CORRECT"
    assert response.confidence == 0.92


def test_markdown_fences_are_stripped() -> None:
    """Gemini sometimes wraps JSON in ```json ... ``` despite application/json."""
    inner = json.dumps(
        {
            "verdict": "BLOCK",
            "confidence": 0.88,
            "reason": "No plausible match. Revise your plan.",
            "suggestion": None,
        }
    )
    fenced = f"```json\n{inner}\n```"
    verifier = GeminiFlashVerifier(client=MockLLMClient([fenced]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is not None
    assert response.verdict == "BLOCK"


def test_empty_suggestion_object_is_normalized_to_none() -> None:
    """Gemini returns suggestion: {} instead of null — must be treated as None."""
    canned = json.dumps(
        {
            "verdict": "BLOCK",
            "confidence": 0.88,
            "reason": "No plausible match. Revise your plan.",
            "suggestion": {},
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([canned]))
    registry = _two_tool_registry()
    response = verifier.verify(_phantom_req(), registry, _top_candidates(registry))

    assert response is not None
    assert response.suggestion is None


def test_strip_fences_plain_json_unchanged() -> None:
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


def test_strip_fences_with_json_fence() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    assert _strip_fences(raw) == '{"a": 1}'


def test_strip_fences_with_plain_fence() -> None:
    raw = "```\n{\"a\": 1}\n```"
    assert _strip_fences(raw) == '{"a": 1}'


def test_normalize_gemini_data_converts_empty_suggestion() -> None:
    data = {"verdict": "BLOCK", "suggestion": {}}
    result = _normalize_gemini_data(data)
    assert isinstance(result, dict) and result["suggestion"] is None


def test_normalize_gemini_data_leaves_valid_suggestion_alone() -> None:
    data = {"suggestion": {"tool_name": "web_search", "rationale": "top-1"}}
    result = _normalize_gemini_data(data)
    assert isinstance(result, dict) and result["suggestion"] == {"tool_name": "web_search", "rationale": "top-1"}


# ============================================================================
# StubVerifier
# ============================================================================


def test_stub_verifier_returns_configured_response() -> None:
    canned = VerifierResponse(
        verdict="AUTO_CORRECT",
        confidence=0.9,
        reason="stub fixed response",
        suggestion=VerifierSuggestion(tool_name="web_search", rationale="stub says so"),
    )
    stub = StubVerifier(response=canned)
    registry = _two_tool_registry()
    response = stub.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is canned
    assert stub.call_count == 1


def test_stub_verifier_returns_none_by_default() -> None:
    """A StubVerifier with no preconfigured response always returns None.
    This is the boot-safe fallback mode the factory uses when GEMINI_API_KEY
    is missing — every verify() returns None so the cascade degrades to L2."""
    stub = StubVerifier()
    registry = _two_tool_registry()
    response = stub.verify(_phantom_req(), registry, _top_candidates(registry))
    assert response is None


def test_stub_verifier_records_call_context() -> None:
    """Stub captures what it was called with — useful for assertion in
    higher-level cascade tests."""
    stub = StubVerifier()
    registry = _two_tool_registry()
    req = _phantom_req()
    stub.verify(req, registry, _top_candidates(registry))

    assert stub.last_call is not None
    assert stub.last_call["tool_name"] == req.tool_name
    assert stub.last_call["registry_size"] == 2
    assert stub.last_call["candidates"][0][1] == "web_search"


# ============================================================================
# get_verifier() factory
# ============================================================================


def test_get_verifier_falls_back_to_stub_when_no_gemini_key(monkeypatch) -> None:
    """Constitution Principle IV: daemon must boot even with no Gemini key.
    Missing key -> StubVerifier(response=None) -> cascade degrades."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SENTINEL_VERIFIER_PROVIDER", "gemini")
    reset_verifier_cache()

    v = get_verifier()
    assert isinstance(v, StubVerifier)
    assert v.response is None  # always-None mode


def test_get_verifier_returns_stub_when_provider_is_stub(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_VERIFIER_PROVIDER", "stub")
    reset_verifier_cache()
    v = get_verifier()
    assert isinstance(v, StubVerifier)


def test_get_verifier_returns_stub_on_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_VERIFIER_PROVIDER", "claude-haiku-eventually")
    reset_verifier_cache()
    v = get_verifier()
    assert isinstance(v, StubVerifier)


def test_get_verifier_is_singleton(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_VERIFIER_PROVIDER", "stub")
    reset_verifier_cache()
    a = get_verifier()
    b = get_verifier()
    assert a is b


# ============================================================================
# Prompt builder
# ============================================================================


def test_prompt_includes_phantom_call_and_candidates() -> None:
    """The rendered prompt must surface every input field Gemini needs to
    follow the decision rules — if any is missing the verifier flies blind."""
    verifier = GeminiFlashVerifier(client=MockLLMClient([]))
    registry = _two_tool_registry()
    req = DetectRequest(
        tool_name="phantom_x",
        tool_input={"query": "foo"},
        agent_reasoning="user wants web search",
    )
    prompt = verifier._build_prompt(req, registry, _top_candidates(registry))

    assert "phantom_x" in prompt
    assert "web_search" in prompt  # registry tool
    assert "mcp__lint_check" in prompt  # registry tool
    assert "0.810" in prompt or "0.81" in prompt  # candidate sim
    assert "user wants web search" in prompt  # agent reasoning
    assert "query" in prompt  # tool_input keys leak through json.dumps


def test_prompt_handles_missing_reasoning() -> None:
    """No agent_reasoning should still produce a valid prompt."""
    verifier = GeminiFlashVerifier(client=MockLLMClient([]))
    registry = _two_tool_registry()
    req = DetectRequest(tool_name="phantom_x", tool_input={})  # no reasoning
    prompt = verifier._build_prompt(req, registry, _top_candidates(registry))
    assert "(none provided)" in prompt


def test_prompt_handles_empty_registry() -> None:
    verifier = GeminiFlashVerifier(client=MockLLMClient([]))
    registry = ToolRegistry()
    req = _phantom_req()
    prompt = verifier._build_prompt(req, registry, [])
    assert "(empty registry)" in prompt
    assert "(no candidates)" in prompt


# ============================================================================
# layer3() wrapper — returns (response, elapsed_ms)
# ============================================================================


def test_layer3_wrapper_returns_response_and_elapsed_ms() -> None:
    canned = json.dumps(
        {
            "verdict": "SUGGEST",
            "confidence": 0.7,
            "reason": "ambiguous",
            "suggestion": {"tool_name": "web_search", "rationale": "top-1"},
        }
    )
    verifier = GeminiFlashVerifier(client=MockLLMClient([canned]))
    registry = _two_tool_registry()

    response, elapsed_ms = layer3(_phantom_req(), registry, _top_candidates(registry), verifier)

    assert response is not None
    assert response.verdict == "SUGGEST"
    assert elapsed_ms >= 0.0


def test_layer3_wrapper_returns_none_on_verifier_failure() -> None:
    verifier = GeminiFlashVerifier(client=MockLLMClient([TimeoutError("boom")]))
    registry = _two_tool_registry()
    response, elapsed_ms = layer3(_phantom_req(), registry, _top_candidates(registry), verifier)
    assert response is None
    assert elapsed_ms >= 0.0


# ============================================================================
# Fixture: reset singleton between tests
# ============================================================================


@pytest.fixture(autouse=True)
def _isolate_verifier_singleton() -> None:
    reset_verifier_cache()
    yield
    reset_verifier_cache()
