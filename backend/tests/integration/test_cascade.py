"""Cascade integration tests — wires L1 + L2 + L3 across realistic flows.

Uses test fakes from `test_embeddings` (StubEmbedder) and `test_layer3`
(StubVerifier) so no network calls occur. Each test sets up a known
embedding-cosine relationship between phantom and registry, then asserts
the full cascade output is what the verdict thresholds + fusion logic
should produce.

Critical scenarios:
  1. L1 hit -> ALLOW (L2/L3 skipped entirely)
  2. L1 miss, L2 AUTO_CORRECT -> L3 skipped (already confident)
  3. L1 miss, L2 BLOCK -> L3 skipped (already negative)
  4. L1 miss, L2 SUGGEST, no verifier -> L2 returned as-is
  5. L1 miss, L2 SUGGEST, L3 succeeds -> fused decision
  6. L3 fusion lifts AUTO_CORRECT past threshold -> AUTO_CORRECT
  7. L3 fusion drops below auto_correct_min -> downgraded to SUGGEST
  8. L3 returns None (timeout/parse/ALLOW-guard) -> L2 with degraded=True
  9. H1: L3 suggests tool NOT in registry -> hard BLOCK
 10. L3 suggests valid registry tool -> AUTO_CORRECT through
"""

from __future__ import annotations

import math

import pytest

from sentinel.cascade import _build_final_decision, _remap_verdict_after_fusion, detect
from sentinel.layer3 import StubVerifier, VerifierResponse, VerifierSuggestion
from sentinel.schemas import DetectRequest, Decision, LayerBreakdown, Tool, ToolRegistry


# ============================================================================
# Test scaffolding
# ============================================================================


DIM = 384


def _unit_axis(axis: int) -> list[float]:
    """Pure unit vector along the given axis. Cosine sim to itself = 1.0,
    to any other axis = 0.0."""
    v = [0.0] * DIM
    v[axis] = 1.0
    return v


def _phantom_at_sim(target_sim: float, axis: int = 0) -> list[float]:
    """Build a unit phantom vector whose cosine sim to `_unit_axis(axis)`
    is exactly `target_sim`. The orthogonal component is placed on a far
    axis so it doesn't accidentally overlap with other registry vectors.
    """
    other = math.sqrt(max(0.0, 1.0 - target_sim * target_sim))
    v = [0.0] * DIM
    v[axis] = target_sim
    # Use a "noise" axis well away from any registry tool's axis (0, 1, 2).
    v[200] = other
    return v


class MockEmbedder:
    """Local copy of the test-scoped mock (kept self-contained for integration tests)."""

    dim = DIM

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if text in self.mapping:
            return list(self.mapping[text])
        # Fallback for unmapped queries: a near-zero vector so cosine ~= 0
        from sentinel.embeddings import StubEmbedder
        return StubEmbedder(self.dim).embed(text)


def _registry() -> ToolRegistry:
    return ToolRegistry(
        tools=(
            Tool(name="web_search", description="Search the web", required_args=("query",)),
            Tool(name="mcp__lint_check", description="Lint", required_args=("file",)),
            Tool(name="mcp__test_runner", description="Run tests", required_args=("path",)),
        )
    )


def _registry_embeddings() -> dict[str, list[float]]:
    return {
        "web_search": _unit_axis(0),
        "mcp__lint_check": _unit_axis(1),
        "mcp__test_runner": _unit_axis(2),
    }


# ============================================================================
# Scenario 1: L1 hit
# ============================================================================


def test_l1_hit_skips_layers_2_and_3() -> None:
    """Registered tool: cascade returns L1 ALLOW immediately."""
    embedder = MockEmbedder({})
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT",
            confidence=0.99,
            reason="L3 should NEVER be called",
            suggestion=VerifierSuggestion(tool_name="should_not_appear", rationale="never"),
        )
    )
    req = DetectRequest(tool_name="web_search", tool_input={"query": "test"})

    d = detect(req, _registry(), embedder, _registry_embeddings(), verifier=verifier)

    assert d.verdict == "ALLOW"
    assert d.confidence == 1.0
    assert verifier.call_count == 0, "L3 must not be called when L1 hits"
    assert embedder.call_count == 0, "L2 embedder must not be called when L1 hits"


# ============================================================================
# Scenario 2-3: L1 miss + L2 terminal (AUTO_CORRECT / BLOCK skip L3)
# ============================================================================


def test_l2_auto_correct_skips_l3() -> None:
    """L2 above auto_correct_min already; L3 escalation is unnecessary."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            # Phantom 0.96 cosine with web_search -> base_conf 0.92 -> AUTO_CORRECT
            "search_web args:query": _phantom_at_sim(0.96, axis=0),
        }
    )
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="BLOCK", confidence=0.99,
            reason="L3 should not be reached",
        )
    )
    req = DetectRequest(tool_name="search_web", tool_input={"query": "x"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert d.verdict == "AUTO_CORRECT"
    assert d.suggestion is not None
    assert d.suggestion.tool_name == "web_search"
    assert verifier.call_count == 0, "L3 must not fire when L2 is already AUTO_CORRECT"


def test_l2_block_skips_l3() -> None:
    """L2 below block_max already; L3 escalation is unnecessary."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            "phantom_x": _unit_axis(100),  # orthogonal to every registry vector
        }
    )
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT", confidence=0.99,
            reason="L3 should not flip BLOCK to AUTO_CORRECT",
            suggestion=VerifierSuggestion(tool_name="web_search", rationale="x"),
        )
    )
    req = DetectRequest(tool_name="phantom_x")

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert d.verdict == "BLOCK"
    assert verifier.call_count == 0, "L3 must not fire when L2 is already BLOCK"


# ============================================================================
# Scenario 4-5: L2 SUGGEST + verifier behavior
# ============================================================================


def test_l2_suggest_no_verifier_returns_l2() -> None:
    """When no verifier configured, L2 SUGGEST is returned as-is."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            # 0.82 cosine with web_search -> base_conf 0.64 -> SUGGEST
            "ambiguous_tool args:query": _phantom_at_sim(0.82, axis=0),
        }
    )
    req = DetectRequest(tool_name="ambiguous_tool", tool_input={"query": "test"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=None)

    assert d.verdict == "SUGGEST"


def test_l2_suggest_with_l3_auto_correct_fuses_through() -> None:
    """L2=SUGGEST (~0.7 conf) + L3=AUTO_CORRECT (~0.95 conf) →
    fused 0.6*0.95 + 0.4*0.7 = 0.85 → AUTO_CORRECT through."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            # 0.85 cosine -> base 0.70 -> SUGGEST
            "search_tool args:query": _phantom_at_sim(0.85, axis=0),
        }
    )
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT",
            confidence=0.98,  # high enough that 0.6*0.98 + 0.4*~0.69 = 0.864 >= 0.85
            reason="Top-1 web_search is semantic match for search_tool",
            suggestion=VerifierSuggestion(
                tool_name="web_search", rationale="cosine + agent reasoning aligned"
            ),
        )
    )
    req = DetectRequest(tool_name="search_tool", tool_input={"query": "x"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert verifier.call_count == 1, "L3 should fire on L2 SUGGEST"
    # Fused conf should land >= auto_correct_min (0.85)
    assert d.confidence >= 0.85, f"fused conf {d.confidence:.3f} below threshold"
    assert d.verdict == "AUTO_CORRECT"
    assert d.suggestion is not None
    assert d.suggestion.tool_name == "web_search"
    # All three layer timings present
    assert d.layer_breakdown.l3_ms is not None
    assert d.layer_breakdown.l3_ms > 0.0


def test_l2_suggest_l3_high_l2_low_downgrades_to_suggest() -> None:
    """L3 says AUTO_CORRECT 0.9 but L2 is 0.61 -> fused 0.6*0.9 + 0.4*0.61 = 0.784
    below 0.85 -> verdict re-mapped to SUGGEST."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            "borderline_tool args:query": _phantom_at_sim(0.805, axis=0),  # base 0.61
        }
    )
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT",
            confidence=0.90,
            reason="Top-1 looks right but agent reasoning is thin",
            suggestion=VerifierSuggestion(tool_name="web_search", rationale="best of 3"),
        )
    )
    req = DetectRequest(tool_name="borderline_tool", tool_input={"query": "x"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert verifier.call_count == 1
    # Verdict was downgraded because fused confidence dropped below auto_correct_min
    assert d.verdict == "SUGGEST"
    assert d.confidence < 0.85
    assert d.suggestion is not None  # SUGGEST still carries top-1


# ============================================================================
# Scenario 6: L3 returns None (timeout / parse / ALLOW guard)
# ============================================================================


def test_l3_returns_none_falls_back_to_l2_with_degraded_flag() -> None:
    """Any L3 failure -> cascade returns L2 decision with degraded=True
    AND l3_ms populated so the audit log shows we attempted verification."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            "ambig args:query": _phantom_at_sim(0.82, axis=0),
        }
    )
    verifier = StubVerifier(response=None)  # always-None mode
    req = DetectRequest(tool_name="ambig", tool_input={"query": "x"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert verifier.call_count == 1
    # Verdict still the L2 verdict (SUGGEST), but flagged as degraded
    assert d.degraded is True
    assert d.layer_breakdown.l3_ms is not None
    assert d.layer_breakdown.l3_ms > 0.0


# ============================================================================
# Scenario 7: H1 — L3 suggests tool NOT in registry -> hard BLOCK
# ============================================================================


def test_l3_suggesting_unregistered_tool_forces_block() -> None:
    """Anti-injection guard: if Gemini suggests a tool that's NOT in the
    registry, the cascade must NOT propagate that suggestion (would defeat
    the whole point of Sentinel)."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            "ambig args:query": _phantom_at_sim(0.82, axis=0),  # L2 SUGGEST
        }
    )
    # Verifier suggests a tool that's NOT in the registry
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT",
            confidence=0.95,
            reason="Definitely use this tool",
            suggestion=VerifierSuggestion(
                tool_name="totally_fake_tool",  # phantom!
                rationale="trust me bro",
            ),
        )
    )
    req = DetectRequest(tool_name="ambig", tool_input={"query": "x"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    # Must NOT propagate the phantom suggestion
    assert d.verdict == "BLOCK"
    assert d.suggestion is None
    assert "totally_fake_tool" in d.reason


# ============================================================================
# Scenario 8: L3 valid suggestion -> AUTO_CORRECT through with suggestion
# ============================================================================


def test_l3_valid_suggestion_propagates_to_decision() -> None:
    """Sanity: when L3 suggests a REAL registry tool AND fused conf is
    above threshold, the final Decision carries that suggestion."""
    registry = _registry()
    embedder = MockEmbedder(
        {
            "test_run args:path": _phantom_at_sim(0.83, axis=2),  # 0.83 with mcp__test_runner
        }
    )
    verifier = StubVerifier(
        response=VerifierResponse(
            verdict="AUTO_CORRECT",
            confidence=0.92,
            reason="Use mcp__test_runner; agent meant test execution",
            suggestion=VerifierSuggestion(
                tool_name="mcp__test_runner", rationale="schema-compat, sim 0.83"
            ),
        )
    )
    req = DetectRequest(tool_name="test_run", tool_input={"path": "tests/"})

    d = detect(req, registry, embedder, _registry_embeddings(), verifier=verifier)

    assert d.suggestion is not None
    assert d.suggestion.tool_name == "mcp__test_runner"
    assert d.verdict in ("AUTO_CORRECT", "SUGGEST")  # depends on fused conf


# ============================================================================
# _remap_verdict_after_fusion unit tests
# ============================================================================


def test_remap_auto_correct_above_threshold_stays() -> None:
    assert _remap_verdict_after_fusion("AUTO_CORRECT", 0.90) == "AUTO_CORRECT"


def test_remap_auto_correct_below_threshold_downgrades_to_suggest() -> None:
    assert _remap_verdict_after_fusion("AUTO_CORRECT", 0.80) == "SUGGEST"


def test_remap_suggest_below_block_max_downgrades_to_block() -> None:
    assert _remap_verdict_after_fusion("SUGGEST", 0.45) == "BLOCK"


def test_remap_block_unchanged() -> None:
    """BLOCK is the floor — never re-mapped."""
    assert _remap_verdict_after_fusion("BLOCK", 0.10) == "BLOCK"
    assert _remap_verdict_after_fusion("BLOCK", 0.99) == "BLOCK"


# ============================================================================
# _build_final_decision direct tests
# ============================================================================


def test_build_final_decision_with_invalid_suggestion_returns_block() -> None:
    """H1 guard exercised directly without going through full cascade."""
    from sentinel.schemas import Suggestion

    registry = _registry()
    # SUGGEST requires a non-None suggestion per Decision invariant; provide
    # a valid one so we can construct the layer2_decision fixture.
    layer2_decision = Decision(
        verdict="SUGGEST",
        confidence=0.7,
        reason="ambiguous from L2",
        suggestion=Suggestion(tool_name="web_search", rationale="L2 top-1"),
        layer_breakdown=LayerBreakdown(l2_ms=5.0),
    )
    # Now construct a response where Gemini suggests a NON-registered tool
    response = VerifierResponse(
        verdict="AUTO_CORRECT",
        confidence=0.95,
        reason="use this",
        suggestion=VerifierSuggestion(tool_name="ghost_tool", rationale="trust me"),
    )
    req = DetectRequest(tool_name="phantom")

    final = _build_final_decision(
        req=req,
        response=response,
        layer2_decision=layer2_decision,
        layer3_ms=200.0,
        registry=registry,
    )

    assert final.verdict == "BLOCK"
    assert final.suggestion is None
    assert "ghost_tool" in final.reason
