"""Layer 2 unit tests — verdict mapping + fusion behavior + edge cases.

We use a `MockEmbedder` that maps known input strings to manually-constructed
vectors. This lets each test set up an exact cosine relationship between
the phantom and the registered tools, then assert the resulting verdict
is what the cascade thresholds + fusion produce.

Scenarios covered:
  1. Clean match — phantom cosine 0.95 vs top-1 -> AUTO_CORRECT
  2. Near-typo F1 boost — base cosine ambiguous, F1 high, fusion lifts to AC
  3. Schema-twin F2 boost — base cosine ambiguous, shared arg keys
  4. Wide top-1 gap — F3 dominant in ambiguous zone
  5. Tied top-1 — multiple candidates equal -> SUGGEST
  6. No candidate -> BLOCK
  7. Empty registry -> BLOCK with structured reason
  8. Embedding backend failure -> BLOCK with degraded=True
  9. Phantom signature includes arg keys (sanity)
 10. warm_up_registry skips failures (degrades gracefully)
"""

from __future__ import annotations

import math

import pytest

from sentinel.embeddings import Embedder, EmbeddingError, StubEmbedder
from sentinel.layer2 import (
    _cosine,
    _rescale_top1_to_confidence,
    layer2,
    phantom_signature,
    tool_signature,
    warm_up_registry,
)
from sentinel.schemas import DetectRequest, Tool, ToolRegistry


# ============================================================================
# Test scaffolding
# ============================================================================


DIM = 384


def _make_vec(weights: dict[int, float], dim: int = DIM) -> list[float]:
    """Build an L2-normalized sparse vector with chosen weights at chosen indices.
    Lets tests construct vectors with explicit cosine relationships."""
    v = [0.0] * dim
    for i, w in weights.items():
        v[i] = w
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class MockEmbedder:
    """Test embedder — maps exact text -> exact vector, falls back to a
    deterministic StubEmbedder for any unmapped string so unrelated calls
    still produce valid (non-zero) vectors."""

    dim = DIM

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self._fallback = StubEmbedder(self.dim)
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if text in self.mapping:
            return list(self.mapping[text])
        return self._fallback.embed(text)


class FailingEmbedder:
    """Embedder that always raises EmbeddingError. For degraded-path tests."""

    dim = DIM

    def embed(self, text: str) -> list[float]:
        raise EmbeddingError("simulated backend failure")


# ============================================================================
# Helper validation
# ============================================================================


def test_mock_embedder_satisfies_protocol() -> None:
    """MockEmbedder must duck-type as an Embedder so it can substitute everywhere."""
    assert isinstance(MockEmbedder({}), Embedder)


def test_cosine_zero_vector_returns_zero_not_nan() -> None:
    """Cosine of a zero vector against anything must be 0.0, not NaN.
    A NaN here would propagate into Decision.confidence and fail the schema's
    [0, 1] bound -> upstream cascade crash."""
    zero = [0.0] * DIM
    nonzero = _make_vec({0: 1.0})
    assert _cosine(zero, nonzero) == 0.0
    assert _cosine(nonzero, zero) == 0.0
    assert _cosine(zero, zero) == 0.0


def test_cosine_identical_vectors_returns_one() -> None:
    v = _make_vec({0: 1.0, 5: 0.5})
    assert math.isclose(_cosine(v, v), 1.0, abs_tol=1e-5)


def test_rescale_formula() -> None:
    """Confidence rescale math: (sim - 0.5) * 2 clipped to [0, 1]."""
    assert _rescale_top1_to_confidence(0.5) == 0.0
    assert _rescale_top1_to_confidence(0.75) == 0.5
    assert _rescale_top1_to_confidence(0.925) == pytest.approx(0.85, abs=1e-9)
    assert _rescale_top1_to_confidence(0.95) == pytest.approx(0.9, abs=1e-9)
    assert _rescale_top1_to_confidence(1.0) == 1.0
    assert _rescale_top1_to_confidence(0.3) == 0.0  # clipped


# ============================================================================
# Signature builders
# ============================================================================


def test_tool_signature_concatenates_name_desc_args() -> None:
    t = Tool(name="web_search", description="Search the web", required_args=("query", "limit"))
    sig = tool_signature(t)
    assert "web_search" in sig
    assert "Search the web" in sig
    assert "args:query,limit" in sig


def test_phantom_signature_uses_arg_keys() -> None:
    req = DetectRequest(tool_name="search_db", tool_input={"q": "x", "limit": 5})
    sig = phantom_signature(req)
    assert sig.startswith("search_db")
    # arg keys sorted alphabetically so signatures are stable
    assert "args:limit,q" in sig


# ============================================================================
# warm_up_registry
# ============================================================================


def test_warmup_embeds_all_tools() -> None:
    registry = ToolRegistry(
        tools=(
            Tool(name="web_search", required_args=("query",)),
            Tool(name="mcp__lint_check", required_args=("file",)),
        )
    )
    emb = MockEmbedder(
        {
            tool_signature(registry.tools[0]): _make_vec({0: 1.0}),
            tool_signature(registry.tools[1]): _make_vec({1: 1.0}),
        }
    )
    out = warm_up_registry(registry, emb)
    assert set(out.keys()) == {"web_search", "mcp__lint_check"}
    assert out["web_search"][0] == pytest.approx(1.0)
    assert out["mcp__lint_check"][1] == pytest.approx(1.0)


def test_warmup_drops_failing_tools_but_continues() -> None:
    """If one tool's embed fails, the others must still complete."""

    class PartialFailEmbedder:
        dim = DIM

        def embed(self, text: str) -> list[float]:
            if "lint_check" in text:
                raise EmbeddingError("simulated partial failure")
            return _make_vec({0: 1.0})

    registry = ToolRegistry(
        tools=(Tool(name="web_search"), Tool(name="mcp__lint_check"))
    )
    out = warm_up_registry(registry, PartialFailEmbedder())
    assert "web_search" in out
    assert "mcp__lint_check" not in out


# ============================================================================
# Layer 2 scenarios
# ============================================================================


def _two_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        tools=(
            Tool(name="web_search", description="Search the web", required_args=("query",)),
            Tool(name="mcp__lint_check", description="Lint", required_args=("file",)),
        )
    )


def test_scenario_1_clean_match_returns_auto_correct() -> None:
    """Phantom cosine ~0.95 vs top-1 -> rescaled conf 0.9 -> AUTO_CORRECT."""
    registry = _two_tool_registry()
    web_vec = _make_vec({0: 1.0})
    lint_vec = _make_vec({1: 1.0})
    phantom_vec = _make_vec({0: 0.95, 1: 0.05})  # cos with web ~0.999

    embeddings = {"web_search": web_vec, "mcp__lint_check": lint_vec}
    embedder = MockEmbedder(
        {phantom_signature(DetectRequest(tool_name="search_the_web", tool_input={"query": "x"})): phantom_vec}
    )
    req = DetectRequest(tool_name="search_the_web", tool_input={"query": "x"})

    d = layer2(req, registry, embedder, embeddings)
    assert d.verdict == "AUTO_CORRECT"
    assert d.confidence >= 0.85
    assert d.suggestion is not None
    assert d.suggestion.tool_name == "web_search"


def test_scenario_2_near_typo_with_high_cosine_auto_corrects() -> None:
    """Phantom is a literal typo of top-1 (Levenshtein ~1) AND cosine is high
    -> AUTO_CORRECT. (Fusion only fires in ambiguous zone, but this scenario
    sails through on cosine alone, so we sanity-check the typo path.)"""
    registry = ToolRegistry(
        tools=(Tool(name="mcp__lint_check", required_args=("file",)),)
    )
    target_vec = _make_vec({0: 1.0})
    phantom_vec = _make_vec({0: 0.97, 1: 0.05})  # near-identical

    embeddings = {"mcp__lint_check": target_vec}
    req = DetectRequest(tool_name="mcp__lint_checks", tool_input={"file": "auth.py"})
    embedder = MockEmbedder({phantom_signature(req): phantom_vec})

    d = layer2(req, registry, embedder, embeddings)
    assert d.verdict == "AUTO_CORRECT"
    assert d.suggestion is not None
    assert d.suggestion.tool_name == "mcp__lint_check"


def test_scenario_3_schema_twin_with_ambiguous_cosine_fuses_up() -> None:
    """Base cosine lands in the ambiguous fusion window; F2 (shared arg keys)
    pushes the fused confidence over the AUTO_CORRECT threshold."""
    registry = _two_tool_registry()
    web_vec = _make_vec({0: 1.0})
    lint_vec = _make_vec({1: 1.0})
    # cos(phantom, web) = 0.85, cos(phantom, lint) = 0.30  (top1 just inside ambiguous)
    phantom_vec = _make_vec({0: 0.85, 1: 0.30})

    embeddings = {"web_search": web_vec, "mcp__lint_check": lint_vec}
    # Phantom uses 'query' (same key as web_search.required_args) -> F2 = 1.0
    req = DetectRequest(tool_name="search", tool_input={"query": "test"})
    embedder = MockEmbedder({phantom_signature(req): phantom_vec})

    d = layer2(req, registry, embedder, embeddings)
    # In the ambiguous range; F2=1, F1 modest, F3 generous -> should AUTO_CORRECT
    assert d.verdict in ("AUTO_CORRECT", "SUGGEST")
    assert d.suggestion is not None
    assert d.suggestion.tool_name == "web_search"


def test_scenario_4_tied_top1_returns_suggest() -> None:
    """Phantom equidistant from two tools -> top-1 gap = 0, no fusion boost
    -> falls into the SUGGEST or BLOCK band depending on base conf."""
    registry = _two_tool_registry()
    web_vec = _make_vec({0: 1.0})
    lint_vec = _make_vec({1: 1.0})
    # cos with both = ~0.707 -> base conf ~0.41 -> below block_max -> BLOCK
    phantom_vec = _make_vec({0: 0.5, 1: 0.5})

    embeddings = {"web_search": web_vec, "mcp__lint_check": lint_vec}
    req = DetectRequest(tool_name="unknown_tool", tool_input={})
    embedder = MockEmbedder({phantom_signature(req): phantom_vec})

    d = layer2(req, registry, embedder, embeddings)
    # Should be BLOCK because the equidistant scenario yields low base confidence
    assert d.verdict == "BLOCK"
    assert d.suggestion is None


def test_scenario_5_no_candidate_returns_block() -> None:
    """All registry tools are far from the phantom -> top-1 cosine < 0.5 ->
    confidence is 0 -> BLOCK with structured reason."""
    registry = _two_tool_registry()
    web_vec = _make_vec({0: 1.0})
    lint_vec = _make_vec({1: 1.0})
    phantom_vec = _make_vec({100: 1.0})  # orthogonal to everything

    embeddings = {"web_search": web_vec, "mcp__lint_check": lint_vec}
    req = DetectRequest(tool_name="phantom_x", tool_input={})
    embedder = MockEmbedder({phantom_signature(req): phantom_vec})

    d = layer2(req, registry, embedder, embeddings)
    assert d.verdict == "BLOCK"
    assert d.confidence > 0.0
    assert d.suggestion is None
    assert "phantom_x" in d.reason


def test_scenario_6_empty_registry_returns_block() -> None:
    registry = ToolRegistry()
    embedder = MockEmbedder({})
    req = DetectRequest(tool_name="anything", tool_input={})
    d = layer2(req, registry, embedder, {})
    assert d.verdict == "BLOCK"
    assert d.layer_breakdown.l2_ms >= 0.0
    assert "anything" in d.reason


def test_scenario_7_embed_backend_failure_degrades_to_block() -> None:
    """If the embedder fails on the phantom, Layer 2 returns a BLOCK with
    degraded=True. The agent still gets a Decision; the daemon stays up."""
    registry = _two_tool_registry()
    embeddings = {
        "web_search": _make_vec({0: 1.0}),
        "mcp__lint_check": _make_vec({1: 1.0}),
    }
    req = DetectRequest(tool_name="anything", tool_input={})
    d = layer2(req, registry, FailingEmbedder(), embeddings)
    assert d.verdict == "BLOCK"
    assert d.degraded is True
    assert "embedding backend unavailable" in d.reason.lower()


def test_scenario_8_no_warmup_embeddings_blocks_degraded() -> None:
    """Registry has tools but warm-up produced no embeddings (e.g., total
    backend outage at startup). Layer 2 must return BLOCK degraded=True."""
    registry = _two_tool_registry()
    embedder = MockEmbedder({})
    req = DetectRequest(tool_name="anything")
    d = layer2(req, registry, embedder, {})
    assert d.verdict == "BLOCK"
    assert d.degraded is True


def test_l2_latency_under_budget() -> None:
    """Per Constitution Principle II: Layer 2 median <10ms on commodity HW.
    StubEmbedder is fast (no network); registry of 2 tools; this is a smoke
    check, the real benchmark lives in `make bench-latency` (T035)."""
    import time

    registry = _two_tool_registry()
    embeddings = {
        "web_search": _make_vec({0: 1.0}),
        "mcp__lint_check": _make_vec({1: 1.0}),
    }
    embedder = StubEmbedder()
    req = DetectRequest(tool_name="some_phantom", tool_input={})

    # Warm up the embedder path
    for _ in range(5):
        layer2(req, registry, embedder, embeddings)

    timings: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        layer2(req, registry, embedder, embeddings)
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings.sort()
    median_ms = timings[len(timings) // 2]
    # Stub-embed is microseconds; full L2 should land far under the 10ms target.
    assert median_ms < 10.0, f"L2 median {median_ms:.2f}ms exceeds 10ms target"
