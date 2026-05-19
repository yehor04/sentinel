"""Embedding-backend tests — respx-mocked HTTP, deterministic stub, cache.

The cascade depends on `get_embedder()` returning *something* even when:
- The Featherless API key is missing
- The Featherless API returns malformed JSON
- The Featherless endpoint times out
- The configured provider is unknown

These tests pin every one of those behaviors.
"""

from __future__ import annotations

import os

import httpx
import pytest

from sentinel.embeddings import (
    Embedder,
    EmbeddingError,
    FeatherlessEmbedder,
    SemanticStubEmbedder,
    StubEmbedder,
    get_embedder,
    reset_embedder_cache,
)


# ============================================================================
# StubEmbedder — pure, deterministic, no I/O
# ============================================================================


def test_stub_returns_correct_dim() -> None:
    e = StubEmbedder(dim=384)
    vec = e.embed("hello")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_stub_is_deterministic() -> None:
    """Same text => same vector. Critical for cache-correctness tests."""
    e = StubEmbedder()
    a = e.embed("web_search")
    b = e.embed("web_search")
    assert a == b


def test_stub_distinguishes_inputs() -> None:
    """Different text => different vector. (Sanity check that we aren't
    accidentally returning zeros for everything.)"""
    e = StubEmbedder()
    assert e.embed("web_search") != e.embed("mcp__lint_check")


def test_stub_satisfies_protocol() -> None:
    """StubEmbedder must satisfy the `Embedder` protocol so the cascade
    can take it interchangeably with FeatherlessEmbedder."""
    assert isinstance(StubEmbedder(), Embedder)


def test_stub_values_bounded() -> None:
    """Stub vectors are scaled to [-1, 1] so cosine sim stays well-defined."""
    e = StubEmbedder()
    vec = e.embed("anything")
    assert max(abs(x) for x in vec) <= 1.0


# ============================================================================
# FeatherlessEmbedder — respx-mocked HTTP
# ============================================================================


def _ok_response(vec: list[float]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [{"embedding": vec, "index": 0}],
            "model": "BAAI/bge-small-en-v1.5",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )


def _build_mocked_client(response: httpx.Response) -> httpx.Client:
    """Return an httpx.Client whose POST /embeddings always yields `response`."""
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    transport = httpx.MockTransport(handler)
    return httpx.Client(
        base_url=FeatherlessEmbedder.BASE_URL,
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )


def test_featherless_returns_vector_on_200() -> None:
    expected = [0.01 * i for i in range(384)]
    client = _build_mocked_client(_ok_response(expected))
    emb = FeatherlessEmbedder(api_key="test-key", model="bge-small", http_client=client)
    vec = emb.embed("web_search")
    assert vec == expected
    assert len(vec) == 384


def test_featherless_rejects_empty_api_key() -> None:
    """An empty API key is a config bug; refuse to construct."""
    with pytest.raises(EmbeddingError, match="FEATHERLESS_API_KEY not set"):
        FeatherlessEmbedder(api_key="", model="bge-small")


def test_featherless_raises_on_dim_mismatch() -> None:
    """Wrong-dim payload almost certainly means the model field is misconfigured
    (e.g. the wrong size of bge model). Catch loud, not silent garbage."""
    short = [0.0] * 128  # not 384
    client = _build_mocked_client(_ok_response(short))
    emb = FeatherlessEmbedder(api_key="test-key", model="bge-small", http_client=client)
    with pytest.raises(EmbeddingError, match="dim mismatch"):
        emb.embed("anything")


def test_featherless_raises_on_malformed_payload() -> None:
    """Defensive parsing — Featherless returning an unexpected shape should
    yield a typed exception, not a KeyError that bubbles out of the cascade."""
    bad = httpx.Response(200, json={"unexpected": "shape"})
    client = _build_mocked_client(bad)
    emb = FeatherlessEmbedder(api_key="test-key", model="bge-small", http_client=client)
    with pytest.raises(EmbeddingError, match="malformed embedding payload"):
        emb.embed("anything")


def test_featherless_raises_on_5xx() -> None:
    """Featherless upstream errors should surface as EmbeddingError so the
    cascade orchestrator can decide between fallback vs degrading."""
    err = httpx.Response(503, json={"error": {"message": "service unavailable"}})
    client = _build_mocked_client(err)
    emb = FeatherlessEmbedder(api_key="test-key", model="bge-small", http_client=client)
    with pytest.raises(EmbeddingError, match="Featherless embed failed"):
        emb.embed("anything")


def test_featherless_satisfies_protocol() -> None:
    client = _build_mocked_client(_ok_response([0.0] * 384))
    emb = FeatherlessEmbedder(api_key="test-key", model="bge-small", http_client=client)
    assert isinstance(emb, Embedder)


# ============================================================================
# Cache behavior — second embed of identical text skips the network
# ============================================================================


def test_featherless_uses_cache_on_repeated_text(tmp_path) -> None:
    """If a (provider, model, text) tuple was already embedded, the second
    call must hit the cache and skip the network round-trip entirely.
    """
    from sentinel.embeddings import _DiskCache

    expected = [0.5 - 0.001 * i for i in range(384)]
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return _ok_response(expected)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        base_url=FeatherlessEmbedder.BASE_URL,
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )
    cache = _DiskCache(tmp_path / "embeds", size_limit_bytes=10 * 1024 * 1024)
    emb = FeatherlessEmbedder(
        api_key="test-key", model="bge-small", http_client=client, cache=cache
    )

    v1 = emb.embed("web_search")
    v2 = emb.embed("web_search")
    v3 = emb.embed("mcp__lint_check")  # different text — should re-hit network

    assert v1 == v2 == expected
    assert len(v3) == 384
    assert call_count["n"] == 2, f"expected 2 network calls, got {call_count['n']}"

    cache.close()


# ============================================================================
# get_embedder() factory — boot-safe fallback to stub when API key missing
# ============================================================================


def test_get_embedder_falls_back_to_stub_when_no_api_key(monkeypatch) -> None:
    """Constitution principle IV (Demo-First): the daemon must boot even with
    no API keys. get_embedder() returns a stub rather than crashing.
    """
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    monkeypatch.setenv("SENTINEL_EMBED_PROVIDER", "featherless")
    reset_embedder_cache()
    emb = get_embedder()
    # Either stub directly, or some embedder that satisfies the protocol.
    assert isinstance(emb, Embedder)
    # Specifically: with no key, the factory must NOT raise.
    vec = emb.embed("smoke")
    assert len(vec) > 0


def test_get_embedder_returns_semantic_stub_on_explicit_stub_provider(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_EMBED_PROVIDER", "stub")
    reset_embedder_cache()
    emb = get_embedder()
    assert isinstance(emb, SemanticStubEmbedder)


def test_get_embedder_returns_semantic_stub_on_unknown_provider(monkeypatch) -> None:
    """Unknown provider strings shouldn't crash; safe-fallback to SemanticStubEmbedder."""
    monkeypatch.setenv("SENTINEL_EMBED_PROVIDER", "nonexistent-cloud")
    reset_embedder_cache()
    emb = get_embedder()
    assert isinstance(emb, SemanticStubEmbedder)


def test_get_embedder_is_singleton(monkeypatch) -> None:
    """The factory caches the embedder for the process lifetime — avoids
    re-creating the HTTP client per request (which would defeat keep-alive).
    """
    monkeypatch.setenv("SENTINEL_EMBED_PROVIDER", "stub")
    reset_embedder_cache()
    a = get_embedder()
    b = get_embedder()
    assert a is b


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the embedder cache between tests so monkeypatched env vars
    don't leak across them."""
    reset_embedder_cache()
    yield
    reset_embedder_cache()
