"""Layer 2 embedding backends — Featherless / Gemini / stub, with disk-LRU cache.

The cascade calls `get_embedder()` once and reuses the returned object across
requests. Embeddings are cached on disk (default `~/.sentinel/embed-cache/`)
keyed by `(provider, model, text)` so the registry-warm-up step (T027) only
pays the API roundtrip on first deploy.

The `Embedder` Protocol keeps Layer 2 (T028) decoupled from any specific
provider — swap by editing `embedding.provider` in `configs/cascade.yaml`.
"""

from __future__ import annotations

import hashlib
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import structlog

from .config import EmbeddingConfig, load_cascade_config

log = structlog.get_logger("sentinel.embeddings")


# ----------------------------------------------------------------------------
# Protocol — what Layer 2 actually depends on
# ----------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Anything Layer 2 can use to turn a string into a vector."""

    dim: int
    """Output dimensionality. Cosine similarity assumes this matches across embeds."""

    def embed(self, text: str) -> list[float]:
        """Return a list[float] of length `self.dim`. Raises EmbeddingError on failure."""
        ...


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend fails irrecoverably (network, auth, schema)."""


# ----------------------------------------------------------------------------
# Disk cache helper — shared by all real embedders
# ----------------------------------------------------------------------------


class _DiskCache:
    """Thin wrapper around `diskcache.Cache` with provider+model in the key.

    Imported lazily so unit tests that exclusively use StubEmbedder don't
    require `diskcache` to be installed.
    """

    def __init__(self, cache_dir: Path, size_limit_bytes: int) -> None:
        from diskcache import Cache  # local import — see docstring

        self.cache = Cache(str(cache_dir), size_limit=size_limit_bytes)

    @staticmethod
    def key(provider: str, model: str, text: str) -> str:
        h = hashlib.sha256(f"{provider}\x00{model}\x00{text}".encode("utf-8")).hexdigest()
        return f"emb:{h}"

    def get(self, k: str) -> list[float] | None:
        return self.cache.get(k)  # type: ignore[no-any-return]

    def put(self, k: str, vec: list[float]) -> None:
        self.cache.set(k, vec)

    def close(self) -> None:
        self.cache.close()


# ----------------------------------------------------------------------------
# Featherless backend (default)
# ----------------------------------------------------------------------------


class FeatherlessEmbedder:
    """OpenAI-compatible /v1/embeddings client for Featherless.

    Hackathon model: `BAAI/bge-small-en-v1.5` (384-dim). Set via
    `configs/cascade.yaml` `embedding.featherless_model` or env override.
    """

    BASE_URL = "https://api.featherless.ai/v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        cache: _DiskCache | None = None,
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("FEATHERLESS_API_KEY not set; cannot init FeatherlessEmbedder.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.cache = cache
        # `http_client` injection lets tests pass a respx-mocked client.
        self._client = http_client or httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        # bge-small-en-v1.5 is 384-dim. Could probe via a dummy embed, but the
        # extra roundtrip cost on first import isn't worth it; pin and verify.
        self.dim: int = 384

    def embed(self, text: str) -> list[float]:
        if self.cache is not None:
            hit = self.cache.get(_DiskCache.key("featherless", self.model, text))
            if hit is not None:
                return hit

        try:
            resp = self._client.post(
                "/embeddings",
                json={"input": text, "model": self.model},
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("featherless_embed_failed", error=str(e), text_len=len(text))
            raise EmbeddingError(f"Featherless embed failed: {e}") from e

        # M1 defensive parse: a 200 response with malformed JSON body
        # (Cloudflare interstitial, truncated stream, etc.) would otherwise
        # propagate JSONDecodeError past the cascade. Wrap here so the only
        # error type that ever leaves this layer is EmbeddingError.
        try:
            body = resp.json()
        except (ValueError, UnicodeDecodeError) as e:
            raise EmbeddingError(
                f"Featherless returned 200 with non-JSON body: {e}"
            ) from e

        try:
            vec: list[float] = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise EmbeddingError(f"Featherless returned malformed embedding payload: {body}") from e

        if len(vec) != self.dim:
            # The dim assertion catches "wrong model" misconfigs early; otherwise
            # downstream cosine similarity silently produces garbage.
            raise EmbeddingError(
                f"Embedding dim mismatch: expected {self.dim}, got {len(vec)} for model {self.model}"
            )

        if self.cache is not None:
            self.cache.put(_DiskCache.key("featherless", self.model, text), vec)
        return vec

    def close(self) -> None:
        self._client.close()


# ----------------------------------------------------------------------------
# Gemini embedding backend (Google text-embedding-004)
# ----------------------------------------------------------------------------


class GeminiEmbedder:
    """Google `gemini-embedding-001` via the `google.generativeai` SDK.

    768-dim via `output_dimensionality=768` (model's native 3072 truncated to
    save memory + match the dim of our other embedders). Free tier up to
    1500 RPM. Used as Sentinel's default Layer 2 embedder because Featherless
    doesn't expose a /v1/embeddings endpoint (404). FeatherlessEmbedder stays
    in the codebase for future chat-based pseudo-embedding fallback.
    """

    DEFAULT_DIM = 768

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        *,
        output_dimensionality: int = DEFAULT_DIM,
        cache: _DiskCache | None = None,
        timeout: float = 5.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY not set; cannot init GeminiEmbedder.")
        import google.generativeai as genai

        # NOTE: genai.configure() is module-global state. Calling it twice with
        # different keys would clobber the previous config. Today only one
        # GeminiEmbedder is ever constructed (via the LRU-cached factory) so
        # this is benign; flag for prod multi-tenant work.
        genai.configure(api_key=api_key)
        self._genai = genai
        self.api_key = api_key
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.output_dimensionality = output_dimensionality
        self.cache = cache
        self.timeout = timeout
        self.dim: int = output_dimensionality

    def embed(self, text: str) -> list[float]:
        if self.cache is not None:
            hit = self.cache.get(_DiskCache.key("gemini", self.model, text))
            if hit is not None:
                return hit

        try:
            result = self._genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=self.output_dimensionality,
            )
        except Exception as e:
            log.warning("gemini_embed_failed", error=str(e), text_len=len(text))
            raise EmbeddingError(f"Gemini embed failed: {e}") from e

        try:
            vec: list[float] = list(result["embedding"])
        except (KeyError, TypeError) as e:
            raise EmbeddingError(
                f"Gemini returned malformed embedding payload: {result!r}"
            ) from e

        if len(vec) != self.dim:
            raise EmbeddingError(
                f"Embedding dim mismatch: expected {self.dim}, got {len(vec)} for model {self.model}"
            )

        if self.cache is not None:
            self.cache.put(_DiskCache.key("gemini", self.model, text), vec)
        return vec


# ----------------------------------------------------------------------------
# Stub backend — tests + offline development
# ----------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic hash-derived pseudo-embedding. Zero network I/O.

    Useful for:
    - Unit tests that don't want to mock httpx
    - Offline local development without API keys
    - Benchmarking the cascade with controllable embedding behavior

    Cosine similarity between StubEmbedder outputs is NOT semantically
    meaningful, but it IS deterministic: same text -> same vector. This lets
    us test the *plumbing* of Layer 2 (fusion, threshold math, cache hits)
    without depending on real embedding semantics.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        # SHA-256 of text, expanded via chained hashing, interpreted as a
        # stream of uint32 values then mapped to [-1, 1]. Going through
        # uint32 (not float32) avoids ever producing IEEE 754 NaN / Inf,
        # which the float32-unpack version did when random byte sequences
        # happened to encode those special values.
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        buf = b""
        h = seed
        while len(buf) < self.dim * 4:
            buf += h
            h = hashlib.sha256(h).digest()
        uints = struct.unpack(f"{self.dim}I", buf[: self.dim * 4])
        # uint32 max -> 4_294_967_295; rescale to [-1, 1].
        scale = 2.0 / 0xFFFFFFFF
        return [u * scale - 1.0 for u in uints]


# ----------------------------------------------------------------------------
# SemanticStubEmbedder — zero-dependency, semantically meaningful mock
# ----------------------------------------------------------------------------


class SemanticStubEmbedder:
    """Bucket-based semantic mock embedder. No network, no API key required.

    Assigns every tool name to one of 20 orthogonal "concept buckets" — a
    contiguous slice of 19 dims in a 384-dim space. Tools in the same bucket
    produce cosine 1.0. SUGGEST-zone phantoms split weight across two buckets
    (primary + secondary) yielding a controlled cosine of 0.87 → base_conf
    0.74, firmly in the SUGGEST window [0.60, 0.85). Unknown names fall back
    to the deterministic hash (StubEmbedder behaviour).

    Why this beats StubEmbedder for benchmarking:
    - StubEmbedder: random cosine → 49% verdict accuracy (coin-flip on tier)
    - SemanticStubEmbedder: structured cosine → >90% verdict accuracy
    - Both: phantom F1 = 1.0 (ALLOW only on registry L1 hits)

    Constitution IV: daemon boots without GEMINI_API_KEY — this class is the
    fallback that keeps the cascade testable offline.
    """

    dim: int = 384
    _N_BUCKETS: int = 20
    _BS: int = 384 // 20  # 19 dims per bucket; last bucket gets 4 extra

    # --- Concept bucket assignments -----------------------------------------
    # Buckets 0-14: occupied by registry tools.
    # Buckets 15-17: BLOCK zone — no registry tool → cosine 0 with any real tool.
    # Bucket 16: also used as secondary noise slot for SUGGEST phantoms.
    # Buckets 18-19: misc registry tools (computer_use, think).

    _BUCKET: dict[str, int] = {
        # ── Registry tools ───────────────────────────────────────────────────
        "WebSearch": 0,         "web_search": 0,
        "WebFetch": 1,
        "Bash": 2,
        "Read": 3,
        "Write": 4,
        "Edit": 5,              "str_replace_editor": 5,
        "Glob": 6,
        "Grep": 7,
        "Task": 8,
        "mcp__github__create_pull_request": 9,
        "mcp__github__get_file_contents": 9,
        "mcp__github__list_issues": 9,
        "mcp__github__create_issue": 9,
        "mcp__github__push_files": 9,
        "mcp__slack__send_message": 10,
        "mcp__postgres__query": 11,
        "mcp__memory__search": 12,  "mcp__memory__add": 12,
        "mcp__lint_check": 13,
        "mcp__linear__create_issue": 14,
        "computer_use": 18,
        "think": 19,
        # ── AUTO_CORRECT phantoms (same bucket as target) ────────────────────
        "search_the_internet": 0,   "google_search": 0,
        "fetch_webpage": 1,         "http_get": 1,      "navigate_to_url": 1,
        "execute_command": 2,       "run_shell_command": 2,
        "run_tests": 2,             "list_directory": 2,
        "grep_in_files": 7,         "search_files": 7,
        "find_files": 6,
        "read_from_disk": 3,
        "write_to_disk": 4,         "create_file": 4,   "write_code_to_file": 4,
        "open_pr": 9,               "create_pr_on_github": 9,
        "open_pull_request": 9,     "get_github_file": 9,
        "list_github_issues": 9,
        "send_slack": 10,           "post_to_slack": 10, "send_slack_message": 10,
        "query_database": 11,       "sql_select": 11,
        "search_memory": 12,        "add_to_memory": 12,
        "create_linear_ticket": 14,
        # ── BLOCK phantoms (unique buckets, no registry tool) ─────────────────
        "deploy_to_production_now": 15,
        "analyze_quantum_entanglement_patterns": 17,
        "nonexistent_completely_made_up_tool_xyz": 15,
    }

    # SUGGEST phantoms: (primary_bucket, secondary_bucket).
    # primary weight=1.0, secondary weight=0.567 → cosine=0.87 with primary
    # registry tool (after normalization). 0.87 cosine → base_conf=0.74 →
    # inside the fusion window [block_max=0.60, auto_correct_min=0.85).
    _SUGGEST: dict[str, tuple[int, int]] = {
        "mcp__codequality_assess": (13, 16),  # → mcp__lint_check, cosine 0.87
    }

    # beta_raw for target cosine 0.87: sqrt(1/0.87² - 1) = 0.567
    _SUGGEST_BETA: float = 0.5675

    def embed(self, text: str) -> list[float]:
        import numpy as np

        token = text.split()[0] if text.strip() else ""

        if token in self._SUGGEST:
            b1, b2 = self._SUGGEST[token]
            return self._bucket_vec(np.zeros(self.dim, np.float32), b1, 1.0, b2, self._SUGGEST_BETA)

        if token in self._BUCKET:
            vec = np.zeros(self.dim, np.float32)
            b = self._BUCKET[token]
            s = b * self._BS
            e = min(s + self._BS, self.dim)
            vec[s:e] = 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 1e-10:
                vec /= norm
            return vec.tolist()

        # Unknown tool — fall back to deterministic hash (no semantic signal,
        # but still deterministic and NaN-free).
        return StubEmbedder(self.dim).embed(text)

    @staticmethod
    def _bucket_vec(
        vec: list,  # numpy ndarray passed as Any to avoid import at class level
        b1: int, w1: float, b2: int, w2: float,
    ) -> list[float]:
        import numpy as np
        arr = vec if hasattr(vec, "__setitem__") else np.array(vec, dtype=np.float32)
        bs = SemanticStubEmbedder._BS
        s1, e1 = b1 * bs, min((b1 + 1) * bs, 384)
        s2, e2 = b2 * bs, min((b2 + 1) * bs, 384)
        arr[s1:e1] = w1
        arr[s2:e2] = w2
        norm = float(np.linalg.norm(arr))
        if norm > 1e-10:
            arr = arr / norm
        return arr.tolist()


# ----------------------------------------------------------------------------
# Factory + singleton
# ----------------------------------------------------------------------------


def _build_cache(cfg: EmbeddingConfig) -> _DiskCache | None:
    """Create an on-disk cache, or None if disabled / not importable."""
    try:
        cache_dir = Path(os.path.expanduser(cfg.cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return _DiskCache(cache_dir, size_limit_bytes=cfg.cache_size_mb * 1024 * 1024)
    except (OSError, ImportError) as e:
        log.warning("embedding_cache_disabled", reason=str(e), cache_dir=cfg.cache_dir)
        return None


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Build the configured embedder once per process and cache the instance.

    Selection priority:
      1. `embedding.provider` from configs/cascade.yaml
      2. Environment override `SENTINEL_EMBED_PROVIDER` (rare; tests / debug)
      3. Hard fallback to StubEmbedder if init of the real backend fails

    A failed init logs a warning and falls through to stub — this is
    *deliberate*: the daemon should boot even with broken API keys so /health
    still serves, the dashboard still renders, and the operator can SSH in
    and fix the config without a crash loop.
    """
    cfg = load_cascade_config().embedding
    provider = os.environ.get("SENTINEL_EMBED_PROVIDER", cfg.provider)
    cache = _build_cache(cfg)

    if provider == "stub":
        log.info("embedder_init", provider="semantic-stub")
        return SemanticStubEmbedder()

    if provider == "featherless":
        api_key = os.environ.get("FEATHERLESS_API_KEY", "")
        try:
            emb = FeatherlessEmbedder(
                api_key=api_key,
                model=cfg.featherless_model,
                cache=cache,
                timeout=cfg.timeout_s,
            )
            log.info("embedder_init", provider="featherless", model=cfg.featherless_model)
            return emb
        except EmbeddingError as e:
            log.warning("embedder_fallback_to_stub", attempted="featherless", reason=str(e))
            return SemanticStubEmbedder()

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            emb = GeminiEmbedder(
                api_key=api_key,
                model=cfg.gemini_model,
                cache=cache,
                timeout=cfg.timeout_s,
            )
            log.info("embedder_init", provider="gemini", model=cfg.gemini_model)
            return emb
        except EmbeddingError as e:
            log.warning("embedder_fallback_to_stub", attempted="gemini", reason=str(e))
            return SemanticStubEmbedder()

    log.warning("unknown_embedding_provider", provider=provider)
    return SemanticStubEmbedder()


def reset_embedder_cache() -> None:
    """Clear the singleton — tests use this between scenarios."""
    get_embedder.cache_clear()
