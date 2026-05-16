"""Layer 2 — embedding similarity + F1/F2/F3 fusion.

When Layer 1's exact-match check misses, the cascade enters Layer 2:

  1. Embed the phantom call (tool_name + any arg keys agent supplied).
  2. Cosine-similarity it against every registered tool's precomputed
     embedding (built once at startup via `warm_up_registry`).
  3. Top-3 candidates. Rescale top-1 cosine to base confidence in [0, 1].
  4. If base lands in the ambiguous window [block_max, auto_correct_min):
     blend with F1 (Levenshtein name distance), F2 (Jaccard arg-keys
     overlap), F3 (top-1 vs top-2 gap) via `heuristics.fuse`.
  5. Map final confidence to verdict per the cascade thresholds:
       >= auto_correct_min  ->  AUTO_CORRECT (with top-1 as suggestion)
       in [block_max, ac)   ->  SUGGEST       (top-3 surfaced)
       <  block_max         ->  BLOCK         (no plausible candidate)

Constitution II: Layer 2 must clear <10ms median on commodity Linux. The
cosine pass is `O(N * dim)` numpy dots over the registry, which for
N=500 / dim=384 is ~250µs on a 2 vCPU box. The expensive piece is the
phantom's own embedding lookup — that's an HTTP roundtrip on cache miss
(amortized away once the cache is warm for repeat phantoms).

This module is also where `warm_up_registry()` lives — it's tightly
coupled to the cosine pass (same data structure shape) and only used at
daemon boot.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import structlog

from .config import load_cascade_config
from .embeddings import Embedder, EmbeddingError
from .heuristics import f1_levenshtein, f2_jaccard, f3_gap, fuse
from .schemas import (
    Decision,
    DetectRequest,
    LayerBreakdown,
    Suggestion,
    Tool,
    ToolRegistry,
)

log = structlog.get_logger("sentinel.layer2")


# ----------------------------------------------------------------------------
# Signature builders — keep the strings we embed deterministic + auditable
# ----------------------------------------------------------------------------


def tool_signature(tool: Tool) -> str:
    """Canonical embedding-input string for a registered tool.

    Combines name + description + required arg keys. Both the registry
    warm-up pass AND any per-call recomputation MUST use this exact
    function so embeddings stay key-stable across daemon restarts (and
    the on-disk LRU cache stays valid).
    """
    parts: list[str] = [tool.name]
    if tool.description:
        parts.append(tool.description)
    if tool.required_args:
        parts.append("args:" + ",".join(tool.required_args))
    return " ".join(parts)


def phantom_signature(req: DetectRequest) -> str:
    """Canonical embedding-input string for an incoming phantom call.

    Agent doesn't supply a description for a tool it just invented, so
    we use the tool_name plus whatever arg keys we happen to see. The
    arg keys ride into Layer 2 the same way they ride into F2 — they're
    a strong intent signal when the model gets the *shape* right and
    only the *name* wrong.
    """
    parts: list[str] = [req.tool_name]
    if req.tool_input:
        keys = sorted(str(k) for k in req.tool_input.keys())
        parts.append("args:" + ",".join(keys))
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Warm-up — daemon startup pass that embeds every registry tool
# ----------------------------------------------------------------------------


def warm_up_registry(
    registry: ToolRegistry, embedder: Embedder
) -> dict[str, list[float]]:
    """Embed every registered tool's signature; return a name -> vector map.

    Called exactly once at daemon startup. With the on-disk cache warm
    (second deploy onwards), this is microseconds. First deploy pays
    one API roundtrip per tool — for a 20-tool registry that's typically
    <1 second wall-clock.

    Embeddings of tools whose `tool.embed()` raises EmbeddingError are
    silently dropped from the returned map (logged as a warning). At
    detection time, Layer 2 skips registry entries with no embedding
    rather than crashing the request — graceful degradation.
    """
    start = time.perf_counter()
    out: dict[str, list[float]] = {}
    failures = 0
    for tool in registry.tools:
        try:
            sig = tool_signature(tool)
            out[tool.name] = embedder.embed(sig)
        except EmbeddingError as e:
            failures += 1
            log.warning("warmup_embed_failed", tool=tool.name, error=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log.info(
        "registry_warmed_up",
        embedded=len(out),
        total=len(registry.tools),
        failures=failures,
        elapsed_ms=round(elapsed_ms, 2),
    )
    return out


# ----------------------------------------------------------------------------
# Cosine similarity — numpy fast path
# ----------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 when either vector is the zero vector (rather than NaN)
    so downstream confidence math stays well-defined. The caller (Layer 2)
    relies on this never returning NaN — a NaN here would silently
    propagate into Decision.confidence and fail the schema's [0, 1] bound.
    """
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(arr_a))
    nb = float(np.linalg.norm(arr_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (na * nb))


def _rescale_top1_to_confidence(top1_sim: float) -> float:
    """Stretch raw cosine (typically 0.4-0.95 for natural-language pairs)
    into the [0, 1] confidence space the cascade thresholds expect.

    Formula: (sim - 0.5) * 2, clipped to [0, 1]. So:
      sim = 0.5   ->  conf 0.0
      sim = 0.75  ->  conf 0.5
      sim = 0.85  ->  conf 0.7
      sim = 0.925 ->  conf 0.85  (right at the AUTO_CORRECT threshold)
      sim = 0.95  ->  conf 0.9
    """
    return max(0.0, min(1.0, (top1_sim - 0.5) * 2.0))


# ----------------------------------------------------------------------------
# Layer 2 orchestrator
# ----------------------------------------------------------------------------


def layer2(
    req: DetectRequest,
    registry: ToolRegistry,
    embedder: Embedder,
    registry_embeddings: dict[str, list[float]],
) -> Decision:
    """Run the Layer 2 cascade on a phantom that Layer 1 missed.

    Args:
        req: Incoming detection request.
        registry: Active tool registry.
        embedder: Backend that produces the phantom's embedding. The
                  registry's tool embeddings were precomputed via
                  `warm_up_registry`; only the phantom is embedded
                  per-request.
        registry_embeddings: Output of `warm_up_registry(registry, embedder)`.
                             Keyed by `Tool.name`.

    Returns:
        A `Decision` with `verdict` in {AUTO_CORRECT, SUGGEST, BLOCK}
        and a populated `layer_breakdown.l2_ms`. NEVER returns ALLOW —
        that's exclusively Layer 1's outcome.
    """
    start = time.perf_counter()
    cfg = load_cascade_config()
    thresholds = cfg.verdict_thresholds

    # Empty registry — config issue, not runtime degradation. The daemon
    # was started without any tools to compare against; the caller should
    # fix configs/registry.yaml, not retry.
    if not registry.tools:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Decision(
            verdict="BLOCK",
            confidence=1.0,
            reason=(
                f"Tool '{req.tool_name}' not registered and registry is empty. "
                "Configure tools in registry.yaml or restart daemon."
            ),
            layer_breakdown=LayerBreakdown(l2_ms=elapsed_ms),
        )

    # Registry has tools but NO warm embeddings — runtime degradation
    # (warm_up_registry failed for all entries, typically due to a global
    # embedding-backend outage at startup). Distinct from the empty-registry
    # case: this IS recoverable by restarting once the backend is back up,
    # so flag degraded=True so dashboards / oncall can tell the difference.
    if not registry_embeddings:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Decision(
            verdict="BLOCK",
            confidence=0.8,
            reason=(
                f"Tool '{req.tool_name}' not registered and no registry tools "
                "have warm embeddings. Restart daemon to retry warm-up."
            ),
            layer_breakdown=LayerBreakdown(l2_ms=elapsed_ms),
            degraded=True,
        )

    # Embed the phantom. On backend failure we degrade gracefully (BLOCK with
    # degraded=True flag) rather than crashing the cascade — the upstream
    # hook should still get a parseable Decision so the agent can revise.
    try:
        phantom_vec = embedder.embed(phantom_signature(req))
    except EmbeddingError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.warning("layer2_embed_failed", tool=req.tool_name, error=str(e))
        return Decision(
            verdict="BLOCK",
            confidence=0.5,
            reason=(
                f"Tool '{req.tool_name}' not registered; embedding backend "
                "unavailable for similarity search. Try again or revise plan."
            ),
            layer_breakdown=LayerBreakdown(l2_ms=elapsed_ms),
            degraded=True,
        )

    # Score every registered tool. Skip tools whose warmup-pass embed failed.
    scored: list[tuple[float, Tool]] = []
    for tool in registry.tools:
        vec = registry_embeddings.get(tool.name)
        if vec is None:
            continue
        scored.append((_cosine(phantom_vec, vec), tool))

    if not scored:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Decision(
            verdict="BLOCK",
            confidence=0.8,
            reason=(
                f"Tool '{req.tool_name}' not registered and no registry tools "
                "have warm embeddings. Restart daemon to retry warm-up."
            ),
            layer_breakdown=LayerBreakdown(l2_ms=elapsed_ms),
            degraded=True,
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    top3 = scored[:3]
    top1_sim, top1_tool = top3[0]
    top3_sims = [s for s, _ in top3]

    # Compute fusion features (cheap — Levenshtein + Jaccard + scalar).
    feat_f1 = f1_levenshtein(req.tool_name, top1_tool.name)
    feat_f2 = f2_jaccard(set(req.tool_input.keys()), set(top1_tool.required_args))
    feat_f3 = f3_gap(top3_sims, multiplier=cfg.fusion.f3_multiplier)

    base_conf = _rescale_top1_to_confidence(top1_sim)

    # Fusion gate — only blend in the ambiguous window. Outside that range
    # the heuristics are noise; the cosine alone is more trustworthy.
    if thresholds.block_max <= base_conf < thresholds.auto_correct_min:
        final_conf = fuse(
            base_conf, feat_f1, feat_f2, feat_f3, weights=cfg.fusion.as_tuple
        )
    else:
        final_conf = base_conf

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    breakdown = LayerBreakdown(l2_ms=elapsed_ms)

    # Verdict mapping — short reasons since stderr line is capped at 240 chars
    # by the Decision schema.
    if final_conf >= thresholds.auto_correct_min:
        reason = (
            f"Tool '{req.tool_name}' not in registry. Use '{top1_tool.name}' "
            f"(sim {top1_sim:.2f}). Retry."
        )
        return Decision(
            verdict="AUTO_CORRECT",
            confidence=final_conf,
            reason=reason[:240],
            suggestion=Suggestion(
                tool_name=top1_tool.name,
                rationale=(
                    f"sim={top1_sim:.2f} F1={feat_f1:.2f} "
                    f"F2={feat_f2:.2f} F3={feat_f3:.2f}"
                )[:240],
            ),
            layer_breakdown=breakdown,
        )

    if final_conf >= thresholds.block_max:
        candidate_list = ", ".join(t.name for _, t in top3)
        reason = (
            f"Tool '{req.tool_name}' ambiguous. Candidates: {candidate_list}. "
            "Pick or revise."
        )
        return Decision(
            verdict="SUGGEST",
            confidence=final_conf,
            reason=reason[:240],
            suggestion=Suggestion(
                tool_name=top1_tool.name,
                rationale=f"top-1 of {len(top3)} (sim {top1_sim:.2f})"[:240],
            ),
            layer_breakdown=breakdown,
        )

    # BLOCK — best candidate isn't plausible. Confidence here means
    # "confidence that no good replacement exists" = 1 - top1_match_strength.
    block_conf = max(0.6, 1.0 - top1_sim)
    return Decision(
        verdict="BLOCK",
        confidence=block_conf,
        reason=(
            f"Tool '{req.tool_name}' not in registry; closest "
            f"'{top1_tool.name}' at sim {top1_sim:.2f}. Revise plan."
        )[:240],
        layer_breakdown=breakdown,
    )
