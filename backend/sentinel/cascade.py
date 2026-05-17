"""3-layer cascade orchestrator — the public entry point for the detector.

This is the function that `app/main.py` POST `/detect` calls. It wires
Layer 1 → Layer 2 → Layer 3 according to the blueprint:

  L1 — Registry exact match. Hit -> ALLOW, done.
  L2 — Embedding similarity + F1/F2/F3 fusion. Miss -> always falls here.
  L3 — Gemini Flash semantic verifier. Fires ONLY when L2 confidence
       lands in the ambiguous range [block_max, auto_correct_min) — i.e.
       when L2's verdict is SUGGEST.

Confidence fusion when L3 fires (docs/blueprint.md §4):

    fused_confidence = 0.6 * L3.confidence + 0.4 * L2.confidence

Verdict re-mapping after fusion: if the fused confidence no longer
satisfies the threshold for L3's chosen verdict (e.g. AUTO_CORRECT but
fused drops below auto_correct_min), the verdict is DOWNGRADED one tier.

H1 registry validation: any suggestion produced by L3 must point at a
tool that ACTUALLY exists in the registry. A Gemini-fabricated or
prompt-injected `suggestion.tool_name` that's NOT in the registry would
defeat Sentinel's whole purpose — we'd hand the agent another phantom.
This module enforces that invariant before constructing the final
Decision.

Failure modes:
  - Embedding backend down  -> L2 returns BLOCK degraded=True.
  - Verifier returns None   -> fall back to L2 decision, set degraded=True.
  - Verifier suggests non-registered tool -> downgrade to BLOCK.
  - Verifier returns ALLOW  -> rejected at L3 (layer3.py guard); caller
                                sees None and degrades to L2.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from .config import load_cascade_config
from .embeddings import Embedder, EmbeddingError
from .layer1 import layer1
from .layer2 import _cosine, layer2, phantom_signature
from .layer3 import VerifierResponse, layer3
from .schemas import (
    Decision,
    DetectRequest,
    LayerBreakdown,
    Suggestion,
    Tool,
    ToolRegistry,
    Verdict,
)

log = structlog.get_logger("sentinel.cascade")


# ----------------------------------------------------------------------------
# Top candidates derivation — reused for Layer 3 escalation
# ----------------------------------------------------------------------------


def _top_candidates(
    req: DetectRequest,
    registry: ToolRegistry,
    embedder: Embedder,
    registry_embeddings: dict[str, list[float]],
    *,
    k: int = 3,
) -> list[tuple[float, Tool]]:
    """Top-k registered tools by cosine sim against the phantom.

    Re-embeds the phantom — but the disk-LRU cache absorbs the cost so
    this is microseconds on a cache hit, which is the steady state once
    we've seen the phantom once.
    """
    try:
        phantom_vec = embedder.embed(phantom_signature(req))
    except EmbeddingError:
        return []

    scored: list[tuple[float, Tool]] = []
    for tool in registry.tools:
        vec = registry_embeddings.get(tool.name)
        if vec is None:
            continue
        scored.append((_cosine(phantom_vec, vec), tool))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


# ----------------------------------------------------------------------------
# Verdict re-mapping helper
# ----------------------------------------------------------------------------


def _remap_verdict_after_fusion(verdict: Verdict, fused_conf: float) -> Verdict:
    """If fusion drops confidence below the verdict's required threshold,
    downgrade one tier so the Decision schema invariants hold.

    AUTO_CORRECT needs conf >= auto_correct_min; below that becomes SUGGEST.
    SUGGEST needs conf >= block_max; below that becomes BLOCK.
    """
    thresholds = load_cascade_config().verdict_thresholds
    if verdict == "AUTO_CORRECT" and fused_conf < thresholds.auto_correct_min:
        return "SUGGEST"
    if verdict == "SUGGEST" and fused_conf < thresholds.block_max:
        return "BLOCK"
    return verdict


# ----------------------------------------------------------------------------
# Final Decision constructor — applies all invariants
# ----------------------------------------------------------------------------


def _build_final_decision(
    *,
    req: DetectRequest,
    response: VerifierResponse,
    layer2_decision: Decision,
    layer3_ms: float,
    registry: ToolRegistry,
) -> Decision:
    """Fuse L2 + L3 and produce the final, schema-valid Decision.

    Applies in order:
      1. Confidence fusion (0.6 L3 + 0.4 L2).
      2. H1 registry-membership check on L3's suggestion.
      3. Verdict re-mapping after fusion.
      4. Schema-compatibility downgrades when invariants would fail
         (AUTO_CORRECT without suggestion, BLOCK with short reason).
    """
    breakdown = LayerBreakdown(
        l1_ms=layer2_decision.layer_breakdown.l1_ms,
        l2_ms=layer2_decision.layer_breakdown.l2_ms,
        l3_ms=layer3_ms,
    )

    # 1. Fuse confidences (blueprint §4)
    fused_conf = 0.6 * response.confidence + 0.4 * layer2_decision.confidence
    fused_conf = max(0.0, min(1.0, fused_conf))

    # 2. H1 — validate Layer 3's suggested tool is actually in the registry.
    #    A Gemini-fabricated or injected suggestion that's not registered
    #    would have us hand the agent another phantom. Hard rejection.
    suggested_tool_valid = True
    if response.suggestion is not None:
        if registry.find(response.suggestion.tool_name) is None:
            suggested_tool_valid = False
            log.warning(
                "layer3_suggested_phantom_tool",
                tool=req.tool_name,
                suggested=response.suggestion.tool_name,
                action="downgrade_to_block",
            )

    # Invalid suggestion -> immediate BLOCK; do not pass go.
    if not suggested_tool_valid:
        reason = (
            f"Tool '{req.tool_name}' not in registry; Layer 3 suggested "
            f"'{response.suggestion.tool_name}' which is also not registered. "
            "Revise plan with known tools."
        )[:240]
        return Decision(
            verdict="BLOCK",
            confidence=max(0.6, fused_conf),
            reason=reason,
            layer_breakdown=breakdown,
        )

    # 3. Verdict re-mapping after fusion
    verdict = _remap_verdict_after_fusion(response.verdict, fused_conf)

    # 4. Schema-compatibility downgrades
    suggestion: Suggestion | None = None
    if response.suggestion is not None:
        suggestion = Suggestion(
            tool_name=response.suggestion.tool_name,
            rationale=response.suggestion.rationale[:240],
        )

    # AUTO_CORRECT requires a non-None suggestion AND conf >= 0.85
    # (model_validator on Decision enforces both); guarantee both here.
    if verdict == "AUTO_CORRECT" and suggestion is None:
        verdict = "BLOCK"

    # BLOCK requires reason length >= 10
    reason = response.reason[:240]
    if verdict == "BLOCK" and len(reason) < 10:
        reason = f"Tool '{req.tool_name}' rejected by cascade after L3 verification."[:240]

    return Decision(
        verdict=verdict,
        confidence=fused_conf,
        reason=reason,
        suggestion=suggestion if verdict in ("AUTO_CORRECT", "SUGGEST") else None,
        layer_breakdown=breakdown,
        degraded=False,
    )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def detect(
    req: DetectRequest,
    registry: ToolRegistry,
    embedder: Embedder,
    registry_embeddings: dict[str, list[float]],
    verifier=None,  # Verifier | None — typed loosely so callers don't import the protocol
) -> Decision:
    """Run the full 3-layer cascade.

    Args:
        req: Hook-encoded detection request.
        registry: Active tool registry.
        embedder: Layer 2 embedding backend (typically a singleton from
                  `get_embedder()`).
        registry_embeddings: Warmed-up `{tool_name -> embedding}` produced
                             by `layer2.warm_up_registry()` at startup.
        verifier: Optional Layer 3 verifier (typically singleton from
                  `get_verifier()`). When None, Layer 3 is skipped entirely
                  even on ambiguous L2 verdicts.

    Returns:
        A schema-valid `Decision`. NEVER raises.
    """
    cascade_start = time.perf_counter()

    # Layer 1 — exact match short-circuit
    d1 = layer1(req.tool_name, registry)
    if d1 is not None:
        log.debug(
            "cascade_l1_allow",
            session_id=req.session_id,
            tool=req.tool_name,
            l1_ms=d1.layer_breakdown.l1_ms,
        )
        return d1

    # Layer 2 — embedding similarity + F1/F2/F3 fusion
    d2 = layer2(req, registry, embedder, registry_embeddings)

    # Decide whether to escalate to Layer 3.
    # Skip when:
    # - No verifier configured (e.g., GEMINI_API_KEY missing)
    # - L2 verdict is not SUGGEST (the ambiguous-window outcome).
    #   AUTO_CORRECT is already confident; BLOCK is already negative.
    # - L2 is already degraded (embedding backend failure) — Gemini can't
    #   recover without the candidate list.
    if verifier is None or d2.verdict != "SUGGEST" or d2.degraded:
        log.debug(
            "cascade_l2_terminal",
            session_id=req.session_id,
            tool=req.tool_name,
            verdict=d2.verdict,
            confidence=round(d2.confidence, 3),
            degraded=d2.degraded,
            total_ms=round((time.perf_counter() - cascade_start) * 1000.0, 2),
        )
        return d2

    # Re-derive top candidates for Layer 3 (cache-hit thanks to embedder LRU)
    candidates = _top_candidates(req, registry, embedder, registry_embeddings)
    if not candidates:
        # Embedding backend dropped between L2 and now — degrade gracefully.
        return d2.model_copy(update={"degraded": True})

    # Layer 3 — Gemini Flash verifier
    response, l3_ms = layer3(req, registry, candidates, verifier)

    if response is None:
        # Verifier failed: timeout, malformed JSON, schema violation, or
        # rejected-ALLOW guard fired. Return L2 with degraded flag + L3 ms
        # populated so audit logs reflect that we DID attempt verification.
        log.warning(
            "cascade_l3_failed_falling_back_to_l2",
            session_id=req.session_id,
            tool=req.tool_name,
            l3_ms=round(l3_ms, 2),
        )
        return d2.model_copy(
            update={
                "degraded": True,
                "layer_breakdown": LayerBreakdown(
                    l1_ms=d2.layer_breakdown.l1_ms,
                    l2_ms=d2.layer_breakdown.l2_ms,
                    l3_ms=l3_ms,
                ),
            }
        )

    # Fuse L2 + L3 into the final Decision
    final = _build_final_decision(
        req=req,
        response=response,
        layer2_decision=d2,
        layer3_ms=l3_ms,
        registry=registry,
    )

    log.info(
        "cascade_l3_resolved",
        session_id=req.session_id,
        tool=req.tool_name,
        l2_verdict=d2.verdict,
        l3_verdict=response.verdict,
        final_verdict=final.verdict,
        l2_conf=round(d2.confidence, 3),
        l3_conf=round(response.confidence, 3),
        final_conf=round(final.confidence, 3),
        total_ms=round((time.perf_counter() - cascade_start) * 1000.0, 2),
    )
    return final
