"""Sentinel FastAPI daemon — Day 3 wiring.

Day 1 shipped a fully mocked `/detect` to clear the deploy pipeline.
Day 2 wired live Layer 1 (registry exact-match) + structured audit log.
Day 3 wires Layer 2: warm every registered tool's embedding at startup,
and on a Layer 1 miss run the embedding similarity + F1/F2/F3 fusion
cascade to produce a real AUTO_CORRECT / SUGGEST / BLOCK decision.

Layer 3 (Gemini Flash verifier) escalation lands with T030/T032.

Run locally: `uvicorn backend.app.main:app --host 0.0.0.0 --port 7777`
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from sentinel import (
    Decision,
    DetectRequest,
    Embedder,
    ToolRegistry,
    get_embedder,
    layer1,
    layer2,
    load_registry,
    warm_up_registry,
)
from sentinel.config import load_cascade_config


# ---------- Structured logging setup ----------

logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger("sentinel.daemon")


# ---------- App + state ----------

app = FastAPI(
    title="Sentinel",
    version="0.3.0-day3",
    description="Phantom tool-call detector with 3-layer cascade. Day 3 wiring (Layer 1 + Layer 2 live).",
)


class _State:
    """Process-local daemon state: registry, warm embeddings, embedder
    instance, and SSE subscribers. Initialised on the FastAPI startup hook."""

    def __init__(self) -> None:
        self.registry: ToolRegistry = ToolRegistry()
        self.embedder: Embedder | None = None
        self.registry_embeddings: dict[str, list[float]] = {}
        self.event_subscribers: list[asyncio.Queue[Decision]] = []


_state = _State()


@app.on_event("startup")  # type: ignore[deprecated]
async def _on_startup() -> None:
    """Load registry, build embedder, warm up registry embeddings."""
    _state.registry = load_registry()
    _state.embedder = get_embedder()

    warm_start = time.perf_counter()
    _state.registry_embeddings = warm_up_registry(_state.registry, _state.embedder)
    warm_elapsed_ms = (time.perf_counter() - warm_start) * 1000.0

    cfg = load_cascade_config()
    log.info(
        "daemon_startup",
        version=app.version,
        registry_version=_state.registry.version,
        registry_size=len(_state.registry.tools),
        registered_names=list(_state.registry.names),
        warm_embeddings=len(_state.registry_embeddings),
        warmup_ms=round(warm_elapsed_ms, 2),
        embedding_provider=cfg.embedding.provider,
        embedding_model=cfg.embedding.featherless_model,
        auto_correct_min=cfg.verdict_thresholds.auto_correct_min,
        block_max=cfg.verdict_thresholds.block_max,
    )


# ---------- Endpoints ----------


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": app.version,
        "registry_loaded": len(_state.registry.tools) > 0,
        "registry_size": len(_state.registry.tools),
        "registry_version": _state.registry.version,
        "warm_embeddings": len(_state.registry_embeddings),
        "embedder": type(_state.embedder).__name__ if _state.embedder else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/detect", response_model=Decision)
async def detect(req: DetectRequest) -> Decision:
    """Live Layer 1 + Layer 2 cascade.

    Flow:
      1. Resolve registry: req.registry if provided, else daemon default.
      2. Resolve embeddings: precomputed for daemon's registry; warm on
         the fly for per-call registry overrides (Day-5 multi-agent path).
      3. Layer 1 — exact match -> ALLOW (early return).
      4. Layer 2 — cosine + fusion -> AUTO_CORRECT | SUGGEST | BLOCK.
      5. Broadcast Decision to SSE subscribers.
      6. structlog audit line.
    """
    request_start = time.perf_counter()
    assert _state.embedder is not None, "embedder must be initialised at startup"

    # Resolve registry + warm embeddings for this request
    if req.registry is not None:
        # Per-call registry override (Day-5 multi-agent path). Warm on the fly;
        # the embedder's disk LRU absorbs the cost on repeat overrides.
        registry = req.registry
        embeddings = warm_up_registry(registry, _state.embedder)
    else:
        registry = _state.registry
        embeddings = _state.registry_embeddings

    # Layer 1 — exact match short-circuit
    decision = layer1(req.tool_name, registry)
    if decision is not None:
        log.debug(
            "tool_allowed",
            session_id=req.session_id,
            tool=req.tool_name,
            l1_ms=decision.layer_breakdown.l1_ms,
        )
        _broadcast(decision)
        return decision

    # Layer 2 — embedding + fusion. NEVER returns ALLOW; always one of
    # AUTO_CORRECT / SUGGEST / BLOCK.
    decision = layer2(req, registry, _state.embedder, embeddings)

    total_elapsed_ms = (time.perf_counter() - request_start) * 1000.0
    log.warning(
        "phantom_intercepted",
        session_id=req.session_id,
        tool=req.tool_name,
        registered_count=len(registry.tools),
        verdict=decision.verdict,
        confidence=round(decision.confidence, 3),
        suggested=decision.suggestion.tool_name if decision.suggestion else None,
        agent_content_present=req.agent_content is not None,
        degraded=decision.degraded,
        l2_ms=round(decision.layer_breakdown.l2_ms, 3),
        total_ms=round(total_elapsed_ms, 3),
    )

    _broadcast(decision)
    return decision


def _broadcast(decision: Decision) -> None:
    """Fan-out a Decision to all SSE subscribers; never blocks /detect."""
    for q in _state.event_subscribers:
        try:
            q.put_nowait(decision)
        except asyncio.QueueFull:
            pass  # slow subscriber: drop, don't block the cascade


@app.get("/events")
async def events() -> StreamingResponse:
    """Server-Sent Events stream of every Decision. Dashboard subscribes here."""
    queue: asyncio.Queue[Decision] = asyncio.Queue(maxsize=128)
    _state.event_subscribers.append(queue)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            yield ": sentinel stream open\n\n"
            while True:
                decision = await queue.get()
                yield f"data: {decision.model_dump_json()}\n\n"
        finally:
            try:
                _state.event_subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SENTINEL_PORT", "7777"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
