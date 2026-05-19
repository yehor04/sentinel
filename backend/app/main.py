"""Sentinel FastAPI daemon — Day 3 final wiring (cascade live).

Day 1 mocked /detect, Day 2 wired live Layer 1, Day 3 (this revision)
wires the full 3-layer cascade through `cascade.detect()`. The daemon
loads registry + embedder + verifier at startup, warms registry
embeddings once, then every /detect call runs the orchestrator that
strings L1 → L2 → (optional L3) and produces a schema-valid Decision.

Run locally: `uvicorn backend.app.main:app --host 0.0.0.0 --port 7777`
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from sentinel import (
    Decision,
    DetectRequest,
    Embedder,
    ToolRegistry,
    Verifier,
    detect as cascade_detect,
    get_embedder,
    get_verifier,
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
    version="0.3.1-day3-cascade",
    description="Phantom tool-call detector — 3-layer cascade live (L1 + L2 + L3).",
)


class _State:
    """Process-local daemon state: registry, warm embeddings, embedder,
    verifier, and SSE subscribers. Initialised on the FastAPI startup hook."""

    def __init__(self) -> None:
        self.registry: ToolRegistry = ToolRegistry()
        self.embedder: Embedder | None = None
        self.verifier: Verifier | None = None
        self.registry_embeddings: dict[str, list[float]] = {}
        self.event_subscribers: list[asyncio.Queue[Decision]] = []
        self.history: collections.deque[dict] = collections.deque(maxlen=200)


_state = _State()


@app.on_event("startup")  # type: ignore[deprecated]
async def _on_startup() -> None:
    """Load registry, build embedder + verifier, warm up registry embeddings."""
    _state.registry = load_registry()
    _state.embedder = get_embedder()
    _state.verifier = get_verifier()  # may be StubVerifier when GEMINI_API_KEY missing

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
        verifier_provider=cfg.verifier.provider,
        verifier_model=cfg.verifier.gemini_model,
        verifier_type=type(_state.verifier).__name__,
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
        "verifier": type(_state.verifier).__name__ if _state.verifier else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/detect", response_model=Decision)
async def detect(req: DetectRequest) -> Decision:
    """Full 3-layer cascade.

    Flow:
      1. Resolve registry + embeddings (per-call override or daemon default).
      2. Hand off to cascade.detect() which runs L1 -> L2 -> (maybe L3).
      3. Broadcast Decision to SSE subscribers + structlog audit.
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

    decision = cascade_detect(
        req,
        registry,
        _state.embedder,
        embeddings,
        verifier=_state.verifier,
    )

    total_elapsed_ms = (time.perf_counter() - request_start) * 1000.0

    # Audit log. ALLOW gets a debug line, anything else gets warning
    # (those are the "interesting" events the dashboard timeline displays).
    log_method = log.debug if decision.verdict == "ALLOW" else log.warning
    log_method(
        "cascade_decision",
        session_id=req.session_id,
        tool=req.tool_name,
        verdict=decision.verdict,
        confidence=round(decision.confidence, 3),
        suggested=decision.suggestion.tool_name if decision.suggestion else None,
        degraded=decision.degraded,
        l1_ms=round(decision.layer_breakdown.l1_ms, 3),
        l2_ms=round(decision.layer_breakdown.l2_ms, 3),
        l3_ms=(
            round(decision.layer_breakdown.l3_ms, 3)
            if decision.layer_breakdown.l3_ms is not None
            else None
        ),
        total_ms=round(total_elapsed_ms, 3),
        registered_count=len(registry.tools),
        agent_content_present=req.agent_content is not None,
    )

    _broadcast(decision, req)
    return decision


_MAX_SSE_SUBSCRIBERS = 50


def _broadcast(decision: Decision, req: DetectRequest) -> None:
    """Fan-out a Decision to all SSE subscribers; never blocks /detect."""
    import json as _json
    record = {
        **decision.model_dump(),
        "tool_name": req.tool_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _state.history.append(record)
    record_json = _json.dumps(record)
    for q in _state.event_subscribers:
        try:
            q.put_nowait(record_json)
        except asyncio.QueueFull:
            pass  # slow subscriber: drop, don't block the cascade


@app.get("/history")
async def history(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """Recent decisions for the dashboard — newest first."""
    items = list(_state.history)[-limit:][::-1]
    total = len(_state.history)
    phantoms = sum(1 for r in _state.history if r["verdict"] != "ALLOW")
    auto_correct = sum(1 for r in _state.history if r["verdict"] == "AUTO_CORRECT")
    block = sum(1 for r in _state.history if r["verdict"] == "BLOCK")
    suggest = sum(1 for r in _state.history if r["verdict"] == "SUGGEST")
    confs = [r["confidence"] for r in _state.history if r["verdict"] != "ALLOW"]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    return {
        "stats": {
            "total": total,
            "phantoms": phantoms,
            "auto_correct": auto_correct,
            "suggest": suggest,
            "block": block,
            "avg_confidence": avg_conf,
        },
        "decisions": items,
    }


@app.get("/events")
async def events() -> StreamingResponse:
    """Server-Sent Events stream of every Decision. Dashboard subscribes here."""
    if len(_state.event_subscribers) >= _MAX_SSE_SUBSCRIBERS:
        from fastapi.responses import Response
        return Response(status_code=503, content="SSE subscriber limit reached")  # type: ignore[return-value]

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=128)
    _state.event_subscribers.append(queue)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            yield ": sentinel stream open\n\n"
            while True:
                record_json = await queue.get()
                yield f"data: {record_json}\n\n"
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


# Serve dashboard HTML — mount AFTER all API routes so /detect etc. take priority
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SENTINEL_PORT", "7777"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
