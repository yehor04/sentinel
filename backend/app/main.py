"""Sentinel FastAPI daemon — Day 2 wiring.

Day 1 shipped a fully mocked `/detect` to clear the deploy pipeline. Day 2
wires the real Layer-1 registry check (T025): on each request we run
`layer1(tool_name, registry)`. A hit returns a real ALLOW. A miss currently
returns a placeholder AUTO_CORRECT (mocking what Layer 2/3 will produce on
Day 3, T032).

Structured audit logs (structlog, JSON) are emitted for every intercept so
the dashboard SSE timeline + the bench corpus can reconstruct what happened.

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
    LayerBreakdown,
    Suggestion,
    ToolRegistry,
    layer1,
    load_registry,
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
    version="0.2.0-day2",
    description="Phantom tool-call detector with 3-layer cascade. Day 2 wiring (Layer 1 live).",
)


class _State:
    """Process-local daemon state — registry + SSE subscribers. Initialised on
    startup via the FastAPI lifespan hook below."""

    def __init__(self) -> None:
        self.registry: ToolRegistry = ToolRegistry()
        self.event_subscribers: list[asyncio.Queue[Decision]] = []


_state = _State()


@app.on_event("startup")  # type: ignore[deprecated]
async def _on_startup() -> None:
    """Load registry + cascade config exactly once at boot."""
    _state.registry = load_registry()
    cfg = load_cascade_config()
    log.info(
        "daemon_startup",
        version=app.version,
        registry_version=_state.registry.version,
        registry_size=len(_state.registry.tools),
        registered_names=list(_state.registry.names),
        auto_correct_min=cfg.verdict_thresholds.auto_correct_min,
        block_max=cfg.verdict_thresholds.block_max,
    )


# ---------- Endpoints ----------


@app.get("/health")
async def health() -> dict:
    """Liveness probe — used by Vultr smoke test, Caddy healthcheck."""
    return {
        "status": "ok",
        "version": app.version,
        "registry_loaded": len(_state.registry.tools) > 0,
        "registry_size": len(_state.registry.tools),
        "registry_version": _state.registry.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/detect", response_model=Decision)
async def detect(req: DetectRequest) -> Decision:
    """Live Layer 1 + (mocked Layer 2/3 for Day 2).

    Flow:
      1. Use req.registry if provided; otherwise the daemon's loaded registry.
      2. Run Layer 1 — exact match -> ALLOW.
      3. Miss -> placeholder AUTO_CORRECT (Day 2 mock; T032 lands real L2/L3).
      4. Broadcast to SSE subscribers (dashboard timeline).
      5. structlog audit line for every intercept.
    """
    request_start = time.perf_counter()
    registry = req.registry or _state.registry

    # Layer 1 — registry exact match
    decision = layer1(req.tool_name, registry)

    if decision is not None:
        # ALLOW path — registered tool, pass through
        log.debug(
            "tool_allowed",
            session_id=req.session_id,
            tool=req.tool_name,
            l1_ms=decision.layer_breakdown.l1_ms,
        )
    else:
        # PHANTOM path — Layer 1 miss. Day-2 mock: pretend Layer 2 found a
        # close match in the registry. Day 3 (T028+T032) replaces this with
        # real cascade output.
        l1_elapsed_ms = (time.perf_counter() - request_start) * 1000.0
        decision = _mock_phantom_decision(req, registry, l1_elapsed_ms=l1_elapsed_ms)
        log.warning(
            "phantom_intercepted",
            session_id=req.session_id,
            tool=req.tool_name,
            registered_count=len(registry.tools),
            registered_names=list(registry.names),
            verdict=decision.verdict,
            confidence=decision.confidence,
            suggested=decision.suggestion.tool_name if decision.suggestion else None,
            agent_content_present=req.agent_content is not None,
        )

    # SSE broadcast — never blocks the response path
    for q in _state.event_subscribers:
        try:
            q.put_nowait(decision)
        except asyncio.QueueFull:
            pass  # slow subscriber; drop rather than block

    return decision


def _mock_phantom_decision(
    req: DetectRequest, registry: ToolRegistry, *, l1_elapsed_ms: float
) -> Decision:
    """Placeholder Day-2 response for Layer-1 misses.

    Picks the first registered tool as the "closest semantic match" — pure
    mock; real semantics arrive Day 3 with Layer 2 embeddings. Confidence
    fixed at 0.91 so the AUTO_CORRECT path exercises the schema invariant.
    """
    candidate = registry.tools[0].name if registry.tools else "mcp__lint_check"
    return Decision(
        verdict="AUTO_CORRECT",
        confidence=0.91,
        reason=(
            f"Tool '{req.tool_name}' not in registry. "
            f"Use '{candidate}' (semantic match 0.91, schema-compatible). Retry."
        ),
        suggestion=Suggestion(
            tool_name=candidate,
            rationale="day-2 mock — Layer 2 embedding lands Day 3 (T028)",
        ),
        layer_breakdown=LayerBreakdown(l1_ms=l1_elapsed_ms, l2_ms=0.0, l3_ms=None),
        degraded=False,
    )


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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SENTINEL_PORT", "7777"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
