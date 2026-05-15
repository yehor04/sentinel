"""Sentinel FastAPI daemon — Day 1 scaffold.

Public HTTP surface for the cascade. On Day 1 this only mocks /detect with a
fixed AUTO_CORRECT decision so the hook-loop integration can be tested end-to-
end. Real cascade wiring lands on Day 3 (T036).

Run locally: `uvicorn backend.app.main:app --host 0.0.0.0 --port 7777`
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ---------- Schemas (placeholder; final versions land in backend/sentinel/schemas.py on Day 2) ----------

class DetectRequest(BaseModel):
    tool_name: str
    tool_input: dict = Field(default_factory=dict)
    session_id: str = "default"
    agent_reasoning: str | None = None
    # registry intentionally omitted from Day-1 mock; real handler will load via registry.py


class Suggestion(BaseModel):
    tool_name: str
    rationale: str


class LayerBreakdown(BaseModel):
    l1_ms: float = 0.0
    l2_ms: float = 0.0
    l3_ms: float | None = None


class Decision(BaseModel):
    verdict: str  # ALLOW | AUTO_CORRECT | SUGGEST | BLOCK (Literal-typed in real schemas.py)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggestion: Suggestion | None = None
    layer_breakdown: LayerBreakdown = Field(default_factory=LayerBreakdown)
    degraded: bool = False


# ---------- App ----------

app = FastAPI(
    title="Sentinel",
    version="0.1.0-day1",
    description="Phantom tool-call detector with 3-layer cascade. Day 1 mock.",
)


# In-memory broadcast queue for SSE subscribers. Day 1 keeps it trivial; Day 4
# replaces with a per-subscriber asyncio.Queue + persisted backlog from SQLite.
_event_subscribers: list[asyncio.Queue[Decision]] = []


@app.get("/health")
async def health() -> dict:
    """Liveness probe — used by Vultr smoke test and Day 1 deploy verification."""
    return {
        "status": "ok",
        "version": app.version,
        "registry_loaded": False,  # Day 2 wires real registry; today we always say False
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/detect", response_model=Decision)
async def detect(req: DetectRequest) -> Decision:
    """Day 1 mock — returns a fixed AUTO_CORRECT decision regardless of input.

    Real cascade lands in T032 (cascade.detect()). This shape lets the hook
    script + Claude Code + agent retry loop be tested end-to-end today.
    """
    start = time.perf_counter()

    # Fake a Layer 1 miss + Layer 2 high-confidence match. The bait demo prompt
    # invokes `mcp__codequality_assess`; we pretend our cascade routed it to
    # `mcp__lint_check` at confidence 0.91.
    mock_decision = Decision(
        verdict="AUTO_CORRECT",
        confidence=0.91,
        reason=(
            f"Tool '{req.tool_name}' does not exist in registry. "
            "Use 'mcp__lint_check' (semantic match 0.91, schema-compatible). Retry."
        ),
        suggestion=Suggestion(
            tool_name="mcp__lint_check",
            rationale="closest semantic match by embedding similarity (mocked)",
        ),
        layer_breakdown=LayerBreakdown(
            l1_ms=0.4,
            l2_ms=7.2,
            l3_ms=None,  # not invoked in mock
        ),
        degraded=False,
    )

    # Broadcast to SSE subscribers so the dashboard sees something during demo
    for q in _event_subscribers:
        with _ignore_full(q):
            q.put_nowait(mock_decision)

    elapsed_ms = (time.perf_counter() - start) * 1000
    # Day 1: don't enforce latency gate yet; Day 3 latency gate (T035) does
    if elapsed_ms > 50:  # only log slow paths
        print(f"[sentinel] /detect served in {elapsed_ms:.1f}ms (mock)")

    return mock_decision


@app.get("/events")
async def events() -> StreamingResponse:
    """Server-Sent Events stream of every Decision. Dashboard subscribes here."""
    queue: asyncio.Queue[Decision] = asyncio.Queue(maxsize=128)
    _event_subscribers.append(queue)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            # Initial comment so the SSE connection is "live" from the start
            yield ": sentinel stream open\n\n"
            while True:
                decision = await queue.get()
                yield f"data: {decision.model_dump_json()}\n\n"
        finally:
            try:
                _event_subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for live demo
        },
    )


# ---------- Local helpers ----------


class _ignore_full:
    """Context manager that swallows asyncio.QueueFull so a slow dashboard
    subscriber can never block the detect path. Day 4 replaces with a smarter
    per-subscriber backlog policy."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    def __enter__(self) -> "_ignore_full":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is asyncio.QueueFull:
            return True  # suppress
        return False


if __name__ == "__main__":
    # Convenience runner for `python -m backend.app.main`
    import uvicorn

    port = int(os.getenv("SENTINEL_PORT", "7777"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
