"""Sentinel public schemas — the only authoritative contracts in the system.

Every module in `backend/sentinel/`, the FastAPI daemon, the Claude Code hook,
the dashboard, and the benchmark consume/produce these exact types. Changes
here ripple everywhere — modify with deliberation.

Constitution Principle III (Confidence-Gated Self-Correction): Decision MUST
populate verdict + confidence + reason. AUTO_CORRECT requires confidence
>= 0.85 (enforced via model_validator). BLOCK requires non-empty reason.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ----------------------------------------------------------------------------
# Verdict literal — the four possible cascade outcomes
# ----------------------------------------------------------------------------

Verdict = Literal["ALLOW", "AUTO_CORRECT", "SUGGEST", "BLOCK"]
"""Cascade decision.

- ALLOW:         tool exists in registry, pass through unchanged
- AUTO_CORRECT:  high-confidence (>=0.85) replacement; inject correction
- SUGGEST:       ambiguous (0.60-0.85); return top-3 candidates to agent
- BLOCK:         no plausible match (<0.60); agent must revise plan
"""


# ----------------------------------------------------------------------------
# Tool registry
# ----------------------------------------------------------------------------

class Tool(BaseModel):
    """A single tool available to the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1, description="Canonical tool name as agent sees it.")]
    description: str = Field(default="", description="Human-readable purpose; consumed by Layer 2 embedding.")
    required_args: tuple[str, ...] = Field(default=(), description="Required argument keys for F2 Jaccard heuristic.")
    optional_args: tuple[str, ...] = Field(default=(), description="Optional argument keys.")


class ToolRegistry(BaseModel):
    """The active tool registry against which detection runs."""

    model_config = ConfigDict(extra="forbid")

    tools: tuple[Tool, ...] = Field(default=(), description="Available tools, ordered by registry sequence.")
    version: str = Field(default="0", description="Registry version tag; bumps invalidate Layer 2 embedding cache.")

    @property
    def names(self) -> tuple[str, ...]:
        """Tool names in original case. Used for display + audit logs."""
        return tuple(t.name for t in self.tools)

    @property
    def names_lower(self) -> frozenset[str]:
        """Tool names lowercased for Layer 1 hash-lookup."""
        return frozenset(t.name.lower() for t in self.tools)

    def find(self, name: str) -> Tool | None:
        """Case-insensitive lookup. Returns None if no match."""
        target = name.lower()
        for t in self.tools:
            if t.name.lower() == target:
                return t
        return None


# ----------------------------------------------------------------------------
# Request — what the hook sends to /detect
# ----------------------------------------------------------------------------

class DetectRequest(BaseModel):
    """Input to the cascade. Hook script translates the Claude Code PreToolUse
    envelope (or any other agent platform's tool-call event) into this shape.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: Annotated[str, Field(min_length=1, description="Tool the agent attempted to invoke.")]
    tool_input: dict = Field(
        default_factory=dict,
        description="Arguments the agent passed; consumed by F2 schema-key Jaccard heuristic.",
    )
    session_id: str = Field(default="default", description="Stable agent-session id; groups traces in dashboard.")
    agent_reasoning: str | None = Field(
        default=None,
        max_length=4096,
        description="Recent <=512 tokens of agent reasoning; used by Layer 3 verifier reasoning-alignment check.",
    )
    agent_content: str | None = Field(
        default=None,
        max_length=16384,
        description=(
            "Assistant `content` field from the model's response, if available. Scanned for ghost claims "
            "(textual phantom tool names appearing without corresponding tool_calls). See "
            "data/evidence/2026-05-16-llama-ghost-claims.md."
        ),
    )
    registry: ToolRegistry | None = Field(
        default=None,
        description="Optional override of the daemon's loaded registry. When None, daemon uses its configured registry.",
    )


# ----------------------------------------------------------------------------
# Decision — the cascade's output (the authoritative contract)
# ----------------------------------------------------------------------------

class Suggestion(BaseModel):
    """A proposed replacement tool when verdict is AUTO_CORRECT or SUGGEST."""

    model_config = ConfigDict(extra="forbid")

    tool_name: Annotated[str, Field(min_length=1)]
    rationale: Annotated[str, Field(max_length=240)]


class LayerBreakdown(BaseModel):
    """Per-layer wall-clock latency for the cascade. Used by benchmark Pareto chart."""

    model_config = ConfigDict(extra="forbid")

    l1_ms: float = Field(ge=0.0, default=0.0, description="Layer 1 (registry exact-match) ms.")
    l2_ms: float = Field(ge=0.0, default=0.0, description="Layer 2 (embedding + F-fusion) ms.")
    l3_ms: float | None = Field(
        ge=0.0,
        default=None,
        description="Layer 3 (Gemini verifier) ms. None when L3 did not fire.",
    )


class GhostClaim(BaseModel):
    """A tool-name-like token observed in assistant `content` that does NOT
    appear in the registry AND is not present in `tool_calls`. The textual
    counterpart to a structured phantom. See evidence corpus for examples.
    """

    model_config = ConfigDict(extra="forbid")

    fragment: Annotated[str, Field(min_length=1, max_length=240, description="Exact substring observed in content.")]
    inferred_name: str = Field(default="", description="Snake-case canonical guess (e.g., 'database_interface_tool').")
    span: tuple[int, int] | None = Field(default=None, description="Character offsets [start, end) in content if known.")


class Decision(BaseModel):
    """The cascade's output. Authoritative across daemon, hook, dashboard, benchmark.

    Invariants (enforced via model_validator):
    - verdict AUTO_CORRECT requires confidence >= 0.85 AND suggestion is not None
    - verdict BLOCK requires len(reason) >= 10
    - verdict SUGGEST requires suggestion is not None
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    reason: Annotated[str, Field(min_length=1, max_length=240)]
    suggestion: Suggestion | None = None
    layer_breakdown: LayerBreakdown = Field(default_factory=LayerBreakdown)
    degraded: bool = Field(default=False, description="True when an L3 timeout or downstream failure forced fallback.")
    ghost_claims: tuple[GhostClaim, ...] = Field(
        default=(),
        description="Phantom tool names observed in assistant content (Day-2 scope refinement).",
    )

    @model_validator(mode="after")
    def _enforce_verdict_invariants(self) -> "Decision":
        if self.verdict == "AUTO_CORRECT":
            if self.confidence < 0.85:
                raise ValueError(
                    f"AUTO_CORRECT requires confidence >= 0.85 (got {self.confidence:.2f}). "
                    "Constitution Principle III violation."
                )
            if self.suggestion is None:
                raise ValueError("AUTO_CORRECT requires a suggestion.")
        elif self.verdict == "SUGGEST":
            if self.suggestion is None:
                raise ValueError("SUGGEST requires a suggestion.")
        elif self.verdict == "BLOCK":
            if len(self.reason) < 10:
                raise ValueError(
                    f"BLOCK requires reason length >= 10 (got {len(self.reason)}). "
                    "Agent needs enough info to revise its plan."
                )
        return self


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """Persisted record of one (DetectRequest, Decision) pair for the dashboard
    timeline and benchmark replay corpus. Stored in SQLite at
    /var/sentinel/sentinel.db (Vultr) or ~/.sentinel/sentinel.db (local).
    """

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    request: DetectRequest
    decision: Decision
    daemon_sha: str = Field(default="", description="Git SHA of the daemon binary at decision time; reproducibility.")
