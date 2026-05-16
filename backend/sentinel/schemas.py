"""Sentinel public schemas — the only authoritative contracts in the system.

Every module in `backend/sentinel/`, the FastAPI daemon, the Claude Code hook,
the dashboard, and the benchmark consume/produce these exact types. Changes
here ripple everywhere — modify with deliberation.

Constitution Principle III (Confidence-Gated Self-Correction): Decision MUST
populate verdict + confidence + reason. AUTO_CORRECT requires confidence
>= cascade.yaml verdict_thresholds.auto_correct_min (default 0.85).

Constitution Principle V (Reproducibility): the AUTO_CORRECT and BLOCK
thresholds load from configs/cascade.yaml via config.load_cascade_config().
No magic floats in this file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from .config import load_cascade_config

# ----------------------------------------------------------------------------
# Verdict literal — the four possible cascade outcomes
# ----------------------------------------------------------------------------

Verdict = Literal["ALLOW", "AUTO_CORRECT", "SUGGEST", "BLOCK"]
"""Cascade decision.

- ALLOW:         tool exists in registry, pass through unchanged
- AUTO_CORRECT:  high-confidence (>= cfg.auto_correct_min) replacement
- SUGGEST:       ambiguous (cfg.block_max .. cfg.auto_correct_min); top-3
- BLOCK:         no plausible match (< cfg.block_max); agent must revise
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
    """Active tool registry. Indices are cached at construction time for O(1)
    lookup on the Layer 1 hot path (Constitution Principle II: <1ms median).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: tuple[Tool, ...] = Field(default=(), description="Available tools, ordered by registry sequence.")
    version: str = Field(default="0", description="Registry version tag; bumps invalidate Layer 2 embedding cache.")

    # Cached indices — built once in the post-init validator below. Using
    # PrivateAttr lets us set these even on a frozen model (object.__setattr__
    # bypass that pydantic provides for private attributes).
    _names_lower: frozenset[str] = PrivateAttr(default=frozenset())
    _by_lower: dict[str, Tool] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _build_indices(self) -> "ToolRegistry":
        """Pre-compute the lowercase index. Runs exactly once per registry."""
        object.__setattr__(self, "_names_lower", frozenset(t.name.lower() for t in self.tools))
        object.__setattr__(self, "_by_lower", {t.name.lower(): t for t in self.tools})
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Tool names in original case. Used for display + audit logs."""
        return tuple(t.name for t in self.tools)

    @property
    def names_lower(self) -> frozenset[str]:
        """Tool names lowercased — O(1) membership test for Layer 1."""
        return self._names_lower

    def find(self, name: str) -> Tool | None:
        """Case-insensitive lookup. O(1) via cached index."""
        return self._by_lower.get(name.lower())


# ----------------------------------------------------------------------------
# Request — what the hook sends to /detect
# ----------------------------------------------------------------------------


class DetectRequest(BaseModel):
    """Input to the cascade. Hook script translates the Claude Code PreToolUse
    envelope (or any other agent platform's tool-call event) into this shape.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description=(
                "Tool the agent attempted to invoke. Capped at 200 chars to bound the audit-log "
                "field size and as a basic DOS guard for oversized payloads. Real tool names "
                "are <=50 chars in every framework I've seen (Claude Code, MCP, OpenAI)."
            ),
        ),
    ]
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
    - verdict AUTO_CORRECT requires confidence >= cfg.auto_correct_min AND suggestion is not None
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
        cfg = load_cascade_config().verdict_thresholds
        if self.verdict == "AUTO_CORRECT":
            if self.confidence < cfg.auto_correct_min:
                raise ValueError(
                    f"AUTO_CORRECT requires confidence >= {cfg.auto_correct_min} "
                    f"(got {self.confidence:.2f}). Constitution Principle III violation."
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
