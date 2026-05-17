"""Layer 3 — Gemini Flash semantic verifier.

When Layer 2 lands a base confidence in the ambiguous window
[block_max, auto_correct_min), the cascade escalates to Layer 3. The
verifier sees the user's task (via agent_reasoning), the agent's proposed
call, the active registry, and Layer 2's top-3 candidates, then produces
a structured JSON judgment that the cascade orchestrator (T032) fuses
with the Layer 2 result.

Architecture:
- `LLMClient` Protocol — anything that turns a prompt into a JSON string.
- `GeminiClient` — real implementation via `google-generativeai`.
- `Verifier` Protocol — anything that turns a `DetectRequest` + candidates
  into a `VerifierResponse`.
- `GeminiFlashVerifier` — real implementation; builds the prompt per
  `docs/blueprint.md` §4, calls Gemini, parses + validates the JSON.
- `StubVerifier` — returns a configured response (or None). Used in tests
  and as the boot-safe fallback when `GEMINI_API_KEY` is missing.
- `get_verifier()` — LRU-cached factory.

Constitution Principle IV (Demo-First): the daemon MUST boot even with no
Gemini API key. Missing key -> StubVerifier(response=None) -> cascade
gracefully degrades to the Layer 2 decision with `degraded=True`.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Annotated, Protocol, Sequence, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import load_cascade_config
from .schemas import DetectRequest, Tool, ToolRegistry, Verdict

log = structlog.get_logger("sentinel.layer3")


# ----------------------------------------------------------------------------
# Verifier output schema — what Gemini MUST return
# ----------------------------------------------------------------------------


class VerifierSuggestion(BaseModel):
    """A proposed replacement tool from Layer 3's verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: Annotated[str, Field(min_length=1)]
    rationale: Annotated[str, Field(max_length=240)]


class VerifierResponse(BaseModel):
    """Strict JSON the Layer 3 verifier MUST produce.

    Mirrors the Decision schema's verdict/confidence/reason/suggestion
    shape so the cascade orchestrator can fuse trivially. The cascade
    NEVER trusts a VerifierResponse blindly — it applies the same
    Decision invariants when constructing the final response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Verdict
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    reason: Annotated[str, Field(min_length=1, max_length=240)]
    suggestion: VerifierSuggestion | None = None


# ----------------------------------------------------------------------------
# LLM transport layer — kept thin so tests can mock it directly
# ----------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns a prompt into a string response."""

    def generate(self, prompt: str) -> str:
        """Return the raw response body. Implementations raise on transport
        failures so the verifier can catch and degrade."""
        ...


class GeminiClient:
    """Thin wrapper over `google.generativeai` for Gemini Flash structured output."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty; cannot init GeminiClient.")
        # Lazy import: tests that exclusively use StubVerifier don't need
        # google-generativeai installed.
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model_name = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
            request_options={"timeout": self.timeout_s},
        )
        return response.text  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------
# Verifier — Layer 3's actual contract surface
# ----------------------------------------------------------------------------


@runtime_checkable
class Verifier(Protocol):
    """Anything that turns a phantom + context into a VerifierResponse."""

    def verify(
        self,
        req: DetectRequest,
        registry: ToolRegistry,
        top_candidates: Sequence[tuple[float, Tool]],
    ) -> VerifierResponse | None:
        """Return a VerifierResponse on success. Return None on any failure
        (timeout, parse error, schema-violating verdict) so the cascade
        orchestrator can degrade to Layer 2's decision."""
        ...


# ----------------------------------------------------------------------------
# Gemini Flash verifier — real implementation
# ----------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are Sentinel-Verify, a tool-call auditor running \
between an autonomous LLM agent and its tool registry. Your job is to detect \
Function Selection Errors — tool-call attempts where the named function does \
not exist in the registry.

Your OUTPUT MUST be a single strict JSON object with this exact shape:

{
  "verdict":    "ALLOW" | "AUTO_CORRECT" | "SUGGEST" | "BLOCK",
  "confidence": float in [0.0, 1.0],
  "reason":     string <= 240 chars (returned to the agent on intercept),
  "suggestion": { "tool_name": string, "rationale": string } | null
}

DECISION RULES (apply IN ORDER):

(1) If PROPOSED_CALL.tool_name appears verbatim in REGISTRY -> ALLOW, conf=1.0.

(2) Else evaluate L2_CANDIDATES against AGENT_REASONING + PROPOSED_CALL.args:
    - top-1 is semantic match AND sim >= 0.85 -> AUTO_CORRECT, conf>=0.85.
    - 0.60 <= sim < 0.85 with plausible match -> SUGGEST, conf in [0.6, 0.85).
    - no candidate plausible -> BLOCK, conf high.

(3) Integrity checks before finalizing:
    - schema_compatibility: if PROPOSED_CALL.tool_input keys are incompatible
      with suggestion's required_args, DOWNGRADE one tier (AUTO_CORRECT ->
      SUGGEST, SUGGEST -> BLOCK).
    - reasoning_alignment: if AGENT_REASONING explicitly names a DIFFERENT
      registered tool by name, OVERRIDE and AUTO_CORRECT to that tool
      (conf=0.95).

FORMAT "reason" as actionable error text the agent can use to retry. Example:
"Tool 'mcp__codequality_assess' not found. Use 'mcp__lint_check' (semantic \
match 0.91, schema-compatible). Retry."

Do NOT speculate. Do NOT call tools. Do NOT exceed 240 chars in reason.
Do NOT include markdown code fences in the response.
"""


class GeminiFlashVerifier:
    """Gemini Flash-backed Layer 3 verifier.

    Calls `client.generate(prompt)`, parses the JSON, validates it against
    `VerifierResponse`. Any failure path (timeout, malformed JSON, schema
    violation) returns None — the cascade orchestrator decides what to do.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def verify(
        self,
        req: DetectRequest,
        registry: ToolRegistry,
        top_candidates: Sequence[tuple[float, Tool]],
    ) -> VerifierResponse | None:
        prompt = self._build_prompt(req, registry, top_candidates)

        try:
            raw = self.client.generate(prompt)
        except Exception as e:  # broad on purpose — any transport failure
            log.warning("layer3_transport_failed", error=str(e), error_type=type(e).__name__)
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("layer3_json_parse_failed", error=str(e), raw_head=raw[:120] if raw else "")
            return None

        try:
            return VerifierResponse.model_validate(data)
        except ValidationError as e:
            log.warning("layer3_schema_violation", error=str(e), raw_head=raw[:120] if raw else "")
            return None

    def _build_prompt(
        self,
        req: DetectRequest,
        registry: ToolRegistry,
        top_candidates: Sequence[tuple[float, Tool]],
    ) -> str:
        """Render the structured prompt. Kept deterministic so Gemini's
        prompt-cache hit-rate stays high on repeat patterns."""
        registry_lines = "\n".join(
            f"  - {t.name}: {t.description or '(no description)'} "
            f"(required_args: {list(t.required_args) or '[]'})"
            for t in registry.tools
        )
        candidate_lines = "\n".join(
            f"  {i + 1}. {tool.name} (cosine_sim={sim:.3f})"
            for i, (sim, tool) in enumerate(top_candidates)
        )

        agent_reasoning = req.agent_reasoning or "(none provided)"
        tool_input_repr = json.dumps(req.tool_input, ensure_ascii=False, sort_keys=True)

        return f"""{_SYSTEM_PROMPT}

INPUT:

PROPOSED_CALL:
  tool_name:  "{req.tool_name}"
  tool_input: {tool_input_repr}

AGENT_REASONING (most recent <=512 tokens, may be empty):
{agent_reasoning}

REGISTRY (active tools the agent has):
{registry_lines or '  (empty registry)'}

L2_CANDIDATES (top-3 by embedding similarity, descending):
{candidate_lines or '  (no candidates)'}

Now produce the strict JSON object. No prose. No markdown.
"""


# ----------------------------------------------------------------------------
# Stub verifier — tests + boot-safe fallback
# ----------------------------------------------------------------------------


class StubVerifier:
    """A verifier whose `verify()` always returns a preconfigured response
    (or None). Used by tests AND as the cascade's default when
    `GEMINI_API_KEY` is missing — in that case `response=None` ensures the
    cascade orchestrator falls back to the Layer 2 decision with degraded=True.
    """

    def __init__(self, response: VerifierResponse | None = None) -> None:
        self.response = response
        self.call_count = 0
        self.last_call: dict | None = None

    def verify(
        self,
        req: DetectRequest,
        registry: ToolRegistry,
        top_candidates: Sequence[tuple[float, Tool]],
    ) -> VerifierResponse | None:
        self.call_count += 1
        self.last_call = {
            "tool_name": req.tool_name,
            "registry_size": len(registry.tools),
            "candidates": [(s, t.name) for s, t in top_candidates],
        }
        return self.response


# ----------------------------------------------------------------------------
# Factory + singleton
# ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_verifier() -> Verifier:
    """Build the configured verifier once per process.

    Selection priority:
      1. `verifier.provider` from configs/cascade.yaml
      2. Environment override `SENTINEL_VERIFIER_PROVIDER` (rare; tests).
      3. Hard fallback to `StubVerifier(response=None)` whenever:
         - Provider is unknown
         - Provider is gemini but `GEMINI_API_KEY` is missing
         - Gemini SDK init raises (bad install / model name / etc.)

    A `StubVerifier(response=None)` signals the cascade orchestrator that
    Layer 3 is effectively inactive — every `verify()` returns None,
    cascade falls back to Layer 2 with `degraded=True`.
    """
    cfg = load_cascade_config().verifier
    provider = os.environ.get("SENTINEL_VERIFIER_PROVIDER", cfg.provider)

    if provider == "stub":
        log.info("verifier_init", provider="stub")
        return StubVerifier(response=None)

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            log.warning("verifier_fallback_to_stub", reason="GEMINI_API_KEY missing")
            return StubVerifier(response=None)
        try:
            client = GeminiClient(
                api_key=api_key,
                model=cfg.gemini_model,
                timeout_s=cfg.timeout_s,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
            )
            log.info("verifier_init", provider="gemini", model=cfg.gemini_model)
            return GeminiFlashVerifier(client=client)
        except Exception as e:
            log.warning("verifier_fallback_to_stub", reason=str(e))
            return StubVerifier(response=None)

    log.warning("unknown_verifier_provider", provider=provider)
    return StubVerifier(response=None)


def reset_verifier_cache() -> None:
    """Clear the singleton — tests use this between scenarios."""
    get_verifier.cache_clear()


# ----------------------------------------------------------------------------
# Convenience wrapper used by cascade.py (T032)
# ----------------------------------------------------------------------------


def layer3(
    req: DetectRequest,
    registry: ToolRegistry,
    top_candidates: Sequence[tuple[float, Tool]],
    verifier: Verifier,
) -> tuple[VerifierResponse | None, float]:
    """Run Layer 3 verification, measure latency.

    Returns:
        (response, elapsed_ms). `response` is None on any verifier failure
        (transport, parse, schema). The cascade orchestrator handles the
        degraded-fallback logic.
    """
    start = time.perf_counter()
    response = verifier.verify(req, registry, top_candidates)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return response, elapsed_ms
