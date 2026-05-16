"""Contract tests for the Decision schema.

These tests enforce Constitution Principle III (Confidence-Gated Self-Correction):
every Decision carries verdict + confidence + reason, and the four verdicts
each carry the right invariants. If any test here fails, we have shipped a
class of failure that Sentinel's whole pitch depends on preventing.

Run: `pytest backend/tests/contract/test_decision_schema.py -v`
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.schemas import Decision, GhostClaim, LayerBreakdown, Suggestion


# ----------------------------------------------------------------------------
# Happy-path constructions across all four verdicts
# ----------------------------------------------------------------------------


def test_allow_minimal_construction() -> None:
    """ALLOW verdict: registry exact match, conf=1.0, no suggestion required."""
    d = Decision(verdict="ALLOW", confidence=1.0, reason="exact match in registry")
    assert d.verdict == "ALLOW"
    assert d.confidence == 1.0
    assert d.suggestion is None
    assert d.degraded is False
    assert d.ghost_claims == ()


def test_auto_correct_at_threshold_min() -> None:
    """AUTO_CORRECT verdict: confidence exactly at the 0.85 threshold is valid."""
    d = Decision(
        verdict="AUTO_CORRECT",
        confidence=0.85,
        reason="Use mcp__lint_check (semantic match 0.85, schema-compatible). Retry.",
        suggestion=Suggestion(tool_name="mcp__lint_check", rationale="closest semantic match"),
    )
    assert d.verdict == "AUTO_CORRECT"
    assert d.suggestion is not None


def test_suggest_with_candidates() -> None:
    """SUGGEST verdict: ambiguous (0.6-0.85), agent picks from candidates."""
    d = Decision(
        verdict="SUGGEST",
        confidence=0.72,
        reason="Closest match: mcp__lint_check (0.72). Confirm or revise.",
        suggestion=Suggestion(tool_name="mcp__lint_check", rationale="top-1 of 3 candidates"),
    )
    assert d.suggestion is not None


def test_block_with_full_reason() -> None:
    """BLOCK verdict: no plausible match, reason must be actionable."""
    d = Decision(
        verdict="BLOCK",
        confidence=0.92,
        reason="No candidate tool resembles 'phantom_x'. Revise plan with available tools.",
    )
    assert d.verdict == "BLOCK"
    assert d.suggestion is None


# ----------------------------------------------------------------------------
# Negative invariants — Constitution Principle III enforcement
# ----------------------------------------------------------------------------


def test_auto_correct_rejects_confidence_below_threshold() -> None:
    """AUTO_CORRECT MUST NOT be emitted with confidence < 0.85."""
    with pytest.raises(ValidationError, match="AUTO_CORRECT requires confidence >= 0.85"):
        Decision(
            verdict="AUTO_CORRECT",
            confidence=0.84,
            reason="too-confident",
            suggestion=Suggestion(tool_name="x", rationale="y"),
        )


def test_auto_correct_rejects_missing_suggestion() -> None:
    """AUTO_CORRECT MUST carry a suggestion (else what would we auto-correct to?)."""
    with pytest.raises(ValidationError, match="AUTO_CORRECT requires a suggestion"):
        Decision(verdict="AUTO_CORRECT", confidence=0.91, reason="no suggestion field set")


def test_suggest_rejects_missing_suggestion() -> None:
    """SUGGEST MUST carry a suggestion."""
    with pytest.raises(ValidationError, match="SUGGEST requires a suggestion"):
        Decision(verdict="SUGGEST", confidence=0.7, reason="ambiguous match found")


def test_block_rejects_short_reason() -> None:
    """BLOCK MUST carry a reason long enough for the agent to actually revise.
    Constitution Principle III: 'agent must always receive enough information in
    the stderr reason field to understand why a call was modified or rejected'.
    """
    with pytest.raises(ValidationError, match="BLOCK requires reason length >= 10"):
        Decision(verdict="BLOCK", confidence=0.95, reason="nope")


def test_confidence_clamped_to_unit_interval() -> None:
    """Confidence MUST be in [0, 1] per schema declaration."""
    with pytest.raises(ValidationError):
        Decision(verdict="ALLOW", confidence=1.1, reason="impossible")
    with pytest.raises(ValidationError):
        Decision(verdict="ALLOW", confidence=-0.1, reason="impossible")


def test_unknown_verdict_rejected() -> None:
    """Verdict literal MUST be one of the 4 documented values."""
    with pytest.raises(ValidationError):
        Decision(verdict="MAYBE", confidence=0.5, reason="unknown verdict literal")  # type: ignore[arg-type]


def test_reason_max_length() -> None:
    """Reason MUST fit in 240 chars (matches Layer 3 prompt contract + stderr line)."""
    long_reason = "x" * 241
    with pytest.raises(ValidationError):
        Decision(verdict="ALLOW", confidence=1.0, reason=long_reason)


def test_empty_reason_rejected() -> None:
    """Even ALLOW needs a non-empty reason for audit log integrity."""
    with pytest.raises(ValidationError):
        Decision(verdict="ALLOW", confidence=1.0, reason="")


# ----------------------------------------------------------------------------
# Layer breakdown
# ----------------------------------------------------------------------------


def test_layer_breakdown_l3_optional() -> None:
    """Layer 3 only fires on ambiguous; breakdown reflects this with None."""
    d = Decision(
        verdict="ALLOW",
        confidence=1.0,
        reason="registry hit",
        layer_breakdown=LayerBreakdown(l1_ms=0.4, l2_ms=0.0, l3_ms=None),
    )
    assert d.layer_breakdown.l3_ms is None


def test_layer_breakdown_negative_ms_rejected() -> None:
    """Latency MUST be non-negative; physics check."""
    with pytest.raises(ValidationError):
        LayerBreakdown(l1_ms=-0.1, l2_ms=0.0)


# ----------------------------------------------------------------------------
# Ghost claims (Day-2 scope refinement)
# ----------------------------------------------------------------------------


def test_ghost_claims_default_empty() -> None:
    """When no ghost claims observed, tuple is empty (NOT None)."""
    d = Decision(verdict="ALLOW", confidence=1.0, reason="clean")
    assert d.ghost_claims == ()
    assert isinstance(d.ghost_claims, tuple)


def test_ghost_claim_construction() -> None:
    """Captures the Llama-3.1-8B ghost-claims evidence (see data/evidence/)."""
    claim = GhostClaim(
        fragment="Database Interface Tool (DIT)",
        inferred_name="database_interface_tool",
        span=(58, 86),
    )
    assert claim.fragment.startswith("Database")
    assert claim.inferred_name == "database_interface_tool"


def test_decision_with_ghost_claims() -> None:
    """A BLOCK decision can carry observed ghost claims for the audit log."""
    d = Decision(
        verdict="BLOCK",
        confidence=0.94,
        reason="Response references 6 nonexistent tools. No real tool was called.",
        ghost_claims=(
            GhostClaim(fragment="Database Interface Tool", inferred_name="database_interface_tool"),
            GhostClaim(fragment="Data Storage Tool", inferred_name="data_storage_tool"),
        ),
    )
    assert len(d.ghost_claims) == 2
    assert d.ghost_claims[0].inferred_name == "database_interface_tool"


# ----------------------------------------------------------------------------
# JSON round-trip — daemon ↔ hook contract
# ----------------------------------------------------------------------------


def test_decision_json_roundtrip() -> None:
    """A Decision must survive JSON serialization unchanged (hook ↔ daemon path).

    The hook reads JSON from the daemon over HTTP. If round-trip drops a field
    or coerces a value, the hook would emit the wrong exit code → wrong agent
    behavior → demo collapse.
    """
    original = Decision(
        verdict="AUTO_CORRECT",
        confidence=0.91,
        reason="Use mcp__lint_check (sim 0.91, schema-compatible). Retry.",
        suggestion=Suggestion(tool_name="mcp__lint_check", rationale="closest match"),
        layer_breakdown=LayerBreakdown(l1_ms=0.4, l2_ms=7.2, l3_ms=None),
    )

    payload = original.model_dump_json()
    rehydrated = Decision.model_validate_json(payload)

    assert rehydrated == original
