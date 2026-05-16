"""Contract tests for the DetectRequest schema.

DetectRequest is the hook → daemon contract. Any field added here must remain
backward-compatible (additive). Removing or renaming fields breaks all installed
hooks in the field.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.schemas import DetectRequest, Tool, ToolRegistry


def test_minimal_request_construction() -> None:
    """The smallest valid request: just a tool name."""
    req = DetectRequest(tool_name="Read")
    assert req.tool_name == "Read"
    assert req.tool_input == {}
    assert req.session_id == "default"
    assert req.agent_reasoning is None
    assert req.agent_content is None
    assert req.registry is None


def test_full_request_construction() -> None:
    """All optional fields populated — the hook's richest possible payload."""
    req = DetectRequest(
        tool_name="mcp__codequality_assess",
        tool_input={"file": "auth.py", "strict": True},
        session_id="abc-123",
        agent_reasoning="I need to verify auth.py meets the new security policy...",
        agent_content="I will use Database Interface Tool to save findings.",
        registry=ToolRegistry(tools=(Tool(name="Read"), Tool(name="mcp__lint_check"))),
    )
    assert req.session_id == "abc-123"
    assert req.registry is not None
    assert len(req.registry.tools) == 2


def test_empty_tool_name_rejected() -> None:
    """Detection requires a tool name; empty is meaningless."""
    with pytest.raises(ValidationError):
        DetectRequest(tool_name="")


def test_agent_reasoning_max_length() -> None:
    """4096-char cap matches Layer 3 prompt context budget."""
    with pytest.raises(ValidationError):
        DetectRequest(tool_name="x", agent_reasoning="x" * 4097)


def test_agent_content_max_length() -> None:
    """16384-char cap on assistant content — enough for typical responses,
    rejects pathological payloads that would blow Layer 3 cost.
    """
    with pytest.raises(ValidationError):
        DetectRequest(tool_name="x", agent_content="x" * 16385)


def test_extra_fields_forbidden() -> None:
    """Unknown fields trip the schema — protects against silent contract drift
    where a hook sends a field the daemon ignores (or vice versa).
    """
    with pytest.raises(ValidationError):
        DetectRequest.model_validate({"tool_name": "x", "unknown_field": 42})


def test_request_json_roundtrip() -> None:
    """The hook serializes to JSON over HTTP. Round-trip must preserve everything."""
    original = DetectRequest(
        tool_name="mcp__codequality_assess",
        tool_input={"file": "auth.py"},
        session_id="rt-test",
        agent_reasoning="some reasoning",
        agent_content="some content",
    )
    payload = original.model_dump_json()
    rehydrated = DetectRequest.model_validate_json(payload)
    assert rehydrated == original


def test_registry_case_insensitive_find() -> None:
    """ToolRegistry.find() is case-insensitive (matches Layer 1 behavior).
    This protects against the 'Read' vs 'read' vs 'READ' name-mismatch class of bugs.
    """
    reg = ToolRegistry(tools=(Tool(name="Read"), Tool(name="mcp__lint_check")))
    assert reg.find("read") is not None
    assert reg.find("READ") is not None
    assert reg.find("Read") is not None
    assert reg.find("mcp__LINT_check") is not None
    assert reg.find("nonexistent") is None


def test_registry_names_lower_is_frozenset() -> None:
    """ToolRegistry.names_lower is the L1 lookup target — must be a frozenset
    for O(1) membership test (latency budget).
    """
    reg = ToolRegistry(tools=(Tool(name="Read"), Tool(name="mcp__Lint_Check")))
    assert reg.names_lower == frozenset({"read", "mcp__lint_check"})
    assert isinstance(reg.names_lower, frozenset)
