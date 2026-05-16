"""Layer 1 unit tests — registry exact-match behavior + latency guarantee."""

from __future__ import annotations

import time

from sentinel.layer1 import layer1
from sentinel.schemas import Tool, ToolRegistry


_REG = ToolRegistry(
    tools=(
        Tool(name="Read", description="Read a file"),
        Tool(name="Edit", description="Edit a file"),
        Tool(name="mcp__lint_check", description="Static analysis lint pass"),
        Tool(name="mcp__test_runner", description="Run the project's tests"),
    )
)


def test_exact_hit_returns_allow() -> None:
    d = layer1("Read", _REG)
    assert d is not None
    assert d.verdict == "ALLOW"
    assert d.confidence == 1.0
    assert "Read" in d.reason


def test_miss_returns_none() -> None:
    """Phantom name not in registry → None → cascade continues to Layer 2."""
    d = layer1("mcp__codequality_assess", _REG)
    assert d is None


def test_case_insensitive_match() -> None:
    """READ / Read / read all hit the same registry entry; output preserves canonical casing."""
    for variant in ("read", "Read", "READ", "ReAd"):
        d = layer1(variant, _REG)
        assert d is not None, f"variant '{variant}' should hit"
        assert d.verdict == "ALLOW"
        # Reason echoes the registry's canonical name, not the variant the agent used
        assert "'Read'" in d.reason


def test_case_insensitive_match_underscore_name() -> None:
    """MCP-style names with underscores follow the same rule."""
    d = layer1("MCP__LINT_check", _REG)
    assert d is not None
    assert "'mcp__lint_check'" in d.reason


def test_empty_registry_misses_everything() -> None:
    empty = ToolRegistry()
    assert layer1("Read", empty) is None
    assert layer1("anything", empty) is None


def test_empty_tool_name_misses() -> None:
    """Empty agent-provided name shouldn't accidentally match an empty registry slot."""
    # Note: DetectRequest schema rejects empty tool_name upstream, but layer1
    # should be safe even if a caller bypasses validation.
    # This test documents the safety behavior.
    assert layer1("", _REG) is None


def test_l1_latency_under_budget() -> None:
    """Constitution Principle II: Layer 1 median <1ms, p95 <2ms.

    This is a smoke check — not a rigorous benchmark. The full latency gate
    lives in `make bench-latency` (T035). Here we just assert any single call
    completes in well under our budget on dev hardware.
    """
    # Warm up
    for _ in range(10):
        layer1("Read", _REG)

    # 100 runs, take max
    deadline_ms = 5.0  # generous on dev laptop; tighter on Vultr
    max_observed = 0.0
    for _ in range(100):
        t0 = time.perf_counter()
        layer1("Read", _REG)
        dt = (time.perf_counter() - t0) * 1000.0
        max_observed = max(max_observed, dt)

    assert max_observed < deadline_ms, (
        f"Layer 1 exceeded {deadline_ms}ms ceiling: max {max_observed:.3f}ms over 100 runs"
    )
