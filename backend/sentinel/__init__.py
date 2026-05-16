"""Sentinel — phantom tool-call detector with self-correction for autonomous LLM agents.

Public API surface (Library-First per Constitution Principle I):

    from sentinel import detect, Decision, DetectRequest, ToolRegistry

The FastAPI daemon, Claude Code hook, and MCP middleware are all thin clients
over the functions and types exposed here.
"""

from __future__ import annotations

from .schemas import (
    Decision,
    DetectRequest,
    GhostClaim,
    LayerBreakdown,
    Suggestion,
    Tool,
    ToolRegistry,
    TraceEvent,
    Verdict,
)

__all__ = [
    "Decision",
    "DetectRequest",
    "GhostClaim",
    "LayerBreakdown",
    "Suggestion",
    "Tool",
    "ToolRegistry",
    "TraceEvent",
    "Verdict",
]

__version__ = "0.2.0-day2"
