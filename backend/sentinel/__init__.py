"""Sentinel — phantom tool-call detector with self-correction for autonomous LLM agents.

Public API surface (Library-First per Constitution Principle I):

    from sentinel import detect, Decision, DetectRequest, ToolRegistry
    from sentinel.config import load_cascade_config
    from sentinel.registry import load_registry

The FastAPI daemon, Claude Code hook, and MCP middleware are all thin clients
over the functions and types exposed here.
"""

from __future__ import annotations

from .config import CascadeConfig, FusionWeights, VerdictThresholds, load_cascade_config
from .layer1 import layer1
from .registry import load_registry, load_registry_yaml
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
    # schemas
    "Decision",
    "DetectRequest",
    "GhostClaim",
    "LayerBreakdown",
    "Suggestion",
    "Tool",
    "ToolRegistry",
    "TraceEvent",
    "Verdict",
    # cascade
    "layer1",
    # config
    "CascadeConfig",
    "FusionWeights",
    "VerdictThresholds",
    "load_cascade_config",
    # registry
    "load_registry",
    "load_registry_yaml",
]

__version__ = "0.2.0-day2"
