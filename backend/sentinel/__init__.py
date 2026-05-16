"""Sentinel — phantom tool-call detector with self-correction for autonomous LLM agents.

Public API surface (Library-First per Constitution Principle I):

    from sentinel import detect, Decision, DetectRequest, ToolRegistry
    from sentinel.config import load_cascade_config
    from sentinel.registry import load_registry

The FastAPI daemon, Claude Code hook, and MCP middleware are all thin clients
over the functions and types exposed here.
"""

from __future__ import annotations

from .config import (
    CascadeConfig,
    EmbeddingConfig,
    FusionWeights,
    VerdictThresholds,
    load_cascade_config,
)
from .embeddings import Embedder, EmbeddingError, FeatherlessEmbedder, StubEmbedder, get_embedder
from .layer1 import layer1
from .layer2 import layer2, phantom_signature, tool_signature, warm_up_registry
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
    "layer2",
    "phantom_signature",
    "tool_signature",
    "warm_up_registry",
    # embeddings
    "Embedder",
    "EmbeddingError",
    "FeatherlessEmbedder",
    "StubEmbedder",
    "get_embedder",
    # config
    "CascadeConfig",
    "EmbeddingConfig",
    "FusionWeights",
    "VerdictThresholds",
    "load_cascade_config",
    # registry
    "load_registry",
    "load_registry_yaml",
]

__version__ = "0.2.0-day2"
