"""Tool-registry loader.

Reads a YAML manifest into a `ToolRegistry`. Search-path resolution mirrors
`config.py`: env override -> repo-relative -> production /etc/ path. The
daemon calls `load_registry()` exactly once at startup.

Future surfaces (Day-3+):
- MCP `tools/list` introspection for live-MCP-server mode
- Claude Code session-start tool list extraction

For v1 we only need the YAML path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .schemas import Tool, ToolRegistry


def _candidate_paths() -> list[Path]:
    """Search order for registry.yaml — first existing file wins."""
    paths: list[Path] = []
    env_override = os.environ.get("SENTINEL_REGISTRY")
    if env_override:
        paths.append(Path(env_override))
    here = Path(__file__).resolve()
    paths.append(here.parent.parent.parent / "configs" / "registry.yaml")
    paths.append(Path("/etc/sentinel/registry.yaml"))
    return paths


def _coerce_tool(raw: dict[str, Any]) -> Tool:
    """Translate a YAML tool entry into a frozen `Tool`."""
    return Tool(
        name=raw["name"],
        description=raw.get("description", ""),
        required_args=tuple(raw.get("required_args", ())),
        optional_args=tuple(raw.get("optional_args", ())),
    )


def load_registry_yaml(path: Path | str) -> ToolRegistry:
    """Parse a YAML file at `path` into a `ToolRegistry`. Raises on missing
    file or malformed structure (a typo in the registry is a startup-time
    error, never a silent runtime miss).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"registry yaml not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"registry yaml root must be a mapping (got {type(data).__name__})")

    raw_tools = data.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError(f"registry yaml `tools` must be a list (got {type(raw_tools).__name__})")

    tools = tuple(_coerce_tool(t) for t in raw_tools)
    version = str(data.get("version", "0"))
    return ToolRegistry(tools=tools, version=version)


@lru_cache(maxsize=1)
def load_registry() -> ToolRegistry:
    """Load the active registry from the first matching candidate path.

    Cached for the process lifetime. The daemon calls this exactly once at
    startup. Returns an empty registry (which always misses Layer 1) if no
    file is found — this is intentional: a missing registry should be a
    loud "every call falls through to L2/L3", not a silent crash.
    """
    for path in _candidate_paths():
        if path.is_file():
            return load_registry_yaml(path)
    return ToolRegistry()


def clear_cache() -> None:
    """Reset the cached registry — for tests that swap YAML mid-run."""
    load_registry.cache_clear()
