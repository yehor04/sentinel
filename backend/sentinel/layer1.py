"""Layer 1 — registry exact-match.

The cheap, deterministic first line of the cascade. Constitution Principle II
requires this to be O(1) hash lookup with median latency <1ms. No regex, no
embedding, no LLM. Hit → ALLOW (conf 1.0). Miss → fall through to Layer 2.

Inputs are normalized lowercase to absorb the `'Read'` vs `'read'` vs `'READ'`
class of bugs. The Decision returned echoes the registry's canonical casing.
"""

from __future__ import annotations

import time

from .schemas import Decision, LayerBreakdown, ToolRegistry


def layer1(tool_name: str, registry: ToolRegistry) -> Decision | None:
    """Return an ALLOW Decision if tool_name exists in registry, else None.

    Args:
        tool_name: The agent's invoked tool name. Case-insensitive matched.
        registry: The active registry.

    Returns:
        Decision(verdict="ALLOW", confidence=1.0, ...) on hit.
        None on miss (caller falls through to Layer 2).
    """
    start = time.perf_counter()

    hit = tool_name.lower() in registry.names_lower

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not hit:
        return None

    canonical = registry.find(tool_name)
    canonical_name = canonical.name if canonical is not None else tool_name

    return Decision(
        verdict="ALLOW",
        confidence=1.0,
        reason=f"Registry exact match: '{canonical_name}'",
        layer_breakdown=LayerBreakdown(l1_ms=elapsed_ms),
    )
