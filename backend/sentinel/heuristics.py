"""Layer 2 fusion heuristics — F1, F2, F3.

Three deterministic features blended with the base embedding similarity when
the cosine match is ambiguous (`block_max <= base <= auto_correct_min`).
Each fires in <1ms; together they catch failure shapes pure embeddings miss.
See `docs/blueprint.md` §4 for derivation; the papers each heuristic is
inspired by are cited inline.

Constitution Principle V: every threshold + weight loads from
`configs/cascade.yaml` via `config.load_cascade_config()`. Nothing hard-coded
here; defaults below come from the loaded config.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from .config import load_cascade_config


def f1_levenshtein(a: str, b: str) -> float:
    """F1 — normalized Levenshtein similarity in [0, 1].

    Captures **near-typo** phantoms: agent invokes `mcp__lint_checks` (with
    trailing 's') when the registry has `mcp__lint_check`. Identical strings
    return 1.0; entirely different strings return ~0.0.

    Inspired by Healy IR §3 Type 1 — Function Selection Error where the
    fabricated name is structurally close to a real tool. Implementation uses
    rapidfuzz (fast C implementation, ~1µs for sub-100-char strings).

    Args:
        a: First string (typically phantom tool name).
        b: Second string (typically top-1 registry candidate name).

    Returns:
        Similarity in [0.0, 1.0]. 1.0 == identical. 0.0 == max edit distance.

    >>> round(f1_levenshtein("mcp__lint_checks", "mcp__lint_check"), 3)
    0.938
    >>> f1_levenshtein("save_db", "save_db")
    1.0
    >>> f1_levenshtein("", "")
    1.0
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        # Convention: two empty strings are maximally similar.
        return 1.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - (dist / max_len)


def f2_jaccard(keys_a: set[str], keys_b: set[str]) -> float:
    """F2 — Jaccard similarity over argument-key sets in [0, 1].

    Captures **schema-twin** phantoms: agent built the right call *shape*
    (same arg keys) but the wrong tool name. High F2 → high confidence the
    schema-matching tool is the intended target.

    Inspired by Healy IR §3 Type 3 (Parameter Error) + Type 4 (Completeness
    Error): if argument keys align with a real tool's signature, the model's
    intent was that tool.

    Args:
        keys_a: Argument keys from the phantom call.
        keys_b: Required argument keys for the candidate real tool.

    Returns:
        |A ∩ B| / |A ∪ B|. 1.0 == identical key sets. 0.0 == disjoint.

    Empty-set convention: when *both* sets are empty we return 0.0, NOT 1.0.
    The vacuously-true Jaccard interpretation would spuriously boost
    confidence on argument-less tools (e.g., both a phantom and a real tool
    take no args -> false 1.0 match on a signal that carries no information).
    Returning 0.0 keeps F2 strictly informative; it only contributes signal
    when at least one side actually has keys.

    >>> f2_jaccard({"file", "strict"}, {"file", "strict"})
    1.0
    >>> f2_jaccard({"file"}, {"file", "strict"})
    0.5
    >>> f2_jaccard({"a"}, {"b"})
    0.0
    >>> f2_jaccard(set(), set())
    0.0
    """
    union = keys_a | keys_b
    if not union:
        # No keys on either side = no signal; don't spuriously boost. See
        # the "Empty-set convention" note in the docstring above for why
        # we deliberately return 0.0 here instead of 1.0.
        return 0.0
    return len(keys_a & keys_b) / len(union)


def f3_gap(
    sims_top3: tuple[float, float, float] | list[float],
    *,
    multiplier: float | None = None,
) -> float:
    """F3 — top-1 vs top-2 similarity gap, scaled and clipped to [0, 1].

    Captures the **"Loud Liar" inverse**: a large gap between best and
    second-best candidate means the agent's intent is unambiguous (clean
    intent → high confidence). A small gap means multiple candidates
    plausibly fit (ambiguous → escalate to Layer 3).

    Inspired by Spectral Guardrails (Noël, Feb 2026) §4.2: the "Loud Liar"
    phenomenon shows that catastrophic hallucinations are paradoxically
    easier to detect than subtle ones. At the embedding layer, a dominant
    top-1 plays the analogous role.

    Args:
        sims_top3: Cosine similarities of top-3 candidates, descending.
                   If fewer than 2 entries present, returns 0.0.
        multiplier: Scale factor for (top1-top2) before clipping. Defaults to
                    `configs/cascade.yaml` fusion.f3_multiplier (5.0 baseline).
                    Tests pass explicit values; production reads config.

    Returns:
        Clipped (top1 - top2) * multiplier. With default 5.0, a gap of 0.20
        maps to 1.0; gap of 0.04 maps to 0.20. The multiplier was chosen so
        gaps comparable to typical embedding noise (~0.05) become non-trivial
        signal, while gaps below noise floor stay near zero.

    >>> f3_gap([0.91, 0.71, 0.55])
    1.0
    >>> round(f3_gap([0.80, 0.78, 0.70]), 3)
    0.1
    >>> f3_gap([0.5])
    0.0
    >>> f3_gap([])
    0.0
    """
    if len(sims_top3) < 2:
        return 0.0
    if multiplier is None:
        multiplier = load_cascade_config().fusion.f3_multiplier
    gap = sims_top3[0] - sims_top3[1]
    scaled = gap * multiplier
    return max(0.0, min(1.0, scaled))


def fuse(
    base_confidence: float,
    f1: float,
    f2: float,
    f3: float,
    *,
    weights: tuple[float, float, float, float] | None = None,
) -> float:
    """Blend base cosine confidence with F1/F2/F3 via weighted average.

    Only meant to be called when `base_confidence` is in the ambiguous range
    `[block_max, auto_correct_min)` (cascade orchestrator handles that gate).
    Outside that range the heuristics are noise; trust the base.

    Args:
        base_confidence: rescaled cosine top-1, already in [0, 1].
        f1: Levenshtein structural similarity.
        f2: Jaccard schema-key similarity.
        f3: Top-1 vs top-2 gap.
        weights: 4-tuple summing to 1.0. Defaults to
                 `configs/cascade.yaml` fusion weights. Tests pass explicit
                 values; production reads config.

    Returns:
        Fused confidence in [0, 1].

    >>> round(fuse(0.70, 1.0, 1.0, 1.0, weights=(0.5, 0.2, 0.2, 0.1)), 3)
    0.85
    >>> round(fuse(0.70, 0.0, 0.0, 0.0, weights=(0.5, 0.2, 0.2, 0.1)), 3)
    0.35
    """
    if weights is None:
        weights = load_cascade_config().fusion.as_tuple
    w_base, w_f1, w_f2, w_f3 = weights
    total = w_base + w_f1 + w_f2 + w_f3
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fusion weights must sum to 1.0, got {total:.4f}")
    raw = w_base * base_confidence + w_f1 * f1 + w_f2 * f2 + w_f3 * f3
    return max(0.0, min(1.0, raw))
