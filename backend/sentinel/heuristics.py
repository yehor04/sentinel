"""Layer 2 fusion heuristics — F1, F2, F3.

These three deterministic features are blended with the base embedding
similarity to produce the Layer 2 confidence score when the cosine match is
ambiguous (0.60 <= base <= 0.85). Each fires in <1ms; together they catch
failure shapes pure embeddings miss. See `docs/blueprint.md` §4 for the
full derivation and the papers each heuristic is inspired by.

Constitution Principle V: every threshold and weight loads from `configs/`,
nothing hard-coded here.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


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
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
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
        Returns 0.0 when both sets are empty (no signal to fuse).

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
        return 0.0
    return len(keys_a & keys_b) / len(union)


def f3_gap(sims_top3: tuple[float, float, float] | list[float]) -> float:
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

    Returns:
        Clipped (top1 - top2) * 5. So a gap of 0.20 maps to 1.0; gap of
        0.04 maps to 0.20. The 5x multiplier was chosen so that gaps
        comparable to typical embedding noise (~0.05) become non-trivial
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
    gap = sims_top3[0] - sims_top3[1]
    scaled = gap * 5.0
    return max(0.0, min(1.0, scaled))


def fuse(
    base_confidence: float,
    f1: float,
    f2: float,
    f3: float,
    *,
    weights: tuple[float, float, float, float] = (0.5, 0.2, 0.2, 0.1),
) -> float:
    """Blend base cosine confidence with F1/F2/F3 via weighted average.

    Only meant to be called when base_confidence is in the ambiguous range
    [0.60, 0.85] (cascade orchestrator handles that gate). Outside that range
    the heuristics are noise; trust the base.

    Args:
        base_confidence: rescaled cosine top-1, already in [0, 1].
        f1: Levenshtein structural similarity.
        f2: Jaccard schema-key similarity.
        f3: Top-1 vs top-2 gap.
        weights: 4-tuple summing to 1.0, default (0.5, 0.2, 0.2, 0.1).
                 Calibrated empirically on the validation split (T047).

    Returns:
        Fused confidence in [0, 1].

    >>> round(fuse(0.70, 1.0, 1.0, 1.0), 3)
    0.85
    >>> round(fuse(0.70, 0.0, 0.0, 0.0), 3)
    0.35
    """
    w_base, w_f1, w_f2, w_f3 = weights
    if abs(w_base + w_f1 + w_f2 + w_f3 - 1.0) > 1e-6:
        raise ValueError(
            f"Fusion weights must sum to 1.0, got {w_base + w_f1 + w_f2 + w_f3:.4f}"
        )
    raw = w_base * base_confidence + w_f1 * f1 + w_f2 * f2 + w_f3 * f3
    return max(0.0, min(1.0, raw))
