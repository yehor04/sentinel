"""Heuristics unit tests — explicit numerical assertions per F1/F2/F3.

Every assertion uses hand-computed expected values so that any future drift
in the math will be caught loudly. The blueprint references these specific
heuristics in §4; the spec references their numerical behavior — they cannot
change silently.
"""

from __future__ import annotations

import math

import pytest

from sentinel.heuristics import f1_levenshtein, f2_jaccard, f3_gap, fuse


# ============================================================================
# F1 — Levenshtein similarity
# ============================================================================


def test_f1_identical_strings_returns_one() -> None:
    assert f1_levenshtein("Read", "Read") == 1.0


def test_f1_both_empty_returns_one() -> None:
    """Convention: two empty strings are 'maximally similar'."""
    assert f1_levenshtein("", "") == 1.0


def test_f1_one_empty_returns_zero() -> None:
    """Empty vs non-empty has max normalized edit distance."""
    assert f1_levenshtein("", "Read") == 0.0
    assert f1_levenshtein("Read", "") == 0.0


def test_f1_near_typo_returns_high() -> None:
    """The 'plural-s' typo case: `mcp__lint_checks` vs `mcp__lint_check`."""
    val = f1_levenshtein("mcp__lint_checks", "mcp__lint_check")
    # 1 edit (deletion of 's'), max_len = 16 → 1 - 1/16 = 0.9375
    assert math.isclose(val, 0.9375, abs_tol=0.001)


def test_f1_completely_different_returns_low() -> None:
    val = f1_levenshtein("foo", "barbaz")
    # all 3 chars of "foo" differ + 3 insertions for "baz" → distance 6, max_len 6 → 0.0
    assert val == 0.0


def test_f1_three_char_swap() -> None:
    """`foo` vs `fop` = 1 substitution / 3 chars = 0.667 similarity."""
    val = f1_levenshtein("foo", "fop")
    assert math.isclose(val, 2.0 / 3.0, abs_tol=0.001)


def test_f1_symmetric() -> None:
    """Levenshtein is symmetric — order shouldn't matter."""
    a = f1_levenshtein("Read", "REader")
    b = f1_levenshtein("REader", "Read")
    assert math.isclose(a, b, abs_tol=0.001)


# ============================================================================
# F2 — Jaccard schema-key similarity
# ============================================================================


def test_f2_identical_sets_returns_one() -> None:
    assert f2_jaccard({"file", "strict"}, {"file", "strict"}) == 1.0


def test_f2_both_empty_returns_zero() -> None:
    """Convention: no key signal to fuse → 0.0 (NOT 1.0; we don't want to
    spuriously boost confidence on argument-less tools)."""
    assert f2_jaccard(set(), set()) == 0.0


def test_f2_half_overlap() -> None:
    """{file} ∩ {file, strict} = {file}; union = {file, strict} → 1/2."""
    val = f2_jaccard({"file"}, {"file", "strict"})
    assert val == 0.5


def test_f2_disjoint_returns_zero() -> None:
    assert f2_jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_f2_three_two_overlap() -> None:
    """{a, b, c} ∩ {b, c, d} = {b, c}; union = {a, b, c, d} → 2/4 = 0.5."""
    val = f2_jaccard({"a", "b", "c"}, {"b", "c", "d"})
    assert val == 0.5


# ============================================================================
# F3 — Top-1 vs top-2 gap
# ============================================================================


def test_f3_large_gap_clipped_to_one() -> None:
    """Gap of 0.20 * 5 = 1.0 (right at the cap)."""
    assert f3_gap([0.91, 0.71, 0.55]) == 1.0


def test_f3_larger_than_threshold_stays_one() -> None:
    """Gap of 0.50 * 5 = 2.5 → clipped to 1.0."""
    assert f3_gap([0.95, 0.45, 0.20]) == 1.0


def test_f3_small_gap_scales_linearly() -> None:
    """Gap of 0.02 * 5 = 0.10."""
    val = f3_gap([0.80, 0.78, 0.70])
    assert math.isclose(val, 0.10, abs_tol=0.001)


def test_f3_negative_gap_clipped_to_zero() -> None:
    """If the input isn't sorted descending, F3 still returns >=0."""
    val = f3_gap([0.5, 0.9, 0.3])
    assert val == 0.0


def test_f3_zero_gap_returns_zero() -> None:
    """Tied top-1 and top-2 → maximally ambiguous → escalate."""
    assert f3_gap([0.80, 0.80, 0.70]) == 0.0


def test_f3_fewer_than_two_returns_zero() -> None:
    """No second candidate to compare against."""
    assert f3_gap([0.9]) == 0.0
    assert f3_gap([]) == 0.0


# ============================================================================
# Fusion
# ============================================================================


def test_fuse_default_weights_max() -> None:
    """All F-features at 1.0, base at 1.0 → result 1.0 (within IEEE 754 noise)."""
    assert math.isclose(fuse(1.0, 1.0, 1.0, 1.0), 1.0, abs_tol=1e-9)


def test_fuse_default_weights_min() -> None:
    """All inputs zero → result zero."""
    assert fuse(0.0, 0.0, 0.0, 0.0) == 0.0


def test_fuse_base_dominated() -> None:
    """0.5 * 0.70 + 0.2*0 + 0.2*0 + 0.1*0 = 0.35."""
    val = fuse(0.70, 0.0, 0.0, 0.0)
    assert math.isclose(val, 0.35, abs_tol=0.001)


def test_fuse_all_features_max() -> None:
    """0.5 * 0.70 + 0.2*1 + 0.2*1 + 0.1*1 = 0.85."""
    val = fuse(0.70, 1.0, 1.0, 1.0)
    assert math.isclose(val, 0.85, abs_tol=0.001)


def test_fuse_custom_weights_validated() -> None:
    """Weights must sum to 1.0 (sanity check on calibration outputs)."""
    val = fuse(0.5, 0.5, 0.5, 0.5, weights=(0.25, 0.25, 0.25, 0.25))
    assert math.isclose(val, 0.5, abs_tol=0.001)


def test_fuse_rejects_weights_not_summing_to_one() -> None:
    """Misconfigured calibration weights are a silent-failure trap; reject loudly."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        fuse(0.5, 0.5, 0.5, 0.5, weights=(0.3, 0.3, 0.3, 0.3))


def test_fuse_output_clipped_to_unit_interval() -> None:
    """Even with weird inputs, the result stays in [0, 1] (downstream depends on this)."""
    val = fuse(1.0, 1.0, 1.0, 1.0)
    assert 0.0 <= val <= 1.0
    val = fuse(0.0, 0.0, 0.0, 0.0)
    assert 0.0 <= val <= 1.0
