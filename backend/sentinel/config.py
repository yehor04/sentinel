"""Cascade configuration loader.

Single source of truth for every numeric threshold + fusion weight. Reads
`configs/cascade.yaml` exactly once per process (LRU cache) and validates the
loaded structure via pydantic so a typo in the YAML fails loudly at startup
rather than silently producing wrong decisions.

Constitution Principle V (Reproducibility Over Cleverness): no floats in code;
everything threshold-shaped lives here.

Usage:

    from sentinel.config import load_cascade_config
    cfg = load_cascade_config()
    if confidence >= cfg.verdict_thresholds.auto_correct_min:
        ...
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class VerdictThresholds(BaseModel):
    """Decision-tier cutoffs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    auto_correct_min: float = Field(ge=0.0, le=1.0, default=0.85)
    block_max: float = Field(ge=0.0, le=1.0, default=0.60)

    @model_validator(mode="after")
    def _check_ordering(self) -> "VerdictThresholds":
        if self.block_max >= self.auto_correct_min:
            raise ValueError(
                f"block_max ({self.block_max}) must be < auto_correct_min ({self.auto_correct_min})"
            )
        return self


class FusionWeights(BaseModel):
    """Layer-2 confidence fusion weights + F3 scaling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_weight: float = Field(ge=0.0, le=1.0, default=0.5)
    f1_weight: float = Field(ge=0.0, le=1.0, default=0.2)
    f2_weight: float = Field(ge=0.0, le=1.0, default=0.2)
    f3_weight: float = Field(ge=0.0, le=1.0, default=0.1)
    f3_multiplier: float = Field(gt=0.0, default=5.0)

    @model_validator(mode="after")
    def _check_weights_sum(self) -> "FusionWeights":
        total = self.base_weight + self.f1_weight + self.f2_weight + self.f3_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0 (got {total:.6f}). "
                "Edit configs/cascade.yaml and rerun."
            )
        return self

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Tuple form for `heuristics.fuse(weights=...)`."""
        return (self.base_weight, self.f1_weight, self.f2_weight, self.f3_weight)


class LatencyBudget(BaseModel):
    """Per-layer wall-clock budgets enforced by `make bench-latency`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    l1_p95_ms: float = Field(gt=0.0, default=2.0)
    l2_p95_ms: float = Field(gt=0.0, default=20.0)
    l3_p95_ms: float = Field(gt=0.0, default=500.0)


class EmbeddingConfig(BaseModel):
    """Layer-2 embedding backend selection + cache + timeouts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["featherless", "gemini", "stub"] = "featherless"
    featherless_model: str = "BAAI/bge-small-en-v1.5"
    gemini_model: str = "text-embedding-004"
    cache_dir: str = "~/.sentinel/embed-cache"
    cache_size_mb: int = Field(gt=0, default=512)
    timeout_s: float = Field(gt=0.0, default=5.0)


class VerifierConfig(BaseModel):
    """Layer-3 semantic verifier — Gemini Flash by default, stub for tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["gemini", "stub"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    timeout_s: float = Field(gt=0.0, default=2.0)
    max_output_tokens: int = Field(gt=0, default=512)


class CascadeConfig(BaseModel):
    """Top-level cascade config; mirrors configs/cascade.yaml structure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict_thresholds: VerdictThresholds = Field(default_factory=VerdictThresholds)
    fusion: FusionWeights = Field(default_factory=FusionWeights)
    latency: LatencyBudget = Field(default_factory=LatencyBudget)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def _candidate_paths() -> list[Path]:
    """Search order for cascade.yaml — first hit wins."""
    paths: list[Path] = []
    env_override = os.environ.get("SENTINEL_CASCADE_CONFIG")
    if env_override:
        paths.append(Path(env_override))
    # repo-relative (works from any cwd inside the project)
    here = Path(__file__).resolve()
    paths.append(here.parent.parent.parent / "configs" / "cascade.yaml")
    # production deploy path
    paths.append(Path("/etc/sentinel/cascade.yaml"))
    return paths


@lru_cache(maxsize=1)
def load_cascade_config() -> CascadeConfig:
    """Load `configs/cascade.yaml`, validate, cache for the process lifetime.

    Returns sensible defaults (constitutional baseline) if no file is found.
    Raises pydantic ValidationError if a found file is structurally invalid —
    a typo in the YAML should fail loud at startup, not silently produce wrong
    decisions later.
    """
    for path in _candidate_paths():
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return CascadeConfig.model_validate(data)
    return CascadeConfig()


def clear_cache() -> None:
    """Clear the cached config — for tests that mutate the YAML mid-run."""
    load_cascade_config.cache_clear()
