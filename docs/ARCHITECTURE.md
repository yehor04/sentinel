# Sentinel — Architecture

## Problem

Reasoning-enhanced LLM agents fabricate tool names that don't exist in their registry (**phantom tool calls**). Published 2026 measurements:

- Llama-3.1-8B-Instruct: **62.5% phantom rate** on natural tasks, **99.7%** under distractor tools *(Reasoning Trap, arXiv:2510.22977)*
- Hidden states encode tool-necessity at AUROC 0.89–0.96 — *higher than the model's own verbalized confidence* *(Sun et al., arXiv:2605.09252)*
- Spectral detection achieves 98.2% recall with a single feature *(Noël, arXiv:2602.08082)*

No commercial product shipped phantom detection before Sentinel.

## Three-Layer Cascade

```
Agent tool call arrives
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 1 · Registry exact match       │  < 1 ms · free
│  In-memory hash lookup (case-fold)    │
│  HIT  → ALLOW (conf 1.0), done        │
│  MISS → fall through                  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Layer 2 · Embedding similarity       │  0.3 ms cached
│  Gemini text-embedding-001 (768-dim)  │
│  Cosine vs precomputed registry vecs  │
│  F1 Levenshtein · F2 Jaccard · F3 gap │
│  ≥ 0.85 → AUTO_CORRECT               │
│  0.60–0.85 → SUGGEST (→ L3)          │
│  < 0.60  → BLOCK                     │
└──────────────────┬────────────────────┘
                   │  only on SUGGEST
                   ▼
┌───────────────────────────────────────┐
│  Layer 3 · Gemini Flash verifier      │  ~2 s · fires < 15% of calls
│  Structured JSON prompt               │
│  Schema-validated VerifierResponse    │
│  Fused: 0.6×L3 + 0.4×L2 confidence   │
│  H1: suggestion must be in registry   │
│  H2: ALLOW verdict rejected (guard)   │
└──────────────────┬────────────────────┘
                   │
                   ▼
              Decision
    verdict · confidence · reason
    suggestion · layer_breakdown
    degraded · ghost_claims
```

## Verdict tiers

| Confidence | Verdict | Agent effect |
|---|---|---|
| 1.0 (L1 hit) | `ALLOW` | Pass through |
| ≥ 0.85 | `AUTO_CORRECT` | Inject "use X instead" into next turn |
| 0.60 – 0.85 | `SUGGEST` | Surface top-3 candidates, agent picks |
| < 0.60 | `BLOCK` | Hard reject, agent must revise plan |

## Fusion heuristics (Layer 2)

Applied when base cosine confidence lands in the ambiguous `[0.60, 0.85)` window:

- **F1** — Levenshtein name distance (catches `execute_command` → `Bash`)
- **F2** — Jaccard overlap of argument key sets (catches schema-twins)
- **F3** — Top-1 vs top-2 cosine gap (penalises ambiguous candidates)

Weights: `base×0.5 + F1×0.2 + F2×0.2 + F3×0.1` — all configurable in `configs/cascade.yaml`.

## Latency budget

| Layer | Median | p95 | Gate |
|---|---|---|---|
| L1 | 0.002 ms | 0.01 ms | < 2 ms |
| L2 (cache hit) | 0.31 ms | 0.5 ms | < 20 ms |
| L1 + L2 combined | **0.13 ms** | 0.38 ms | **< 10 ms** ✅ |
| L3 (Gemini Flash) | ~2 s | ~4 s | fires < 15% |

Benchmark: `make bench-latency` — 1,000 calls, p50 = 0.132 ms (75× under gate).

## Distribution surfaces

| Surface | Status |
|---|---|
| Claude Code `PreToolUse` hook | ✅ Live |
| REST API `/detect` (any agent) | ✅ Live at `https://sentinel.66-245-207-218.nip.io` |
| Live dashboard `/` | ✅ Real-time SSE feed |
| MCP middleware | Stretch goal (Day 5) |

## Stack

- **Language:** Python 3.11 end-to-end
- **API:** FastAPI + Uvicorn on Vultr Milan (2 vCPU / 8 GB)
- **Embeddings:** `gemini-embedding-001` via `google-generativeai` SDK
- **Verifier:** `gemini-2.5-flash` via direct `httpx` REST (no SDK overhead)
- **Proxy:** Caddy auto-TLS (Let's Encrypt)
- **Config:** all thresholds in `configs/cascade.yaml` — no magic numbers in source

## Key engineering decisions

**Why 3 layers?** Each layer trades latency for accuracy. L1 is free and handles the majority of traffic. L2 handles obvious semantic mismatches. L3 fires only on ambiguous cases, spending Gemini quota where it matters most.

**Why `httpx` for L3 instead of the SDK?** The `google.generativeai` SDK is deprecated and causes server-side 504 timeouts on newer Flash models via the Vultr Milan node. Direct REST calls are reliable and give full timeout control.

**Why `gemini-embedding-001` over `gemini-embedding-2`?** Measured on our phantom corpus: `-001` achieves cosine 0.86 vs 0.84, at 267 ms vs 1614 ms (6× faster). Newer ≠ better for short tool-name signatures.
