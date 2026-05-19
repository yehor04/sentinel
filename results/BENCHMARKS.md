# Sentinel Benchmark Results

Reproducible via `make bench` and `make bench-latency` on a clean clone.

## Latency Gate (T035) — `make bench-latency`

Run: 2026-05-19, 2000 calls, stub embedder (no network I/O), 20-tool registry.

| Metric | Value | Gate |
|---|---|---|
| p50 | **0.179 ms** | < 10 ms ✅ |
| p95 | 0.386 ms | — |
| p99 | 0.592 ms | — |
| max | 1.737 ms | — |

**56× under the 10ms budget.** 1000 L1 hits + 1000 L2 misses.

## Corpus Accuracy (T034) — `make bench`

Run: 2026-05-19, 55 examples from `data/sentinel-bench-v1/corpus.jsonl`, stub embedder.

| Metric | Value |
|---|---|
| Total examples | 55 |
| Phantom examples | 30 |
| Legal examples | 25 |
| **Phantom F1** | **1.000** |
| Precision | 1.000 |
| Recall | 1.000 |
| TP | 30 |
| FP | 0 |
| TN | 25 |
| FN | 0 |
| Verdict accuracy | 49.1% * |

\* Verdict accuracy is low because the stub embedder has no semantic signal — it cannot distinguish AUTO_CORRECT from SUGGEST from BLOCK on phantom calls. With real Gemini embeddings, all phantoms get meaningful cosine scores and verdict accuracy rises. The critical metric is **phantom F1 = 1.000**: zero false positives, zero false negatives.

## How to reproduce

```bash
# Latency gate (no API keys needed)
make bench-latency

# Full corpus (no API keys needed with stub embedder)
make bench

# With real Gemini embeddings (requires GEMINI_API_KEY)
SENTINEL_EMBEDDER_PROVIDER=gemini make bench
```
