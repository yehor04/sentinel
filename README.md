# Sentinel

> Phantom tool-call detector with self-correction for autonomous LLM agents. Stops your agent from invoking tools that don't exist — and tells it which real tool it probably meant.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)](#)
[![Status](https://img.shields.io/badge/status-live-brightgreen.svg)](#)

**Live demo:** https://sentinel.66-245-207-218.nip.io  
**Dashboard:** https://sentinel.66-245-207-218.nip.io (real-time intercept feed)  
**Health:** https://sentinel.66-245-207-218.nip.io/health

---

## The problem

Modern reasoning-enhanced LLM agents hallucinate tools they don't have. The 2026 literature confirms this is getting **worse, not better**:

- **Llama-3.1-8B R1-Distill** hits **100% phantom-tool rate** under Distractor-Tool conditions ([The Reasoning Trap, Apr 2026](https://arxiv.org/abs/2510.22977))
- **Qwen2.5-7B** jumps from **34.8% → 90.2%** phantom rate after tool-use RL training — capability and reliability decouple
- A linear probe on hidden states detects tool-call hallucinations at **AUROC 0.89–0.96**, higher than the model's own verbalized reasoning ([Sun et al., May 2026](https://arxiv.org/abs/2605.09252))

No commercial product shipped phantom-tool-call detection before Sentinel.

### Captured live

On 2026-05-16 we baited Llama-3.1-8B-Instruct via Featherless with one real tool (`web_search`) and a task it couldn't complete. The model's response:

> *"To save this message to our internal database, I will use: **Database Interface Tool**, **Data Encryption Tool**, **Data Storage Tool**... The entry has been **successfully saved**."*

Zero of those tools exist. Zero calls were made. The database didn't change. **The model lied about doing work.** Reproducible captures in [`data/evidence/`](data/evidence/).

---

## How it works

A 3-layer cascade intercepts every tool call before execution:

```
Agent tool call
      │
      ▼
┌─────────────────────────────────────────────┐
│ Layer 1  Registry exact match   <1ms  free  │  HIT → ALLOW
│ Layer 2  Embedding similarity  <1ms cached  │  ≥0.85 → AUTO_CORRECT
│          + F1/F2/F3 fusion                  │  0.60–0.85 → SUGGEST (→L3)
│ Layer 3  Gemini Flash verifier   ~2s  rare  │  <0.60 → BLOCK
└─────────────────────────────────────────────┘
```

| Verdict | Confidence | Effect |
|---|---|---|
| `ALLOW` | 1.0 (L1 hit) | Pass through unchanged |
| `AUTO_CORRECT` | ≥ 0.85 | Inject "use X instead" into agent's next turn |
| `SUGGEST` | 0.60–0.85 | Surface top-3 candidates |
| `BLOCK` | < 0.60 | Hard reject; agent must revise plan |

---

## Quick start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker (optional, for deploy path).

```bash
git clone https://github.com/yehor04/sentinel
cd sentinel
cp .env.example .env        # then set GEMINI_API_KEY in .env
make install                # installs hook to ~/.local/bin/sentinel-hook
make dev                    # daemon on localhost:7777
```

Add to `~/.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "*",
      "hooks": [{"type": "command", "command": "~/.local/bin/sentinel-hook"}]
    }]
  }
}
```

Or hit the live public endpoint directly — no setup needed:
```bash
curl -X POST https://sentinel.66-245-207-218.nip.io/detect \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "search_the_internet", "tool_input": {"query": "test"}}'
```

### Docker deploy

```bash
cp .env.example .env        # set GEMINI_API_KEY and SENTINEL_DOMAIN
cd deploy && docker compose up -d
```

All thresholds live in [`configs/cascade.yaml`](configs/cascade.yaml) — no magic numbers in source.

### Run

```bash
make dev            # daemon on localhost:7777
make test           # 140 tests
make bench-latency  # latency gate (p50=0.132ms, gate=10ms)
make bench          # full corpus bench (F1=1.0 on 55 examples)
```

---

## Performance

Measured on Vultr Milan (2 vCPU / 8 GB), production deployment:

| Path | p50 | p95 | Cost |
|---|---|---|---|
| L1 exact match | 0.002 ms | 0.01 ms | $0 |
| L1 + L2 (cache hit) | **0.132 ms** | 0.381 ms | ~$0 |
| L1 + L2 + L3 | ~2.3 s | ~4 s | ~$0.0003 |

**Latency gate:** `make bench-latency` — 1,000 calls, p50 = 0.132ms (75× under the 10ms budget).

L3 (Gemini Flash) fires on fewer than 15% of calls — only when L2 lands in the ambiguous 0.60–0.85 confidence window.

---

## Dashboard

Live at **https://sentinel.66-245-207-218.nip.io**

- Real-time SSE feed of every intercept
- Color-coded verdict badges (ALLOW / AUTO_CORRECT / SUGGEST / BLOCK)
- Stats bar: total calls, phantoms caught, auto-corrected, avg confidence
- Per-row: tool name → correction, confidence %, L2/L3 latency, degraded flag

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full cascade diagram and engineering decisions.

```
Claude Code session
        │  PreToolUse hook (~/.claude/settings.json)
        ▼
sentinel-hook.py  (stdlib-only, <50ms cold start)
        │  POST /detect
        ▼
Sentinel daemon  (FastAPI + Uvicorn, Python 3.11)
        ├── Layer 1: in-memory registry hash (O(1), case-fold)
        ├── Layer 2: Gemini text-embedding-001 + F1/F2/F3 fusion
        │           on-disk LRU cache (diskcache, /var/sentinel/)
        └── Layer 3: Gemini 2.5 Flash (httpx REST, thinkingBudget=0)
                └── H1: suggestion must be in registry (anti-injection)
                └── H2: ALLOW verdict rejected (L1's exclusive domain)
        │
        ├── GET /history   → last 200 decisions + stats
        ├── GET /events    → SSE stream (live dashboard)
        └── GET /          → dashboard (HTML, dark theme)

Deploy: Vultr Milan · Caddy auto-TLS · Docker (non-root sentinel uid 10001)
```

---

## Security

- API key passed via `x-goog-api-key` header (never in URL query params → never in logs)
- All user-derived fields HTML-escaped before dashboard `innerHTML`
- SSE subscriber cap (50 max) prevents connection exhaustion
- Request body limited to 64KB (Caddyfile)
- Security headers: CSP, X-Frame-Options DENY, X-Content-Type-Options, Referrer-Policy
- H1 guard: L3 suggestions validated against registry (prevents phantom correction)
- H2 guard: ALLOW from L3 treated as injection attempt, cascade degrades to L2

---

## Benchmark corpus

[`data/sentinel-bench-v1/`](data/sentinel-bench-v1/) — 55 hand-crafted examples:
- 30 phantom calls (AUTO_CORRECT / SUGGEST / BLOCK expected)
- 25 legal calls (ALLOW expected)
- 20-tool registry matching the Claude Code + MCP surface

Phantom detection with stub embedder: **F1 = 1.000** (TP=30, FP=0, FN=0).

---

## Citations

1. Yin et al. *The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination*. arXiv:2510.22977, Apr 2026.
2. Healy et al. *Internal Representations as Indicators of Hallucinations in Agent Tool Selection*. arXiv:2601.05214, Jan 2026.
3. Noël. *Spectral Guardrails for Agents in the Wild*. arXiv:2602.08082, Feb 2026.
4. Sun et al. *LLM Agents Already Know When to Call Tools*. arXiv:2605.09252, May 2026.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

---

Built for the **AI Agent Olympics Hackathon** at Milan AI Week 2026.  
Tracks: Intelligent Reasoning · Collaborative Systems · Enterprise Utility.
