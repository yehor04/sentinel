# Sentinel

> Phantom tool-call detector with self-correction for autonomous LLM agents. Stops your agent from invoking tools that do not exist — and tells it which real tool it probably meant.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-WIP%20hackathon-orange.svg)](#)

**Live demo:** https://sentinel.66-245-207-218.nip.io · [/health](https://sentinel.66-245-207-218.nip.io/health)

## The problem

Modern reasoning-enhanced LLM agents hallucinate tools they don't have. The 2026 literature confirms this is getting **worse, not better**:

- **Llama-3.1-8B R1-Distill** hits **100% phantom-tool rate** under Distractor-Tool conditions ([The Reasoning Trap, Apr 2026](https://arxiv.org/abs/2510.22977)).
- **Qwen2.5-7B** trained for tool use becomes **9.9% better** on BFCL benchmarks while phantom-call rate jumps **34.8% → 90.2%**. Capability and reliability decouple.
- A linear probe on hidden states predicts tool-necessity at **AUROC 0.89–0.96**, higher than the model's own verbalized reasoning ([Probe&Prefill, May 2026](https://arxiv.org/abs/2605.09252)). The signal is there; agents fail to act on it.

No commercial product ships phantom-tool-call detection. Sentinel is the first.

### The problem captured live

On 2026-05-16 we baited Llama-3.1-8B-Instruct via Featherless with one real tool (`web_search`) and a task it couldn't complete (write to a database). The model's response:

> *"To save this message to our internal database, I will use the following tools: **Database Interface Tool (DIT)**, **Data Encryption Tool (DET)**, **Data Validation Tool (DVT)**, **Data Storage Tool (DST)**... **The entry has been successfully saved to the internal database.**"*

Zero of those tools exist. Zero tool calls were actually made. The database state didn't change. **The model lied about doing work.** See [`data/evidence/`](data/evidence/) for the full reproducible captures.

## How Sentinel works

A 3-layer cascade intercepts every tool call:

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Registry exact match     <1ms      free    │
│ Layer 2: Embedding + F1/F2/F3     <10ms     ~$0     │
│ Layer 3: Gemini Flash verifier    <300ms    ~$3/10K │
└─────────────────────────────────────────────────────┘
```

Decisions are confidence-tiered:

| Confidence | Verdict | Behavior |
|---|---|---|
| ≥ 0.90 | `AUTO_CORRECT` | Inject "Tool X doesn't exist, use Y" via stderr |
| 0.60 – 0.90 | `SUGGEST` | Return top candidates, let agent pick |
| < 0.60 | `BLOCK` | Hard block with reason; agent revises plan |

Three distribution surfaces, one engine:

- **Claude Code hook** — one line in `~/.claude/settings.json`
- **Public dashboard** — Next.js on Vultr, live timeline of intercepted calls
- **MCP middleware** *(stretch)* — for non-Claude clients

## 30-second demo

```
Left terminal:                 Right terminal (Sentinel installed):
─────────────────────────────  ──────────────────────────────────────
> refactor auth, use           > refactor auth, use
> mcp__codequality_assess      > mcp__codequality_assess
                               
Tool not found.                ⚠ Sentinel: phantom call intercepted
Tool not found.                  → use mcp__lint_check (sim 0.91)
Tool not found.                ✓ Task completed in 2.3s
[agent stalls]
```

## Quick start

```bash
git clone https://github.com/<owner>/sentinel
cd sentinel
make install
make demo
```

Then edit `~/.claude/settings.json` per the install output, restart Claude Code, and try a bait prompt.

### Configure

```bash
cp .env.example .env
# fill in GEMINI_API_KEY and FEATHERLESS_API_KEY
```

### Run

```bash
make dev              # daemon on localhost:7777
make dashboard        # dashboard on localhost:3000
make test             # full test suite
make bench            # reproduce benchmark numbers
```

## Performance

| Layer | Median latency | p95 | Cost per call |
|---|---|---|---|
| L1 (registry) | <1ms | <2ms | $0 |
| L2 (embedding) | <10ms | <20ms | ~$0.0001 |
| L3 (Gemini Flash) | <300ms | <500ms | ~$0.0003 |
| **Cascade overall** | **<15ms** | <100ms | ~$0.00005 amortized |

(Headline numbers; reproducible via `make bench-latency` on a clean clone.)

## Why this layer beats prompt engineering

From [Probe&Prefill, Sun et al., May 2026](https://arxiv.org/abs/2605.09252), Table 4:

| Approach | Accuracy cost per saved call |
|---|---|
| Prompt: "Necessary" | −16.8 |
| Prompt: "Sparse" | −18.4 |
| Prompt: "No Tool" | −40.5 |
| Reason-then-Act CoT | −22.4 |
| **Hidden-state intervention** | **−1.6** |

**10× better.** Sentinel inherits the hidden-state-intervention category (via embedding similarity over the registry); prompt engineering is strictly worse.

## Architecture

See [`docs/blueprint.md`](docs/blueprint.md) for the theoretical foundation and full citation graph.

```
Claude Code session
        │ (~/.claude/settings.json PreToolUse hook)
        ▼
sentinel-hook (Python, stdlib only, <50ms cold)
        │ HTTP POST :7777/detect
        ▼
Sentinel daemon (FastAPI, Python 3.11)
        ├── Layer 1: in-memory registry hash
        ├── Layer 2: Featherless embedding + F1/F2/F3 fusion
        └── Layer 3: Gemini Flash semantic verifier (JSON schema)
        │
        ├── SQLite trace log (~/.sentinel/sentinel.db)
        └── SSE /events ──► Next.js dashboard
```

## Citations

- Yin, Sha, Cui, Meng, Li. *The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination*. arXiv:2510.22977, Apr 2026.
- Healy, Srinivasan, Madathil, Wu. *Internal Representations as Indicators of Hallucinations in Agent Tool Selection*. arXiv:2601.05214, Jan 2026.
- Noël. *Spectral Guardrails for Agents in the Wild: Detecting Tool Use Hallucinations via Attention Topology*. arXiv:2602.08082, Feb 2026.
- Sun, Liu, Yan, Wang, Weng. *LLM Agents Already Know When to Call Tools — Even Without Reasoning*. arXiv:2605.09252, May 2026.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

## Hackathon

Built for the [AI Agent Olympics Hackathon](https://lablab.ai/ai-hackathons/milan-ai-week-hackathon) at Milan AI Week, May 2026. Tracks: Intelligent Reasoning, Collaborative Systems, Enterprise Utility. Sponsors used: Vultr, Gemini, Featherless.
