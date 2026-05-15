# CLAUDE.md — Sentinel Project Governance

> This file is the operational contract between you (Claude / any future AI agent) and the Sentinel codebase. Read it at the start of every session. The constitution at `.specify/memory/constitution.md` is the authoritative principles document; this file is the day-to-day operating manual.

## WHY: The problem we are solving

Modern reasoning-enhanced LLM agents (Claude Sonnet 4.6, GPT-5, Qwen3-Think, R1-distilled models) suffer from **phantom tool calls** — invoking functions that do not exist in their registry. The 2026 literature documents this as a fundamental, growing failure mode:

- **The Reasoning Trap** (Yin et al., Penn State / Nanjing UST, 17 Apr 2026, arXiv:2510.22977): Llama-3.1-8B R1-Distill hits **100% Distractor-Tool hallucination rate**. Think-then-act RL takes Qwen2.5-7B from 34.8% to 90.2% phantom rate.
- **LLM Agents Already Know When to Call Tools** (Sun et al., UCSD / AWS, 10 May 2026, arXiv:2605.09252): Hidden states encode tool-necessity with AUROC 0.89–0.96 — *higher than the model's own verbalized reasoning*. The signal exists; agents fail to act on it.
- **Spectral Guardrails for Agents** (Noël, Devoteam, 8 Feb 2026, arXiv:2602.08082): Training-free spectral detection achieves 98.2% recall on Llama-3.1-8B hallucinated tool calls using a single Smoothness feature.
- **Internal Representations** (Healy et al., Amazon, 8 Jan 2026, arXiv:2601.05214): Last-layer feature probing detects tool-call hallucinations at F1 0.85, single forward pass.

No commercial product ships phantom-tool-call detection today. Sentinel is the first.

## WHAT: The product

Sentinel is a three-layer cascade that wraps any autonomous LLM agent and intercepts tool-call attempts before execution:

1. **Layer 1 — Registry exact match** (in-memory hash, <1ms median)
2. **Layer 2 — Embedding similarity** (Featherless-hosted small model, <10ms median)
3. **Layer 3 — Gemini Flash semantic verifier** (~300ms median, fires on <15% of calls)

Decisions are confidence-tiered:

| Confidence | Verdict | Behavior |
|---|---|---|
| ≥ 0.90 | `AUTO_CORRECT` | Inject "Tool X doesn't exist, use Y" into agent's next turn |
| 0.60 – 0.90 | `SUGGEST` | Return top-3 candidates, let agent pick |
| < 0.60 | `BLOCK` | Hard block with reason; agent must revise plan |
| 1.00 (L1 match) | `ALLOW` | Tool passes through unmodified |

Three distribution surfaces, one engine:

- **Claude Code hook** (`PreToolUse` in `~/.claude/settings.json`) — primary demo path
- **Web dashboard** (Next.js on Vultr) — hackathon-required public demo URL
- **MCP middleware** (stretch goal, Day 5) — for non-Claude clients

## HOW: The locked 3-layer cascade

```
Tool call arrives at hook
        │
        ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Registry exact match (<1ms, free)  │
│   - in-memory dict lookup                   │
│   - YES → ALLOW (conf 1.0)                  │
│   - NO  → fall through                      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 2: Featherless embedding (~10ms)      │
│   - embed phantom name + inferred purpose   │
│   - cosine vs precomputed registry vectors  │
│   - apply F1 (Levenshtein), F2 (Jaccard),   │
│     F3 (top-1 vs top-2 gap) fusion          │
│   - top1 ≥ 0.85 → AUTO_CORRECT              │
│   - 0.60-0.85   → escalate                  │
│   - < 0.60      → BLOCK                     │
└────────────────────┬────────────────────────┘
                     │ (only if 0.60 ≤ conf < 0.85)
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 3: Gemini Flash verifier (~300ms)     │
│   - structured prompt (see docs/blueprint)  │
│   - JSON output: verdict/confidence/reason  │
│   - integrity checks: schema, reasoning     │
└─────────────────────────────────────────────┘
                     │
                     ▼
        Decision → hook → stderr → agent retries
```

**Never deviate from this cascade.** Any new detection idea is *added as a feature input* to one of the existing layers, not a new layer. Adding Layer 4+ requires a constitution amendment.

## Stack & universal commands

**Language:** Python 3.11+ (single language end-to-end; no TypeScript on backend, no Go in v1)
**Backend:** FastAPI + Uvicorn (daemon), SQLite (persistence, no migrations)
**Frontend:** Next.js (dashboard only; no auth, no SSR auth)
**External APIs:** Gemini Flash (Layer 3 + embeddings), Featherless (small-model embeddings)
**Deploy:** Vultr 1-vCPU instance, `docker-compose up`, Caddy reverse proxy with auto-TLS
**Dependency manager:** `uv` (fast, lockfile-based)

### Build
```bash
make install           # uv sync, install hook script to ~/.local/bin
make dev               # run daemon on localhost:7777 with reload
make dashboard         # run Next.js dashboard on localhost:3000
make demo              # spin up daemon + dashboard + sample agent in tmux
```

### Test
```bash
make test              # pytest backend/tests/
make test-hook         # integration test: real Python subprocess + daemon
make test-cascade      # unit tests for all 3 layers
make test-contract     # contract tests for Decision schema
```

### Lint / format / typecheck
```bash
make lint              # ruff check
make format            # ruff format
make typecheck         # mypy strict on backend/sentinel/
make check             # lint + format-check + typecheck (CI gate)
```

### Benchmark
```bash
make bench             # full benchmark on SentinelBench-v1, writes results/<date>.json
make bench-latency     # latency-only sweep, fails if median L1+L2 > 10ms
make bench-pareto      # generates Pareto-frontier chart for demo video
```

### Deploy
```bash
make deploy-vultr      # ssh to Vultr, git pull, docker-compose up -d
make smoke-vultr       # curl public URL, assert /health and /detect respond
```

## Operating rules for AI assistants (you)

1. **Constitution wins.** If a request conflicts with `.specify/memory/constitution.md`, surface the conflict and refuse the unsafe change.
2. **Demo-first.** Before implementing any feature, ask: does this raise detection accuracy, reduce latency, or harden the demo? If no, defer to v2.
3. **One language.** Do not introduce TypeScript on the backend, do not introduce Python on the frontend. The hook is Python; the daemon is Python; the dashboard is TypeScript only because Next.js demands it.
4. **No silent corrections.** Every decision must populate `verdict`, `confidence`, `reason`. Empty fields fail the contract test.
5. **No new layers.** The cascade is locked at 3 layers. New detection ideas are F-features fused into Layer 2 or signals consumed by Layer 3.
6. **Reproducibility.** Every threshold lives in `configs/`. Every benchmark result writes to `results/`. Every demo bait prompt lives in `data/bait-corpus/`.
7. **No backwards-compat hacks.** This is a v1 project; if a refactor is cleaner, do it. Do not leave deprecated paths or `// removed` comments.
8. **Trust the schema.** `Decision`, `ToolRegistry`, and `DetectRequest` are the only public types. All internal modules consume/produce these.
9. **Default to terse.** No multi-paragraph docstrings. One-line comments only when WHY is non-obvious.
10. **Read the blueprint.** When in doubt about why a heuristic is named F1/F2/F3 or why a threshold is 0.85, see `docs/blueprint.md`.

## Sprint state (current)

**Hackathon:** AI Agent Olympics, Milan AI Week 2026
**Window:** 2026-05-13 → 2026-05-20 (7 days; today is Day 2)
**Submission deadline:** 2026-05-20
**Mode:** Solo developer
**Stack lock-in date:** 2026-05-14
**Constitution version:** 1.0.0

## Glossary

- **Phantom tool call:** A `tool_use` invocation by an LLM agent where `tool_name` is not present in the active tool registry.
- **Cascade:** The 3-layer detection pipeline (L1 registry / L2 embedding / L3 verifier).
- **F1 / F2 / F3:** Three fusion heuristics added to Layer 2 confidence — Levenshtein distance, schema-key Jaccard, top-1-vs-top-2 gap. See `docs/blueprint.md`.
- **Bait corpus:** Hand-crafted prompts that reliably induce phantom tool calls in current LLMs. Lives in `data/bait-corpus/`.
- **SentinelBench-v1:** Our 1,900-example evaluation set combining SimpleToolHalluBench protocol, When2Tool labels, and Glaive Function Calling v2 samples.
- **Reasoning Trap:** The 2026 paper finding that reasoning-enhanced LLMs hallucinate tools more frequently than their base versions.
