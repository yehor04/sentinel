# Implementation Plan: Sentinel MVP

**Branch**: `001-sentinel-mvp` | **Date**: 2026-05-14 | **Spec**: `./spec.md`

## Summary

Sentinel is a phantom tool-call detector + self-correction layer for autonomous LLM agents. The MVP delivers a 3-layer cascade (registry exact match / Featherless embedding similarity / Gemini Flash semantic verifier), distributed as a Claude Code `PreToolUse` hook plus a FastAPI daemon plus a Next.js dashboard, deployed to a single Vultr VM. The MVP must support a 30-second split-screen demo where a baited Claude Code session is rescued from a phantom call by Sentinel injecting the correct tool name via stderr.

## Technical Context

**Language/Version**: Python 3.11+ (backend + hook), TypeScript 5.x (dashboard only)

**Primary Dependencies**:
- Backend: FastAPI, Uvicorn, pydantic v2, httpx, structlog, sqlite-utils, sentence-transformers (for offline embed fallback), rapidfuzz (Levenshtein)
- API SDKs: `google-generativeai` (Gemini Flash + embeddings), `openai`-compatible client for Featherless
- Frontend: Next.js 15, Tailwind v4, shadcn/ui, recharts (Pareto chart)
- Hook: Python 3.11 stdlib only — `sys`, `json`, `urllib.request`, `os` — NO third-party imports

**Storage**: SQLite (`~/.sentinel/sentinel.db` local, `/var/sentinel/sentinel.db` Vultr). No migrations; schema is created at daemon start if absent.

**Testing**: pytest (backend), pytest-asyncio for FastAPI, Playwright (smoke test on Vultr URL stretch).

**Target Platform**: macOS dev / Ubuntu 24.04 Vultr 1-vCPU 2GB RAM. Single VM.

**Project Type**: Web-service-backed library (Option 2 in template).

**Performance Goals**:
- Layer 1 median <1ms, p95 <2ms
- Layer 2 median <10ms, p95 <20ms
- Layer 3 median <300ms, p95 <500ms
- Cascade overall median <15ms (weighted 85% L1+L2 / 15% L3)
- Hook subprocess cold-start <50ms
- Dashboard initial paint <2s on European mobile

**Constraints**: Solo developer, 6-day build window, no GPU, ≤2GB RAM on production VM, $25 Featherless + $300 Google Cloud + Vultr-provided credits as total compute budget for build week + submission video.

**Scale/Scope**: One demo session (~50 tool calls) + benchmark run (~1,900 examples × 3 seeds = 5,700 detections). No multi-tenant concerns in v1.

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Library-First | ✅ | All logic in `backend/sentinel/`; daemon + hook are thin clients |
| II. 10ms L1+L2 (NON-NEGOTIABLE) | ⚠️ Pending | Verified Day 3 via `make bench-latency`; design respects budget |
| III. Confidence-Gated Self-Correction (NON-NEGOTIABLE) | ✅ | `Decision` schema enforces verdict + confidence + reason |
| IV. Demo-First Engineering | ✅ | Day 1 ships happy path; every subsequent feature serves demo |
| V. Reproducibility Over Cleverness | ✅ | Thresholds in `configs/`; benchmark writes commit SHA + JSON |

## Project Structure

### Documentation (this feature)

```text
specs/001-sentinel-mvp/
├── plan.md              # This file
├── spec.md              # Feature spec
├── tasks.md             # 6-day task breakdown
├── research.md          # (skip — research lives in docs/blueprint.md)
├── data-model.md        # Entities defined inline below
├── quickstart.md        # `make demo` workflow
└── contracts/
    ├── detect_request.schema.json
    └── decision.schema.json
```

### Source Code (repository root)

```text
sentinel/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint, /detect /events /health
│   │   ├── routes/
│   │   │   ├── detect.py        # POST /detect adapter over sentinel.detect()
│   │   │   ├── events.py        # SSE /events
│   │   │   └── health.py
│   │   └── db.py                # SQLite init + persistence helpers
│   ├── sentinel/                # the actual library (Library-First)
│   │   ├── __init__.py          # public API: detect(), Decision, etc.
│   │   ├── schemas.py           # pydantic models (DetectRequest, Decision, ToolRegistry, Tool, TraceEvent)
│   │   ├── cascade.py           # orchestrator: L1 → L2 → L3
│   │   ├── layer1.py            # registry exact match
│   │   ├── layer2.py            # embedding sim + F1/F2/F3 fusion
│   │   ├── layer3.py            # Gemini Flash verifier
│   │   ├── embeddings.py        # Featherless client + LRU cache
│   │   ├── registry.py          # ToolRegistry loader (static yaml / MCP / Claude Code)
│   │   └── heuristics.py        # F1 (Levenshtein), F2 (Jaccard), F3 (top-1 gap)
│   ├── tests/
│   │   ├── contract/
│   │   │   ├── test_decision_schema.py
│   │   │   └── test_detect_request_schema.py
│   │   ├── unit/
│   │   │   ├── test_layer1.py
│   │   │   ├── test_layer2.py
│   │   │   ├── test_layer3.py
│   │   │   └── test_heuristics.py
│   │   └── integration/
│   │       ├── test_cascade_end_to_end.py
│   │       └── test_hook_subprocess.py
│   ├── bench/
│   │   ├── run_bench.py         # entrypoint for `make bench`
│   │   ├── sentinelbench_v1.py  # dataset loader
│   │   ├── calibrate.py         # threshold grid search
│   │   └── pareto.py            # chart renderer
│   ├── pyproject.toml
│   └── uv.lock
├── sentinel-hook.py             # 50-line zero-dep hook script (installed to ~/.local/bin)
├── frontend/                    # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx             # live timeline + benchmark tab
│   │   └── layout.tsx
│   ├── components/
│   │   ├── live-timeline.tsx
│   │   └── pareto-chart.tsx
│   ├── lib/
│   │   └── sse.ts
│   ├── package.json
│   └── next.config.ts
├── deploy/
│   ├── docker-compose.yml       # daemon + dashboard + caddy
│   ├── Caddyfile
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── deploy-vultr.sh
├── configs/
│   ├── cascade.yaml             # thresholds: 0.85 auto-correct, 0.60 block, fusion weights
│   ├── registry.yaml            # demo tool registry (sample, ~20 tools)
│   └── models.yaml              # Gemini Flash model name, Featherless model name
├── data/
│   ├── bait-corpus/             # hand-crafted phantom-trigger prompts (held out from eval)
│   └── sentinel-bench-v1/       # NTA + DT + Glaive subset + labels
├── results/                     # benchmark outputs (gitignored except .gitkeep)
├── docs/
│   ├── blueprint.md             # theoretical foundation
│   ├── ARCHITECTURE.md          # diagrams
│   └── DEMO_SCRIPT.md           # 30-second video shot list
├── CLAUDE.md                    # AI assistant governance
├── README.md                    # public-facing
├── LICENSE                      # Apache 2.0
├── Makefile                     # `make demo`, `make bench`, etc.
├── .gitignore
└── .env.example
```

**Structure Decision**: Web application (Option 2 from template) — `backend/` + `frontend/` because the hackathon requires a public dashboard URL. The hook script lives at repository root for trivial install via `cp sentinel-hook.py ~/.local/bin/sentinel-hook`. The Python library is `backend/sentinel/` so it can be imported standalone (Library-First principle).

## Phase 0 (Day 1) — Pipeline & End-to-End Happy Path

Goal: prove every leg of the architecture works before writing any detection logic.

1. **Vultr "Hello World"** — single FastAPI endpoint deployed, public URL responds with 200, TLS via Caddy. Pre-empts deploy debugging on Day 5.
2. **Hook subprocess loop** — `sentinel-hook.py` reads stdin JSON, POSTs to `localhost:7777/detect`, parses mocked response, returns exit code accordingly.
3. **Daemon `/detect` mock** — returns a fixed `Decision` (no real detection) so the hook + agent loop can be tested.
4. **Claude Code integration test** — hook configured in dev `settings.json`, invoke a tool call manually, confirm hook fires and stderr message appears.
5. **One bait prompt verified** — confirm we can reliably induce a phantom tool call in Claude Sonnet 4.6 with a hand-crafted prompt. If no, fall back to replay strategy from recorded traces.

## Phase 1 (Days 2–3) — Cascade Implementation

1. **`schemas.py`** — DetectRequest, Decision, ToolRegistry, Tool, TraceEvent (pydantic v2). Contract tests written first (TDD waiver only here, given the constitution favors demo-first).
2. **`layer1.py`** — registry exact match. Trivial; covered by 2 unit tests.
3. **`heuristics.py`** — F1 (rapidfuzz Levenshtein), F2 (set Jaccard), F3 (top-1 vs top-2 gap × 5 clipped). Each ≤15 lines.
4. **`embeddings.py`** — Featherless OpenAI-compatible client; on-disk LRU cache keyed by `(model, text)` hash; warm-up pass at daemon start to precompute registry embeddings.
5. **`layer2.py`** — embed phantom name + inferred purpose; cosine vs precomputed registry vectors; fuse with F1/F2/F3 when 0.60 ≤ base ≤ 0.85.
6. **`layer3.py`** — Gemini Flash with structured-output (JSON schema) prompt from `docs/blueprint.md` §4. Fires only when L2 conf is ambiguous.
7. **`cascade.py`** — orchestrator that wires L1 → L2 → L3 and produces the final `Decision`. Returns `layer_breakdown` with per-layer latency.

## Phase 2 (Day 3 PM → Day 4) — Dashboard + Benchmark

1. **`/events` SSE** — daemon emits each `Decision` to subscribers; SQLite persistence is fire-and-forget.
2. **Dashboard live timeline** — Next.js page that subscribes to SSE and renders a Tailwind-styled row per decision with color-coded verdicts.
3. **Dashboard Pareto tab** — static PNG from `results/latest-pareto.png` rendered with `recharts` from the JSON benchmark output.
4. **SentinelBench-v1 dataset** — assemble 1,900 labeled examples (NTA 296×2 + Glaive 1000 + bait corpus 200 + Claude Code traces ~100 if available).
5. **`run_bench.py`** — 60/20/20 stratified split (Healy IR protocol), runs cascade on test split, computes accuracy/F1/recall/latency/cost, writes `results/<date>-<sha>.json`.
6. **`calibrate.py`** — grid search threshold τ over [0.1, 0.9] step 0.05 on val split; writes `configs/cascade.yaml` with selected thresholds + 3-seed mean ± std.
7. **`pareto.py`** — renders 3-axis chart (latency × recall, cost as color) as PNG to `results/<date>-pareto.png`.

## Phase 3 (Day 5) — Polish + Optional MCP Middleware

1. **Edge cases** — daemon-down failover, NaN embeddings, Layer 3 timeout, concurrent invocations stress test (50 RPS for 60s).
2. **Demo bait freeze** — 2 bait prompts × 3 model targets × 10 runs each = 60 sanity tests. Promote to canonical demo set if ≥80% trigger rate.
3. **Stretch: MCP middleware** — `mcp-proxy` mode that intercepts `tools/call` JSON-RPC between client and downstream MCP server. Only ship if Phase 1+2 land clean.
4. **README rewrite** — final marketing copy, install instructions, screenshots from dashboard.

## Phase 4 (Day 6) — Submit

1. **Demo video** — 30 seconds, split-screen, narration optional. Two takes minimum.
2. **Vultr final deploy + smoke test** — `make smoke-vultr` passes.
3. **GitHub repo public** — Apache 2.0, archive of frozen submission tag.
4. **Submission form** — public URL, GitHub link, demo video, architecture diagram, written summary.
5. **Buffer** — last 6 hours reserved for "everything broke" recovery.

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Bait prompts don't reliably trigger phantom calls | M | High | Day 1 PM verification gate; fallback to replay-from-trace if fail |
| Featherless embedding API rate-limited at demo time | L | High | Pre-warm embeddings at daemon boot; cached locally; can run with Gemini embedding fallback |
| Gemini Flash returns malformed JSON | L | Med | Strict pydantic validation; on parse fail, downgrade to L2 decision with `degraded=true` |
| Vultr VM unreachable at demo time | L | Critical | Pre-record dashboard view as MP4 fallback in demo video |
| Demo machine has stale Claude Code version | L | Med | Pin Claude Code version in DEMO_SCRIPT.md; test on Day 5 |
| Latency budget blown by network jitter on Vultr | M | Med | Demo video uses local daemon; Vultr serves dashboard only |
| Submission deadline misjudged | L | Critical | All deadlines computed in CET; final submit 8h before cutoff |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Two languages (Python + TypeScript) | Next.js mandates TS for dashboard | Pure-Python dashboard would require Streamlit or similar; loses SSE polish and shadcn/ui aesthetics; jury reads quality signal from a real frontend |
| SQLite persistence on daemon | Dashboard needs a history beyond in-memory ring buffer; benchmark also persists | In-memory only loses data on crash, and jury hitting `/traces` after a restart should still see history |
| Caddy reverse proxy on Vultr | Need TLS for the public URL; Caddy auto-provisions Let's Encrypt | Self-managing certbot is brittle on demo day; nginx requires manual TLS config |
