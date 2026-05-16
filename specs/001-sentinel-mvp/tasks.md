---
description: "6-day build sprint task list for Sentinel MVP"
---

# Tasks: Sentinel MVP — 6-Day Build Sprint

**Input**: Design documents from `/specs/001-sentinel-mvp/`
**Prerequisites**: plan.md ✅, spec.md ✅, blueprint.md ✅
**Tests**: included where they protect a non-negotiable principle (latency, confidence-gating contract)
**Organization**: Tasks are grouped by *day* rather than by user story, because the 6-day window is the binding constraint. User-story alignment noted on each task via [US1/US2/US3/US4].

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task serves
- File paths included for every implementation task

## Path Conventions

- Backend: `backend/sentinel/`, `backend/app/`, `backend/tests/`
- Frontend: `frontend/`
- Hook: `sentinel-hook.py` at root
- Deploy: `deploy/`
- Configs: `configs/`
- Data: `data/`

---

## Day 1 (2026-05-14) — Pipeline & End-to-End Happy Path

**Purpose**: Prove every leg of the architecture works before writing any detection logic. Constitution Principle IV (Demo-First) is the gate.

- [x] **T001** [US1] Create project directory structure per plan.md
- [x] **T002** [US1] Initialize spec-kit, write constitution.md, CLAUDE.md, blueprint.md
- [x] **T003** [P] [US1] Create `backend/pyproject.toml` declaring deps: fastapi, uvicorn, pydantic>=2, httpx, structlog, sqlite-utils, rapidfuzz, google-generativeai, openai (for Featherless compat); python-version 3.11
- [x] **T004** [P] [US1] Create `backend/app/main.py` — FastAPI app with placeholder POST /detect, GET /health, GET /events (SSE stub). Returns mocked `Decision`.
- [x] **T005** [P] [US1] Create `sentinel-hook.py` at repo root — zero-dependency, 100-line, reads stdin JSON, POSTs to localhost:7777/detect, translates verdict → exit code
- [x] **T006** [P] [US1] Create `deploy/docker-compose.yml` and `deploy/Dockerfile.backend` — single container running uvicorn on :7777
- [x] **T007** [P] [US1] Create `deploy/Caddyfile` — reverse proxy with auto-TLS for sentinel.<domain> (rewritten Day 1 evening to use explicit `handle` blocks after directive-order trap)
- [x] **T008** [US1] Create `Makefile` with targets: `install`, `dev`, `demo`, `test`, `bench`, `deploy-vultr`, `smoke-vultr`
- [x] **T009** [US1] Create `.gitignore`, `.env.example`, `LICENSE` (Apache 2.0), `README.md`
- [x] **T010** [US1] Initialize git repo, two commits on `main`
- [x] **T011** [US1] Verify local daemon roundtrip: `make dev` → POST /detect → JSON response → hook script exit codes verified (exit 2 with daemon up + AUTO_CORRECT; exit 0 fail-open with daemon down)
- [x] **T012** [US1] Provision Vultr VM `vx1-g-2c-8g` in Milan (66.245.207.218), 50GB Ubuntu 24.04 bootable volume, SSH key + firewall group attached
- [x] **T013** [US1] First deploy to Vultr live at **https://sentinel.66-245-207-218.nip.io** — Let's Encrypt cert, `/health` 200 OK, `/detect` returns AUTO_CORRECT JSON. **DAY 1 CHECKPOINT GREEN — 2026-05-15 ✅**
- [x] **T014** [P] [US1] Day 1 bait verification (executed 2026-05-15/16):
  - **Trial 1** Claude Sonnet 4.6, overt phantom prompt → refused cleanly (RLHF abstention working). Not Sentinel's target audience.
  - **Trial 2** Llama-3.1-8B-Instruct via Featherless, subtle prompt + `tool_choice: auto` → Tool Bypass (Healy Type 5): described actions in text instead of calling any tool.
  - **Trial 3** Same model + `tool_choice: required` + "invent the most logical tool" directive → **6 ghost tool claims in `content`** (`Database Interface Tool`, `Data Storage Tool`, etc.) plus a fabricated success message. Captured at `data/evidence/2026-05-16-llama-ghost-claims.md`.
  - **Trial 4** Same model + "JSON-ONLY, invent a tool name if missing" directive → **bare phantom**: response was the single string `` `save_core_database_findings` ``. Captured at `data/evidence/2026-05-16-llama-bare-phantom.md`.
  - **Conclusion:** phantom fabrication is real on cheap open-source models. Featherless silently ignores `tool_choice: "required"` — phantom names emerge in `content` rather than `tool_calls`. **SCOPE REFINEMENT for Day 2:** Layer 1 must scan tool-name-like tokens in both `tool_calls[].function.name` and assistant `content`. Decision schema gains `ghost_claims: list[str]` field.
  - **Featherless platform quirks documented** for the spec/quirks list.

**Checkpoint**: ✅ Public URL works. Hook + daemon + Caddy + TLS end-to-end. Bait verification deferred.

**Day 1 retro (for the record):**
- Vultr UX trap: SSH keys + firewall groups must exist as account-level resources before the deploy wizard's dropdowns will populate them. Cost ~1h of UI confusion.
- Cloud-init YAML: `runcmd` block silently skipped when `package_upgrade: true` exceeds default timeout. Fix: install Docker + UFW rules manually post-SSH; future deploys should set `package_upgrade: false`.
- `sslip.io` is Let's Encrypt rate-limited (250k/week burned by other users). Fix: swap to `nip.io`.
- Caddyfile v2 directive order: bare `reverse_proxy /path upstream` directives lose to a bare `respond` catch-all. Fix: wrap every proxy in `handle /path { ... }`.

---

## Day 2 (2026-05-15) — Schemas, Layer 1, Heuristics

**Purpose**: Define the type contracts and ship the cheap layer.

- [x] **T015** [P] [US1] Wrote `backend/sentinel/schemas.py` — pydantic v2 models with constitutional invariants enforced via `model_validator`. Includes new `GhostClaim` and `tool_name max_length=200` DOS guard.
- [x] **T016** [P] [US1] `backend/tests/contract/test_decision_schema.py` — 18 contract tests, including `test_decision_with_ghost_claims_json_roundtrip`.
- [x] **T017** [P] [US1] `backend/tests/contract/test_detect_request_schema.py` — 9 round-trip + bounds tests.
- [x] **T018** [US1] `backend/sentinel/layer1.py` — O(1) hash lookup via cached `_by_lower` dict (`PrivateAttr`); live latency on Vultr `0.0023ms`.
- [x] **T019** [P] [US1] `backend/tests/unit/test_layer1.py` — 7 tests including tightened latency gate (median <0.5ms over 200 runs).
- [x] **T020** [P] [US1] `backend/sentinel/heuristics.py` — F1 Levenshtein (rapidfuzz), F2 Jaccard, F3 gap with config-driven multiplier, fuse with config-driven weights.
- [x] **T021** [P] [US1] `backend/tests/unit/test_heuristics.py` — 26 tests with hand-computed expected values + `pytest.approx` for IEEE-754 edges.
- [x] **T022** [US1] `backend/sentinel/registry.py` — yaml -> ToolRegistry; env/repo/etc search path; LRU-cached.
- [x] **T023** [US1] `configs/registry.yaml` — `web_search` + `mcp__lint_check` baseline (matches Featherless evidence corpus).
- [x] **T024** [US1] `configs/cascade.yaml` + `backend/sentinel/config.py` — externalised thresholds (0.85, 0.60), fusion weights, F3 multiplier. Single source of truth; daemon loads at startup; `0.85` and `5.0` magic numbers GONE from source. Constitution Principle V cleared.
- [x] **T025** [US1] `backend/app/main.py` — live `load_registry()` on startup, real `layer1()` on every `/detect`, structlog JSON audit (`daemon_startup`, `phantom_intercepted`). `/health` reports `registry_loaded + size + version`.

**Checkpoint**: ✅ Real Layer 1 works against real registry over public HTTPS at https://sentinel.66-245-207-218.nip.io. 61/61 tests pass in 0.09s. Vultr daemon runs as non-root `sentinel` (uid 10001); sshd hardened (key-only, no root password); `.env` chmod 600. **Day 2 GREEN, Complete & Deployed to Milan — 2026-05-16.**

**Day 2 retro (for the record):**
- `tool_choice: "required"` is not enforced by Featherless for Llama 3.1 8B → phantom tools emerge in `content` rather than structured `tool_calls`. Decision schema gained `ghost_claims` field accordingly.
- Pydantic v2 `frozen=True` + `PrivateAttr` works fine together — set via `object.__setattr__` inside `@model_validator(mode="after")`. Cached registry index dropped Layer-1 latency from ~50µs to 1.3µs locally / 2.3µs on Vultr public HTTPS.
- `rsync --delete` nukes `.env` on the VM (because `.env` lives only on the VM, not in the local working tree). Always pass `--exclude='.env'`.
- Docker named volumes inherit the IMAGE's directory permissions ONLY on first creation. If you change `USER` in the Dockerfile, you must drop the existing volume (`docker volume rm`) so the next mount picks up the new ownership. Caddy's volume must NOT be dropped — that wipes the Let's Encrypt cert and triggers a rate-limited re-issue.

---

## Day 3 (2026-05-16) — Layer 2 + Layer 3 + Latency Gate

**Purpose**: Complete the cascade and prove the 10ms median budget on Layer 1+2.

- [ ] **T026** [US1] Write `backend/sentinel/embeddings.py` — Featherless OpenAI-compatible client (chat completions or embeddings endpoint), on-disk LRU cache via `diskcache`
- [ ] **T027** [US1] Write daemon warm-up routine: on startup, embed every tool's signature (name + description + schema keys), store in in-memory dict
- [ ] **T028** [US1] Write `backend/sentinel/layer2.py` — `def layer2(req, registry, embed_fn) -> (confidence, candidates, layer_breakdown)`. Cosine via numpy. Apply F1/F2/F3 fusion when 0.60 ≤ base ≤ 0.85.
- [ ] **T029** [P] [US1] Write `backend/tests/unit/test_layer2.py` — 6 cases: clean match, near-typo (F1 dominant), schema-twin (F2 dominant), wide top-1 gap (F3 dominant), tied top-1 (escalate), no candidate above 0.3 (BLOCK)
- [ ] **T030** [US1] Write `backend/sentinel/layer3.py` — Gemini Flash client with structured-output JSON schema from `docs/blueprint.md` §4. Pydantic-validate the response.
- [ ] **T031** [P] [US1] Write `backend/tests/unit/test_layer3.py` — 3 cases (using `respx` to mock Gemini): ALLOW, AUTO_CORRECT, BLOCK; +1 timeout case → degraded=True
- [ ] **T032** [US1] Write `backend/sentinel/cascade.py` — orchestrator wiring L1 → L2 → L3. Returns final `Decision` with `layer_breakdown`.
- [ ] **T033** [P] [US1] Write `backend/tests/integration/test_cascade_end_to_end.py` — happy path scenarios across all 4 verdicts
- [ ] **T034** [US1] Write `backend/bench/run_bench.py` — minimum viable: load 50 hand-picked examples, run cascade, dump `results/<date>-pilot.json`
- [ ] **T035** [US1] **Latency gate**: `make bench-latency` — run cascade 1,000 times on Layer-1-hit + Layer-2-hit cases (no Layer 3), assert median ≤10ms. Fail build if exceeded. **DAY 3 CHECKPOINT**
- [ ] **T036** [P] [US1] Update mocked `app/main.py` to use real `cascade.detect()`; redeploy to Vultr.

**Checkpoint**: Full cascade works. Latency budget verified. Vultr serves real decisions.

---

## Day 4 (2026-05-17) — Dashboard + Benchmark + Calibration

**Purpose**: Render the live timeline + the headline benchmark chart.

- [ ] **T037** [P] [US2] Scaffold `frontend/` with Next.js 15 + Tailwind v4 + shadcn/ui (`npx create-next-app@latest frontend`)
- [ ] **T038** [P] [US2] Write `frontend/components/live-timeline.tsx` — SSE subscriber, renders rows with color-coded verdicts, confidence bars, suggestion badge
- [ ] **T039** [P] [US2] Write `frontend/components/pareto-chart.tsx` — recharts scatter, x=latency, y=recall, color=cost; reads from `/api/bench/latest`
- [ ] **T040** [P] [US2] Write `frontend/app/page.tsx` — two-tab layout (Live | Benchmark), responsive
- [ ] **T041** [US2] Write `frontend/lib/sse.ts` — EventSource wrapper with reconnect
- [ ] **T042** [US2] Daemon: implement `/events` SSE properly (not stub from T004); broadcast every Decision after persistence
- [ ] **T043** [US2] Daemon: add `/api/bench/latest` returning latest `results/*.json`
- [ ] **T044** [US3] Assemble `data/sentinel-bench-v1/` — NTA + DT (296×2 prompts following RT §3 protocol) + Glaive Function Calling v2 sample (1000)
- [ ] **T045** [US3] Write `backend/bench/sentinelbench_v1.py` — dataset loader, 60/20/20 stratified split, deterministic seed
- [ ] **T046** [US3] Extend `backend/bench/run_bench.py` to full benchmark: per-layer accuracy/F1/recall/latency/cost, write to `results/<date>-<sha>.json`
- [ ] **T047** [US3] Write `backend/bench/calibrate.py` — grid search τ ∈ [0.1, 0.9] step 0.05 on val split; pick F1-max τ*; persist to `configs/cascade.yaml` with mean ± std over 3 seeds
- [ ] **T048** [US3] Write `backend/bench/pareto.py` — render 3-axis chart, save PNG to `results/<date>-pareto.png` and `results/latest-pareto.png`
- [ ] **T049** [US3] **Bench gate**: `make bench` end-to-end ≤15 minutes on a clean clone; results pass F1 ≥0.80
- [ ] **T050** [US2] Deploy dashboard to Vultr alongside daemon (`docker-compose up -d frontend`); confirm public URL renders both tabs

**Checkpoint**: Public URL shows live timeline + Pareto chart. Benchmark reproducible.

---

## Day 5 (2026-05-18) — Hardening + Demo Prep + Stretch

**Purpose**: Make the demo bulletproof. Touch stretch goals only if everything above is clean.

- [ ] **T051** [US1] Edge case hardening: daemon-down failover in hook (test by killing daemon mid-call), Layer 3 timeout (mock 3s response), NaN/503 from Featherless, concurrent 50 RPS for 60s stress test
- [ ] **T052** [US1] Demo bait corpus freeze: 2 final bait prompts × 3 LLMs × 10 attempts each. Document trigger rates in `data/bait-corpus/README.md`. Reject below 80%.
- [ ] **T053** [US1] Record replay fallback: capture 5 full sessions (request → cascade → decision) as JSON, can be replayed via `make demo-replay` if live network fails on demo day
- [ ] **T054** [P] [US1] Write `docs/DEMO_SCRIPT.md` — 30-second video shot list with exact timestamps, narration optional, camera angle notes
- [ ] **T055** [P] [US1] Write `docs/ARCHITECTURE.md` — diagram (mermaid), one paragraph per layer, link to blueprint.md
- [ ] **T056** [P] [US3] Mining experiment E5: scan `~/.claude/projects/*/` for own session traces; extract tool calls; manually label phantoms; run cascade; produce mini-chart for demo video
- [ ] **T057** [P] [US1] **Stretch**: MCP middleware (`backend/sentinel/mcp_proxy.py`) — accepts MCP `tools/call` JSON-RPC, runs cascade, forwards or rejects. Defer if Day 5 morning behind schedule.
- [ ] **T058** [US4] Write `install.sh` — curl-bash one-liner: copies hook to ~/.local/bin, patches `~/.claude/settings.json`, sets up launchd/systemd service. Optional polish.
- [ ] **T059** [US1] README rewrite: tagline, problem statement, install, demo GIF, benchmark table, sponsor acknowledgments, citation list
- [ ] **T060** [US1] Smoke test: clean macOS VM clone, `git clone && make demo` works inside 5 minutes

**Checkpoint**: Demo bulletproof. Replay fallback ready. MCP middleware optional.

---

## Day 6 (2026-05-19) — Submit

**Purpose**: Ship. No new features today.

- [ ] **T061** [US1] Record demo video — split-screen Claude Code, 30 seconds, two takes minimum, pick best
- [ ] **T062** [US1] Final deploy: `make deploy-vultr`, `make smoke-vultr` passes
- [ ] **T063** [US1] GitHub repo public, frozen tag `v0.1.0-hackathon-submission` pushed
- [ ] **T064** [US1] Submit hackathon form: public URL, GitHub URL, demo video URL, architecture diagram, written summary (~300 words)
- [ ] **T065** [US1] **Submission deadline buffer**: submit at 12:00 CET, 8 hours before 20:00 CET deadline
- [ ] **T066** [US1] Post submission: write LinkedIn / X post per Kraken sponsor track social engagement bonus (also a hedge for the Kraken Social Engagement award, even though we're not on the Kraken track)

**Checkpoint**: Submitted with 8h buffer. Buffer reserved for emergency.

---

## Dependencies & Execution Order

### Day-level dependencies

- **Day 1** (T001–T014): no external blockers; needs API keys (Vultr, Gemini, Featherless) acquired early in the day
- **Day 2** (T015–T025): depends on Day 1 T011 (local daemon roundtrip working)
- **Day 3** (T026–T036): depends on Day 2 schemas; latency gate (T035) blocks T036 redeploy
- **Day 4** (T037–T050): depends on Day 3 T032 (cascade complete); benchmark requires real Layer 2 + Layer 3
- **Day 5** (T051–T060): depends on Day 4 deploy; replay fallback (T053) is a safety net for Day 6
- **Day 6** (T061–T066): every prior day's checkpoint must be green

### Parallel opportunities

- Within Day 1: T003–T010 are all [P] (different files)
- Within Day 2: T015–T017 contract layer parallel; T020–T021 heuristics parallel
- Within Day 3: T029, T031, T033, T036 marked [P] but T033 depends on T032 complete
- Within Day 4: T037–T040 frontend scaffolding parallel; T044 dataset assembly parallel to T037 frontend
- Within Day 5: T054, T055, T056, T057 are all [P]

### NON-NEGOTIABLE gates

1. **Day 1 EOD**: Vultr public URL returns 200, hook fires on Claude Code, ≥1 bait prompt triggers phantoms (T013 + T014)
2. **Day 3 EOD**: `make bench-latency` median ≤10ms (T035)
3. **Day 4 EOD**: `make bench` reproduces F1 ≥0.80 (T049)
4. **Day 5 EOD**: `make demo` works on fresh clone in ≤5 min (T060)
5. **Day 6 12:00 CET**: submission complete (T065)

If any gate fails, the demo-first principle kicks in: cut scope (drop MCP middleware, drop E5, drop dashboard polish) to land the gate.

---

## Notes

- [P] tasks = different files, safe to parallelize within a day's slot
- [Story] label maps task to user story for traceability
- Tests included for **contract enforcement** (Decision schema) and **latency gates** — not for code coverage as a goal. The constitution says demo-first; tests where they protect the demo.
- Each task should result in a commit. Daily check-in with self: "did each gate pass?"
- Avoid: any feature not tracked back to a user story; any threshold hardcoded in source (must live in `configs/`); any backwards-compat hack ("just in case we ever support X")
- Submit-day buffer is sacred. Do not write code on Day 6 after 14:00 CET.
