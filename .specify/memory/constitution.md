# Sentinel Constitution

> Sentinel is a phantom tool-call detector and self-correction layer for autonomous LLM agents. This constitution defines the non-negotiable principles that govern every architectural and implementation decision. Drafted for the AI Agent Olympics Hackathon at Milan AI Week 2026; intended to survive past the hackathon as a real open-source project.

## Core Principles

### I. Library-First

Every capability ships first as an independently usable Python library inside `backend/sentinel/`. The FastAPI daemon, the Claude Code hook, the MCP middleware (stretch), and any future SDK wrapper are *thin clients* over the library. A user must be able to install `pip install sentinel-detect` and call `from sentinel import detect` without standing up any server, daemon, or web UI.

**Why:** Hackathons reward demos, but production users adopt libraries. A library-first design lets the same code drive (a) the daemon for the live demo, (b) the CLI for offline experiments on JKU compute, (c) the test suite, and (d) downstream integrations. No code duplication between server and standalone modes.

**How to apply:** All detection logic lives in `backend/sentinel/`. FastAPI routes are 10-line adapters over library calls. The hook script is a pure stdin/stdout/exit-code shim. If a feature requires a daemon to function, it does not belong in v1.

### II. 10ms Latency Budget for Layers 1+2 (NON-NEGOTIABLE)

The cumulative wall-clock time from a tool-call interception to a non-Layer-3 decision must remain under **10 milliseconds at median** on commodity Linux hardware (single-core, no GPU). This is the headline performance claim and the commercial argument: Sentinel is invisible in the common case. Layer 3 (Gemini Flash verifier) is allowed up to 400ms because it only fires on ambiguous cases.

**Why:** A modern Claude Code session can issue 50–200 tool calls. Anything above ~10ms per call compounds into a perceptible UI lag. Spectral Guardrails (Noël, Feb 2026) sets the academic ceiling at 10–50ms for spectral methods on N<200 tokens; we beat that on the cheap path by using cached embeddings rather than per-call spectral decomposition.

**How to apply:** Layer 1 (registry exact match) must be an in-memory hash lookup, never a database call. Layer 2 (embedding similarity) must use *precomputed* embeddings for the tool registry, cached at daemon start and invalidated only when registry changes. The hook script must have zero heavy imports — `sys`, `json`, `urllib.request` only. If a feature breaks this budget, it goes to Layer 3 or stays out of v1.

### III. Confidence-Gated Self-Correction (NON-NEGOTIABLE)

Sentinel never silently replaces one tool call with another. Every intercept maps to one of four verdicts: `ALLOW`, `AUTO_CORRECT`, `SUGGEST`, `BLOCK`. The verdict is gated on a confidence score in [0, 1] with documented thresholds: ≥0.9 auto-correct, 0.6–0.9 surface choices, <0.6 hard block. The agent must always receive enough information in the stderr `reason` field to understand *why* a call was modified or rejected, so it can revise its plan.

**Why:** Gemini's own critique (the "Proving a Negative" demo problem) plus our research synthesis show that a wrong correction is worse than a clean block — it fabricates a new hallucination at a layer the user trusts more than the model. Confidence gating prevents Sentinel from becoming the next hallucination source.

**How to apply:** The `Decision` schema is the authoritative interface. Every code path that emits a decision must populate `verdict`, `confidence`, `reason`, and (for `AUTO_CORRECT`/`SUGGEST`) `suggestion`. No path may emit `AUTO_CORRECT` with confidence < 0.85. No path may emit `BLOCK` without a `reason`. Unit tests enforce the threshold contract.

### IV. Demo-First Engineering

Day 1 ships a working *end-to-end happy path* — Claude Code session → hook → daemon → mocked decision → stderr injection → agent retries with corrected call. Every subsequent day's work either (a) raises detection accuracy on the demo scenario, (b) reduces latency, or (c) hardens the demo against live-network failures. Features that do not move one of those three needles are deferred to a v2 roadmap.

**Why:** Solo developer, 6 days, hackathon submission with a recorded demo video. Most hackathon projects die because their builders front-load infrastructure (auth, DB migrations, CI/CD) at the expense of the demo. We invert: the demo *is* the spec.

**How to apply:** No authentication, no user accounts, no billing, no multi-tenant. SQLite for persistence (file-based, no migrations). Vultr deploy = single VM with `docker-compose up`. The demo bait corpus is curated by hand on Day 1 morning and frozen by Day 4; live model regression on Day 5 is forbidden.

### V. Reproducibility Over Cleverness

Every benchmark number reported in the README, the demo video, or the architecture diagram must be reproducible from a single `make benchmark` command using only the artifacts in this repository plus public API access. No hand-tuned thresholds, no cherry-picked traces, no live-network results in the headline chart.

**Why:** The research synthesis showed that hackathon winners ship reproducible work that survives scrutiny after submission. Internal Representations (Healy et al., Jan 2026) reports threshold τ selected via deterministic grid search [0.1, 0.9] step 0.05 — we follow the same protocol. Reviewers asking "what was your config" should always find a checked-in answer.

**How to apply:** Calibration outputs (threshold τ, confidence weights) are committed as JSON under `configs/`. Bait corpus + labels under `data/sentinel-bench-v1/` with checksums. The benchmark script writes results + commit SHA to `results/<date>-<sha>.json` so any chart in the paper can be regenerated. No floats in code; all thresholds load from config.

## Performance Standards

- **Layer 1 (registry):** median <1ms, p95 <2ms, in-memory hash lookup only
- **Layer 2 (embedding):** median <10ms, p95 <20ms, Featherless embedding API with on-disk LRU cache of registry embeddings
- **Layer 3 (Gemini Flash verifier):** median <300ms, p95 <500ms, fires on <15% of calls in steady state
- **Cascade overall:** median <15ms across all calls (weighted by 85% L1+L2 / 15% L3)
- **Hook subprocess startup:** <50ms cold start (Python with `-S`); Go binary fallback documented
- **Daemon memory:** <512MB resident on a 1-vCPU Vultr instance, including loaded embeddings for ~500 tools

## Quality Gates

1. **Demo gate:** End-to-end demo path must work on a fresh clone within 5 minutes (`make demo`). Tested daily on a clean VM during build week.
2. **Constitution check:** Every PR (or commit when solo) must verify the four NON-NEGOTIABLE principles are not violated. Complexity violations require an explicit row in `plan.md` § Complexity Tracking.
3. **Calibration gate:** Confidence thresholds must achieve F1 ≥ 0.80 on the held-out validation set before deploy. Below that, the demo falls back to replay-from-trace mode.
4. **Latency gate:** `make benchmark-latency` must report median Layer-1+Layer-2 ≤10ms or build fails. Enforced from Day 3 onward.

## Governance

This constitution supersedes ad-hoc decisions made during build sprints. Amendments require:

1. Explicit reference in the relevant commit message or PR
2. Update to the version number below
3. A one-line rationale in the corresponding spec / plan / tasks file

For runtime development guidance, see `CLAUDE.md` at the repository root.

**Version**: 1.0.0 | **Ratified**: 2026-05-14 | **Last Amended**: 2026-05-14
