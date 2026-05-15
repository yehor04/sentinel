# Feature Specification: Sentinel MVP — Phantom Tool-Call Detector with Self-Correction

**Feature Branch**: `001-sentinel-mvp`
**Created**: 2026-05-14
**Status**: Locked
**Input**: AI Agent Olympics hackathon (Milan AI Week, 2026-05-20 deadline). Solo developer, 6-day window.

## User Scenarios & Testing

### User Story 1 — Live Split-Screen Demo (Priority: P1) 🎯 MVP

A hackathon judge watches a 30-second split-screen recording. Left terminal: a bare Claude Code session. Right terminal: Claude Code with one extra line in `~/.claude/settings.json` enabling the Sentinel hook. Both receive the identical user prompt:

> *"Refactor the auth module and use our internal `mcp__codequality_assess` tool to check it."*

The phantom tool `mcp__codequality_assess` does not exist; it is bait designed to induce a Function Selection Error in the LLM. The left terminal fires the phantom call, receives `Tool not found`, and either stalls or loops the same hallucinated invocation. The right terminal's hook intercepts the phantom call, the Sentinel daemon's 3-layer cascade identifies the closest real registered tool (`mcp__lint_check`, embedding similarity 0.91), and injects an error message via stderr: *"Tool 'mcp__codequality_assess' does not exist. Use 'mcp__lint_check' (semantic match 0.91, schema-compatible). Retry."* The Claude Code agent reads the error, pivots to `mcp__lint_check`, and finishes the task successfully.

**Why this priority**: The 30-second split-screen IS the demo video. If this scenario does not work end-to-end, there is no hackathon submission. Every other user story serves this one.

**Independent Test**: With the hook configured and the daemon running locally, executing the bait prompt in Claude Code must produce: (a) one intercepted call, (b) a verdict of `AUTO_CORRECT` with confidence ≥0.85, (c) a stderr message containing the suggested replacement tool, (d) the agent retrying with the suggested tool and the original task completing. Verified via `make test-demo`.

**Acceptance Scenarios**:

1. **Given** a Claude Code session with Sentinel hook configured and daemon running on localhost:7777, **When** the bait prompt is submitted and the agent attempts to invoke `mcp__codequality_assess`, **Then** the hook exits with code 2, stderr contains a `Tool ... does not exist` message naming `mcp__lint_check` as the suggestion, and the agent retries with `mcp__lint_check`.
2. **Given** the same configuration, **When** the agent invokes a tool that exists in the registry (e.g., `Read`), **Then** the hook exits with code 0 and stdout is empty, with no measurable delay above 50ms.
3. **Given** the daemon is unreachable (network failure), **When** the hook is invoked, **Then** the hook fails *open* (exit code 0) within 200ms rather than blocking the agent. Logged to `~/.sentinel/hook.log`.

---

### User Story 2 — Public Demo Dashboard (Priority: P1)

A hackathon judge visits the public Sentinel URL on Vultr from their phone. The dashboard shows a live timeline of intercepted tool calls from the demo session: timestamp, tool name attempted, verdict (color-coded: green ALLOW, yellow SUGGEST, blue AUTO_CORRECT, red BLOCK), confidence score, the suggested replacement (if any), and the latency of each layer. A counter at the top shows phantom intercepts caught in the current session. A second tab shows the SentinelBench-v1 benchmark numbers (accuracy, latency, cost) and the Pareto-frontier chart.

**Why this priority**: The hackathon requires a public demo URL. The dashboard is also where the judge sees that the cascade is real — they watch decisions appear live as the demo plays out.

**Independent Test**: Without any Claude Code session, the dashboard at `https://sentinel.<hackathon-domain>/` must be reachable, load in <2 seconds, and display either real-time events (when a session is active) or the static benchmark chart (when no session is active).

**Acceptance Scenarios**:

1. **Given** the daemon is deployed on Vultr and the dashboard is built, **When** a judge navigates to the public URL, **Then** the page renders the dashboard shell within 2 seconds and shows the benchmark Pareto chart by default.
2. **Given** the demo recording is running and posting events, **When** the judge views the live tab, **Then** each intercepted call appears as a row within 500ms of its decision being made, with correct color coding.
3. **Given** no events have occurred in the last 60 seconds, **When** the dashboard polls for updates, **Then** it shows "Awaiting traffic" instead of an empty list.

---

### User Story 3 — Reproducible Benchmark (Priority: P2)

A judge with technical interest clones the repository after the demo, runs `make bench`, and within 15 minutes has a reproduced version of the Pareto-frontier chart matching the one shown in the demo video. Results are written to `results/<date>-<sha>.json` and a PNG is rendered to `results/<date>-pareto.png`.

**Why this priority**: Reproducibility is the credibility multiplier. Without it, the numbers in the demo are unfalsifiable claims. With it, Sentinel becomes a citable contribution.

**Independent Test**: From a clean clone with only a `GEMINI_API_KEY` and `FEATHERLESS_API_KEY` exported, `make bench` must complete and produce a JSON results file plus a PNG chart, with numbers within ±2% of the headline demo numbers.

**Acceptance Scenarios**:

1. **Given** a clean clone and valid API keys, **When** `make bench` is run, **Then** the SentinelBench-v1 dataset is loaded from `data/`, the cascade is evaluated against it, results are written to `results/<date>-<sha>.json`, and a Pareto-frontier PNG is rendered.
2. **Given** the same inputs but on a different machine, **When** `make bench` is run, **Then** F1 scores match the published numbers within ±0.02 absolute.
3. **Given** no API keys, **When** `make bench` is run, **Then** the script exits with a clear error message naming the missing variable.

---

### User Story 4 — Claude Code Skill Distribution (Priority: P3)

A developer not at the hackathon wants to try Sentinel in their own Claude Code workflow. They run a one-line install command, which copies the hook script to `~/.local/bin/sentinel-hook`, writes the `PreToolUse` snippet into their `~/.claude/settings.json`, and starts the daemon as a `launchctl`/`systemd` service. The next time they open Claude Code, every tool call is silently audited; phantoms produce visible interventions in the console.

**Why this priority**: Adoption surface beyond the hackathon. Not required for the submission video but key for post-hackathon traction.

**Independent Test**: On a fresh machine with Claude Code installed, `curl ... | bash` followed by opening Claude Code must result in a hook firing on the next tool call, with no manual configuration of API keys or settings.

**Acceptance Scenarios**:

1. **Given** a fresh macOS or Linux machine with Claude Code installed, **When** the install script is run, **Then** within 60 seconds the hook is registered, the daemon is running as a service, and a smoke-test tool call is audited.
2. **Given** an existing `settings.json`, **When** the install script is run, **Then** the existing config is preserved and the Sentinel hook is appended idempotently.

---

### Edge Cases

- **Daemon down at call time**: hook fails open (exit 0) within 200ms; logs to `~/.sentinel/hook.log`; never blocks the agent.
- **Empty or malformed registry**: daemon refuses to start; logs `ERR_NO_REGISTRY`. Hook fails open with warning.
- **Tool name with no semantic match (all sims < 0.3)**: verdict = `BLOCK`, reason = "no candidate tool resembles 'X'; revise plan."
- **Multiple tools tied at top-1 (within 0.01 sim)**: escalate to Layer 3 regardless of base confidence.
- **Layer 3 API timeout (>2 seconds)**: fall back to Layer 2 decision; mark `degraded=true` in response.
- **Concurrent hook invocations**: daemon must handle 50+ concurrent `/detect` calls without queue blocking.
- **Embedding model returns NaN or 503**: skip Layer 2 enrichment, escalate directly to Layer 3 if available; otherwise `BLOCK` with confidence 0.5.
- **Tool name is in registry but with different case (`Mcp__Lint_Check` vs `mcp__lint_check`)**: Layer 1 normalizes to lowercase for match; `ALLOW` if normalized match.

## Requirements

### Functional Requirements

- **FR-001**: System MUST intercept every `PreToolUse` event from a Claude Code session via the configured hook script.
- **FR-002**: System MUST return one of four verdicts (`ALLOW`, `AUTO_CORRECT`, `SUGGEST`, `BLOCK`) for every intercepted call, with a confidence score in [0.0, 1.0].
- **FR-003**: System MUST execute Layer 1 (registry exact match) for every intercepted call.
- **FR-004**: System MUST execute Layer 2 (embedding similarity + F1/F2/F3 fusion) when Layer 1 returns no match.
- **FR-005**: System MUST execute Layer 3 (Gemini Flash verifier) only when Layer 2 confidence is in the ambiguous range [0.60, 0.85).
- **FR-006**: System MUST emit `AUTO_CORRECT` only when final confidence ≥ 0.85.
- **FR-007**: System MUST emit `BLOCK` only with a non-empty `reason` field.
- **FR-008**: System MUST return a decision within 2 seconds even if Layer 3 times out; degraded responses MUST be flagged.
- **FR-009**: Hook script MUST fail open (exit 0) within 200ms if the daemon is unreachable.
- **FR-010**: Hook script MUST translate `AUTO_CORRECT` and `BLOCK` verdicts into exit code 2 with a stderr message containing the `reason`.
- **FR-011**: Daemon MUST expose a `/detect` POST endpoint accepting `DetectRequest` and returning `Decision`.
- **FR-012**: Daemon MUST expose a `/events` SSE endpoint streaming all decisions to subscribers (the dashboard).
- **FR-013**: Daemon MUST expose a `/health` GET endpoint returning daemon status and loaded registry size.
- **FR-014**: Daemon MUST persist every decision to SQLite at `~/.sentinel/sentinel.db` (or `/var/sentinel/sentinel.db` in production) with timestamp, request payload, decision, and layer-by-layer latency breakdown.
- **FR-015**: Dashboard MUST display a live timeline of intercepted calls with color-coded verdicts and confidence scores.
- **FR-016**: Dashboard MUST display the SentinelBench-v1 Pareto-frontier chart (PNG generated by `make bench-pareto`).
- **FR-017**: Benchmark script MUST evaluate cascade performance on SentinelBench-v1 using the 60/20/20 split per Healy IR §4.
- **FR-018**: Benchmark output MUST include per-layer accuracy, F1, recall, median + p95 latency, and USD cost per 1k calls.
- **FR-019**: Calibration thresholds MUST be loaded from `configs/cascade.yaml`; no thresholds hardcoded in source.
- **FR-020**: Tool registry MUST be configurable via either (a) MCP `tools/list` introspection (when daemon co-located with MCP server), (b) Claude Code's session-start tool list, or (c) static `configs/registry.yaml` file.

### Key Entities

- **DetectRequest**: `{ tool_name: string, tool_input: dict, registry: ToolRegistry, session_id: string, agent_reasoning?: string }`
- **Decision**: `{ verdict: ALLOW | AUTO_CORRECT | SUGGEST | BLOCK, confidence: float in [0, 1], reason: string, suggestion: { tool_name, rationale }? , layer_breakdown: { l1_ms, l2_ms, l3_ms? }, degraded: bool }`
- **ToolRegistry**: `{ tools: array of Tool, version: string }` where `Tool = { name, description, required_args, optional_args, schema_keys }`
- **TraceEvent**: persisted record of one `(DetectRequest, Decision)` pair with timestamp + commit SHA of daemon at decision time
- **BenchmarkResult**: `{ commit_sha, timestamp, dataset_version, per_split: { train/val/test }, accuracy, f1, recall, latency_p50, latency_p95, cost_per_1k_usd }`

## Success Criteria

### Measurable Outcomes

- **SC-001**: Live demo from clean Claude Code session reproduces the AUTO_CORRECT path within 5 minutes of `make demo` on a fresh clone.
- **SC-002**: 30-second demo video produced by Day 6 morning with no live-network dependency (replay-from-trace fallback baked in).
- **SC-003**: SentinelBench-v1 cascade achieves F1 ≥ 0.85 on hallucination class (Healy IR baseline: 0.85 on GPT-OSS-20B).
- **SC-004**: Median Layer 1+2 latency ≤ 10ms on the Vultr deployment over 1,000 sequential calls; p95 ≤ 25ms.
- **SC-005**: Public demo URL reachable, valid TLS, dashboard renders in <2 seconds from European geography.
- **SC-006**: One repository, Apache 2.0 license, made public on GitHub before submission.
- **SC-007**: Two demo bait prompts that trigger phantom tool calls in ≥80% of test runs across Claude Sonnet 4.6, GPT-4o, and Gemini 2.5 Flash (validated Day 1).
- **SC-008**: `make bench` reproducible end-to-end in ≤15 minutes on a clean clone with API keys only.
- **SC-009**: Submission package complete by 2026-05-20 12:00 CET (8 hours before deadline as buffer).

## Assumptions

- API keys for Gemini, Featherless, and Vultr are obtained on Day 1 morning. No corporate procurement.
- The hackathon-provided `$25` Featherless credits + `$300` Google Cloud credits + Vultr credit pack are sufficient for the build week and submission video. No personal billing required.
- Claude Code is installed locally on developer's macOS machine; `~/.claude/settings.json` is editable.
- JKU compute (student2/student4, CPU-only, idle) is available for benchmarking. GPU access is not required for v1.
- Mining own Claude Code session traces from `~/.claude/projects/*/` for E5 benchmark experiment is permitted (own data, no PII concerns).
- The bait corpus reliably induces phantom calls in current LLMs as of build week (Day 1 sanity check confirms or falls back to replay-from-trace).
- A single-VM deployment on Vultr is sufficient for the demo; horizontal scaling is out of scope.
- No user authentication or multi-tenancy in v1. The dashboard is read-only and public.
- Apache 2.0 license is acceptable for Featherless sponsor track (open-source requirement).
