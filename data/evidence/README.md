# Sentinel Evidence Corpus

Real, reproducible captures of LLM agent failure modes that Sentinel is designed to catch. Used in the demo video, the benchmark, and the README's "the problem is real" section.

Each evidence file is structured as:

- **Date** — capture date in `YYYY-MM-DD`
- **Model** — exact model id used
- **Provider** — API host
- **Setup** — request shape (tools registered, tool_choice, system prompt)
- **Prompt** — verbatim user message
- **Response** — verbatim assistant output
- **Failure classification** — mapped to Healy IR §3 error taxonomy
- **Reproduction** — exact `curl` command + JSON body to regenerate the failure

## Index

| File | Date | Model | Failure type |
|---|---|---|---|
| [`2026-05-16-llama-ghost-claims.md`](./2026-05-16-llama-ghost-claims.md) | 2026-05-16 | meta-llama/Meta-Llama-3.1-8B-Instruct | Multi-tool ghost claims in `content` (Healy Type 1 + Type 5) |
| [`2026-05-16-llama-bare-phantom.md`](./2026-05-16-llama-bare-phantom.md) | 2026-05-16 | meta-llama/Meta-Llama-3.1-8B-Instruct | Bare phantom tool name as content (Healy Type 1) |

## Failure-type taxonomy (Healy et al., Internal Representations, Jan 2026)

| Type | Name | Description |
|---|---|---|
| 1 | Function Selection Error | `f̃ ∉ F` — invoking a non-existent function |
| 2 | Function Appropriateness Error | `f̃ ∈ F` but semantically inappropriate for query |
| 3 | Parameter Error | `a ∉ D_f` — invalid argument value |
| 4 | Completeness Error | Required parameters missing |
| 5 | Tool Bypass Error | Generating output without using available tools |

Sentinel v1 scope: **Type 1 detection (with extensions to Type 5 when fabricated names appear in `content`)**.
