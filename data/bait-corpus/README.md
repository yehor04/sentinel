# Bait Corpus

Hand-crafted prompts that reliably induce phantom tool calls in current LLMs. Held out from `data/sentinel-bench-v1/` to avoid eval leakage — these prompts feed the live demo.

## Day-1 verification target (T014)

For each candidate bait prompt, test against Claude Sonnet 4.6 (via your current session). Promote to `bait-v1.json` if it triggers a phantom tool call in **≥4 of 5 runs**.

## Format

```json
{
  "id": "bait-001",
  "prompt": "...",
  "expected_phantom_tools": ["mcp__codequality_assess"],
  "expected_real_alternative": "mcp__lint_check",
  "trigger_rate": {
    "claude-sonnet-4.6": 0.0,
    "gpt-4o": 0.0,
    "gemini-2.5-flash": 0.0
  },
  "source": "hand-crafted",
  "notes": ""
}
```

## Day-5 freeze

By end of Day 5 (T052), promote only the 2 prompts with the highest cross-model trigger rates. Document trigger rates in this file. If no prompt achieves ≥80% in any model, fall back to replay-from-trace mode (T053).
