"""Sentinel benchmark runner — T034 (corpus accuracy) + T035 (latency gate).

Usage:
  # Latency gate: 1000 calls through L1+L2, assert p50 < gate-ms
  python -m bench.run_bench --latency-only --calls 1000 --gate-ms 10

  # Full corpus bench: accuracy + latency on SentinelBench-v1
  python -m bench.run_bench --dataset ../data/sentinel-bench-v1 --output ../results

Both modes write a JSON result file to --output (default: ../results/).
Exit code 1 if the latency gate fails; 0 on pass.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory so `sentinel` package resolves without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.embeddings import get_embedder
from sentinel.layer1 import layer1
from sentinel.layer2 import layer2, warm_up_registry
from sentinel.schemas import DetectRequest, Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Registry builders
# ---------------------------------------------------------------------------

_BENCH_REGISTRY_TOOLS = [
    Tool(name="Bash", description="Execute shell commands in the terminal", required_args=("command",)),
    Tool(name="Read", description="Read a file from the local filesystem", required_args=("file_path",)),
    Tool(name="Write", description="Write content to a file on the local filesystem", required_args=("file_path", "content")),
    Tool(name="Edit", description="Edit a file with precise string replacement", required_args=("file_path", "old_string", "new_string")),
    Tool(name="Glob", description="Find files matching a glob pattern", required_args=("pattern",)),
    Tool(name="Grep", description="Search for patterns in files using regex", required_args=("pattern",)),
    Tool(name="Task", description="Launch a sub-agent to handle a complex task", required_args=("description", "prompt")),
    Tool(name="WebFetch", description="Fetch content from a URL", required_args=("url",)),
    Tool(name="WebSearch", description="Search the web for information", required_args=("query",)),
    Tool(name="mcp__github__create_pull_request", description="Create a GitHub pull request", required_args=("title", "body")),
    Tool(name="mcp__github__get_file_contents", description="Get file contents from a GitHub repository", required_args=("path",)),
    Tool(name="mcp__github__list_issues", description="List issues in a GitHub repository", required_args=()),
    Tool(name="mcp__slack__send_message", description="Send a message to a Slack channel", required_args=("channel", "text")),
    Tool(name="mcp__memory__search", description="Search the memory knowledge graph", required_args=("query",)),
    Tool(name="mcp__memory__add", description="Add an observation to the memory knowledge graph", required_args=("content",)),
    Tool(name="mcp__postgres__query", description="Execute a SQL query against the PostgreSQL database", required_args=("sql",)),
    Tool(name="computer_use", description="Control the computer GUI via screenshots and actions", required_args=("action",)),
    Tool(name="str_replace_editor", description="Edit files using string replacement operations", required_args=("path", "old_str", "new_str")),
    Tool(name="think", description="Think through a problem step by step before acting", required_args=("thought",)),
    Tool(name="mcp__linear__create_issue", description="Create an issue in Linear project management", required_args=("title",)),
]


def _bench_registry() -> ToolRegistry:
    return ToolRegistry(tools=tuple(_BENCH_REGISTRY_TOOLS), version="bench-v1")


# ---------------------------------------------------------------------------
# Latency-only mode (T035)
# ---------------------------------------------------------------------------

_PHANTOM_NAMES = [
    "search_the_internet",
    "google_search",
    "execute_command",
    "fetch_webpage",
    "create_file",
    "write_to_disk",
    "read_from_disk",
    "find_files",
    "query_database",
    "send_slack",
    "open_pr",
    "search_memory",
    "list_directory",
    "navigate_to_url",
    "run_tests",
]

_LEGAL_NAMES = [t.name for t in _BENCH_REGISTRY_TOOLS]


def _run_latency(n_calls: int, gate_ms: float, output_dir: Path) -> int:
    """Run N cascade calls (mix of L1 hits and L2 misses), report latency.

    Uses the stub embedder so there's zero network I/O — this measures the
    cascade routing + registry hash + cosine math overhead, which is what the
    <10ms median budget covers. Real-world L2 adds the embedding roundtrip,
    but that's absorbed by the on-disk cache in steady state.

    Returns 0 on pass, 1 if p50 > gate_ms.
    """
    print(f"[bench-latency] Warming up stub embedder for {len(_BENCH_REGISTRY_TOOLS)} tools …")
    os.environ.setdefault("SENTINEL_EMBEDDER_PROVIDER", "stub")

    embedder = get_embedder()
    registry = _bench_registry()
    registry_embeddings = warm_up_registry(registry, embedder)

    # Build call list: half L1 hits (legal), half L2 misses (phantom)
    half = n_calls // 2
    calls: list[DetectRequest] = []
    for i in range(half):
        name = _LEGAL_NAMES[i % len(_LEGAL_NAMES)]
        calls.append(DetectRequest(tool_name=name, tool_input={"command": "ls"}, session_id="bench"))
    for i in range(n_calls - half):
        name = _PHANTOM_NAMES[i % len(_PHANTOM_NAMES)]
        calls.append(DetectRequest(tool_name=name, tool_input={"query": "test"}, session_id="bench"))

    print(f"[bench-latency] Firing {n_calls} calls ({half} L1-hits, {n_calls - half} L2-misses) …")

    latencies_ms: list[float] = []
    for req in calls:
        t0 = time.perf_counter()
        d1 = layer1(req.tool_name, registry)
        if d1 is None:
            layer2(req, registry, embedder, registry_embeddings)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    p_max = latencies_ms[-1]

    passed = p50 <= gate_ms

    result = {
        "mode": "latency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedder": embedder.__class__.__name__,
        "n_calls": n_calls,
        "gate_ms": gate_ms,
        "passed": passed,
        "latency_ms": {
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "max": round(p_max, 4),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"latency-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Also write latest.json for bench-pareto
    (output_dir / "latest.json").write_text(json.dumps(result, indent=2))

    status = "PASS" if passed else "FAIL"
    print(
        f"[bench-latency] {status}  p50={p50:.3f}ms  p95={p95:.3f}ms  p99={p99:.3f}ms  "
        f"max={p_max:.3f}ms  (gate={gate_ms}ms)"
    )
    print(f"[bench-latency] Results → {out_path}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Full corpus bench (T034)
# ---------------------------------------------------------------------------

def _run_corpus(dataset_dir: Path, output_dir: Path) -> int:
    """Run every entry in corpus.jsonl through the cascade; report accuracy.

    Uses stub embedder by default (no API key required). Pass
    SENTINEL_EMBEDDER_PROVIDER=gemini to use real Gemini embeddings — the
    accuracy numbers will be higher and more representative.

    Returns 0 always (corpus bench is informational, not a gate).
    """
    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"[bench] corpus not found at {corpus_path}", file=sys.stderr)
        return 1

    lines = [ln for ln in corpus_path.read_text().splitlines() if ln.strip()]
    total = len(lines)
    print(f"[bench] Loading {total} corpus entries …")

    os.environ.setdefault("SENTINEL_EMBEDDER_PROVIDER", "stub")
    embedder = get_embedder()
    registry = _bench_registry()
    registry_embeddings = warm_up_registry(registry, embedder)

    results: list[dict] = []
    correct = 0
    phantom_tp = phantom_fp = phantom_tn = phantom_fn = 0

    for raw in lines:
        entry = json.loads(raw)
        req = DetectRequest(
            tool_name=entry["tool_name"],
            tool_input=entry.get("tool_input", {}),
            agent_reasoning=entry.get("agent_reasoning"),
            session_id="bench-corpus",
        )
        t0 = time.perf_counter()
        d1 = layer1(req.tool_name, registry)
        if d1 is not None:
            decision = d1
        else:
            decision = layer2(req, registry, embedder, registry_embeddings)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        expected = entry["expected_verdict"]
        got = decision.verdict
        match = got == expected

        if match:
            correct += 1

        # Phantom detection confusion matrix
        label = entry["label"]
        is_phantom_pred = got != "ALLOW"
        is_phantom_truth = label == "phantom"
        if is_phantom_truth and is_phantom_pred:
            phantom_tp += 1
        elif is_phantom_truth and not is_phantom_pred:
            phantom_fn += 1
        elif not is_phantom_truth and is_phantom_pred:
            phantom_fp += 1
        else:
            phantom_tn += 1

        results.append({
            "id": entry["id"],
            "label": label,
            "tool_name": entry["tool_name"],
            "expected": expected,
            "got": got,
            "match": match,
            "confidence": round(decision.confidence, 4),
            "elapsed_ms": round(elapsed_ms, 4),
        })

    accuracy = correct / total
    precision = phantom_tp / (phantom_tp + phantom_fp) if (phantom_tp + phantom_fp) > 0 else 0.0
    recall = phantom_tp / (phantom_tp + phantom_fn) if (phantom_tp + phantom_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    output = {
        "mode": "corpus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedder": embedder.__class__.__name__,
        "dataset": str(dataset_dir),
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "phantom_detection": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": phantom_tp,
            "fp": phantom_fp,
            "tn": phantom_tn,
            "fn": phantom_fn,
        },
        "per_example": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bench-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    (output_dir / "latest.json").write_text(json.dumps(output, indent=2))

    print(f"[bench] accuracy={accuracy:.1%}  phantom F1={f1:.3f}  "
          f"P={precision:.3f} R={recall:.3f}")
    print(f"[bench] TP={phantom_tp} FP={phantom_fp} TN={phantom_tn} FN={phantom_fn}")
    print(f"[bench] Results → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel benchmark runner")
    parser.add_argument("--latency-only", action="store_true", help="Run latency gate only (T035)")
    parser.add_argument("--calls", type=int, default=1000, help="Number of cascade calls (latency mode)")
    parser.add_argument("--gate-ms", type=float, default=10.0, help="p50 latency gate in ms")
    parser.add_argument("--dataset", type=Path, default=Path("../data/sentinel-bench-v1"), help="Path to corpus directory")
    parser.add_argument("--output", type=Path, default=Path("../results"), help="Directory for result JSON files")
    args = parser.parse_args()

    if args.latency_only:
        rc = _run_latency(args.calls, args.gate_ms, args.output)
    else:
        rc = _run_corpus(args.dataset, args.output)

    sys.exit(rc)


if __name__ == "__main__":
    main()
