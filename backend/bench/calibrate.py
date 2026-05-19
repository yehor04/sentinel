"""Threshold calibration — grid-search auto_correct_min × block_max.

Runs the 55-example SentinelBench-v1 corpus through L1+L2 for every
(auto_correct_min, block_max) pair and prints the confusion matrix +
verdict accuracy. Only keeps candidates where phantom F1 stays 1.000.

Usage:
  python -m bench.calibrate --dataset ../data/sentinel-bench-v1
  python -m bench.calibrate --dataset ../data/sentinel-bench-v1 --write-best

--write-best patches configs/cascade.yaml with the best-accuracy thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.embeddings import SemanticStubEmbedder, get_embedder, reset_embedder_cache
from sentinel.heuristics import f1_levenshtein, f2_jaccard, f3_gap, fuse
from sentinel.layer1 import layer1
from sentinel.layer2 import _cosine, _rescale_top1_to_confidence, phantom_signature, tool_signature, warm_up_registry
from sentinel.schemas import DetectRequest, Tool, ToolRegistry

# Mirror of run_bench._BENCH_REGISTRY_TOOLS — must stay in sync with corpus legal calls
from bench.run_bench import _BENCH_REGISTRY_TOOLS as _REGISTRY_TOOLS  # noqa: E402


def _run_corpus(
    corpus_path: Path,
    registry: ToolRegistry,
    embedder: SemanticStubEmbedder,
    registry_embeddings: dict[str, list[float]],
    auto_correct_min: float,
    block_max: float,
    fusion_weights: tuple[float, float, float, float] = (0.5, 0.2, 0.2, 0.1),
    f3_multiplier: float = 5.0,
) -> dict:
    lines = [ln for ln in corpus_path.read_text().splitlines() if ln.strip()]
    correct = 0
    tp = fp = tn = fn = 0

    for raw in lines:
        entry = json.loads(raw)
        req = DetectRequest(
            tool_name=entry["tool_name"],
            tool_input=entry.get("tool_input", {}),
            agent_reasoning=entry.get("agent_reasoning"),
            session_id="calibrate",
        )
        expected = entry["expected_verdict"]

        # L1 — exact match
        d1 = layer1(req.tool_name, registry)
        if d1 is not None:
            got = "ALLOW"
        else:
            # L2 — embed + cosine + F-fusion
            phantom_vec = embedder.embed(phantom_signature(req))
            scored = [
                (_cosine(phantom_vec, registry_embeddings[t.name]), t)
                for t in registry.tools
                if t.name in registry_embeddings
            ]
            scored.sort(reverse=True, key=lambda x: x[0])
            if not scored:
                got = "BLOCK"
            else:
                top3 = scored[:3]
                top1_sim, top1_tool = top3[0]
                top3_sims = [s for s, _ in top3]
                base_conf = _rescale_top1_to_confidence(top1_sim)
                feat_f1 = f1_levenshtein(req.tool_name, top1_tool.name)
                feat_f2 = f2_jaccard(set(req.tool_input.keys()), set(top1_tool.required_args))
                feat_f3 = f3_gap(top3_sims, multiplier=f3_multiplier)
                if block_max <= base_conf < auto_correct_min:
                    final_conf = fuse(base_conf, feat_f1, feat_f2, feat_f3, weights=fusion_weights)
                else:
                    final_conf = base_conf
                if final_conf >= auto_correct_min:
                    got = "AUTO_CORRECT"
                elif final_conf >= block_max:
                    got = "SUGGEST"
                else:
                    got = "BLOCK"

        if got == expected:
            correct += 1

        is_phantom_pred = got != "ALLOW"
        is_phantom_truth = entry["label"] == "phantom"
        if is_phantom_truth and is_phantom_pred:
            tp += 1
        elif is_phantom_truth and not is_phantom_pred:
            fn += 1
        elif not is_phantom_truth and is_phantom_pred:
            fp += 1
        else:
            tn += 1

    total = len(lines)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "auto_correct_min": auto_correct_min,
        "block_max": block_max,
        "accuracy": round(correct / total, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "correct": correct,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel threshold calibration")
    parser.add_argument("--dataset", type=Path, default=Path("../data/sentinel-bench-v1"))
    parser.add_argument("--write-best", action="store_true", help="Patch cascade.yaml with best thresholds")
    parser.add_argument("--config", type=Path, default=Path("../configs/cascade.yaml"))
    args = parser.parse_args()

    corpus_path = args.dataset / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("SENTINEL_EMBEDDER_PROVIDER", "stub")
    reset_embedder_cache()
    embedder = SemanticStubEmbedder()
    registry = ToolRegistry(tools=tuple(_REGISTRY_TOOLS), version="calibrate-v1")
    registry_embeddings = warm_up_registry(registry, embedder)

    acm_range = [round(v, 2) for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    bm_range  = [round(v, 2) for v in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]

    print(f"\n{'ACM':>6} {'BM':>6} {'Acc':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 52)

    best: dict | None = None
    all_results: list[dict] = []

    for acm in acm_range:
        for bm in bm_range:
            if bm >= acm:
                continue
            r = _run_corpus(corpus_path, registry, embedder, registry_embeddings, acm, bm)
            all_results.append(r)
            marker = ""
            if r["f1"] == 1.0:
                if best is None or r["accuracy"] > best["accuracy"]:
                    best = r
                    marker = " ← best"
            flag = "✓" if r["f1"] == 1.0 else "✗"
            print(f"{acm:>6.2f} {bm:>6.2f} {r['accuracy']:>7.1%} {flag}{r['f1']:>6.3f}{r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}{marker}")

    print()
    if best:
        print(f"Best (F1=1.000): auto_correct_min={best['auto_correct_min']} block_max={best['block_max']} accuracy={best['accuracy']:.1%}")
        if args.write_best and args.config.exists():
            data = yaml.safe_load(args.config.read_text())
            data["verdict_thresholds"]["auto_correct_min"] = best["auto_correct_min"]
            data["verdict_thresholds"]["block_max"] = best["block_max"]
            args.config.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
            print(f"Patched {args.config}")
    else:
        print("No configuration achieved F1=1.000 — check corpus or embedder.")


if __name__ == "__main__":
    main()
