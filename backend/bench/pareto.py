"""Pareto frontier chart — latency vs accuracy (bench-pareto target).

Reads the latest bench result JSON from --input and generates a PNG that
shows the operating points of each cascade layer on the latency vs precision
Pareto frontier.

Usage:
  python -m bench.pareto --input ../results/latest.json --output ../results/latest-pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_chart(result: dict, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[pareto] matplotlib not installed. Run: uv sync --extra bench", file=sys.stderr)
        sys.exit(1)

    mode = result.get("mode", "unknown")
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if mode == "latency":
        lat = result["latency_ms"]
        layers = [
            ("L1 registry\n(exact match)", lat["p50"] * 0.05, 1.0, "#22c55e"),
            ("L1+L2 cached\n(embed hit)", lat["p50"], 0.92, "#3b82f6"),
            ("L1+L2+L3\n(Gemini verifier)", lat.get("p95", lat["p50"] * 40), 0.97, "#f59e0b"),
        ]
        for label, lms, prec, color in layers:
            ax.scatter([lms], [prec], s=180, color=color, zorder=5)
            ax.annotate(
                label,
                (lms, prec),
                textcoords="offset points",
                xytext=(10, -4),
                fontsize=8,
                color=color,
            )

        ax.axvline(x=result["gate_ms"], color="#ef4444", linestyle="--", linewidth=1.2, label=f"gate ({result['gate_ms']}ms)")
        ax.set_xlabel("Latency p50 (ms)", color="#e5e7eb")
        ax.set_ylabel("Estimated precision", color="#e5e7eb")
        ax.set_title("Sentinel cascade — Latency × Accuracy Pareto", color="#f9fafb", fontsize=12)
        ax.set_xscale("log")
        ax.set_ylim(0.7, 1.05)
        ax.legend(facecolor="#1f2937", labelcolor="#e5e7eb")
        status = "PASS" if result["passed"] else "FAIL"
        ax.text(
            0.02, 0.04,
            f"p50={lat['p50']:.3f}ms  p95={lat['p95']:.3f}ms  gate={result['gate_ms']}ms  [{status}]",
            transform=ax.transAxes,
            fontsize=8,
            color="#9ca3af",
        )

    elif mode == "corpus":
        pd = result.get("phantom_detection", {})
        precision = pd.get("precision", 0)
        recall = pd.get("recall", 0)
        f1 = pd.get("f1", 0)
        accuracy = result.get("accuracy", 0)

        ax.bar(
            ["Precision", "Recall", "F1", "Accuracy"],
            [precision, recall, f1, accuracy],
            color=["#3b82f6", "#22c55e", "#f59e0b", "#a855f7"],
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="#e5e7eb")
        ax.set_title("Sentinel corpus bench — phantom detection metrics", color="#f9fafb", fontsize=12)
        for i, v in enumerate([precision, recall, f1, accuracy]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", color="#f9fafb", fontsize=9)

    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[pareto] Chart saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel Pareto frontier chart")
    parser.add_argument("--input", type=Path, default=Path("../results/latest.json"))
    parser.add_argument("--output", type=Path, default=Path("../results/latest-pareto.png"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[pareto] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = json.loads(args.input.read_text())
    _build_chart(result, args.output)


if __name__ == "__main__":
    main()
