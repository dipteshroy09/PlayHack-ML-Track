"""Break-even analysis: F1 weight at which the current threshold wins.

Solves for the F1 weight where the threshold choice flips, assuming timing
skills share the remaining weight equally.

Outputs:
    breakeven_analysis.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHART_PATH = Path("breakeven_analysis.png")

SWEEP = pd.read_csv("threshold_sweep.csv").set_index("threshold")
CURRENT = round(float(Path("task_a_threshold.txt").read_text(encoding="utf-8").strip()), 2)
CANDIDATES = [0.27, 0.28, 0.31, 0.33, 0.34, 0.36]


def row(threshold: float) -> pd.Series:
    return SWEEP.loc[round(threshold, 2)]


def main() -> None:
    base = row(CURRENT)
    print(f"current threshold {CURRENT:.2f}: F1={base['taskA_f1']:.4f} "
          f"onset={base['onset_skill']:.4f} recovery={base['recovery_skill']:.4f} "
          f"flagged={base['predicted_positive_rate']:.2%}\n")

    records = []
    for threshold in CANDIDATES:
        candidate = row(threshold)
        d_f1 = candidate["taskA_f1"] - base["taskA_f1"]
        d_timing = (
            (candidate["onset_skill"] - base["onset_skill"])
            + (candidate["recovery_skill"] - base["recovery_skill"])
        )
        # Solve w*d_f1 + ((1-w)/2)*d_timing = 0 for the flip point.
        denominator = d_timing / 2 - d_f1
        breakeven = (d_timing / 2) / denominator if denominator != 0 else float("nan")
        records.append(
            {
                "threshold": threshold,
                "F1": candidate["taskA_f1"],
                "onset": candidate["onset_skill"],
                "recovery": candidate["recovery_skill"],
                "flagged": candidate["predicted_positive_rate"],
                "F1_lost": -d_f1,
                "timing_gained": d_timing,
                "net_raw_points": d_f1 + d_timing,
                "breakeven_F1_weight": breakeven,
            }
        )

    frame = pd.DataFrame(records)
    print("Switching from the current threshold to each candidate:")
    print(frame.round(4).to_string(index=False))
    print(
        "\nbreakeven_F1_weight = the share of the total score that Task A F1 must carry"
        f"\nfor {CURRENT:.2f} to remain the better choice. Above it, keep {CURRENT:.2f}; below it, switch."
    )

    write_chart(frame, CURRENT)
    print(f"\nSaved {CHART_PATH}")


def write_chart(frame: pd.DataFrame, current: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    thresholds = frame["threshold"].to_numpy()
    labels = [f"{t:.2f}" for t in thresholds]
    x = np.arange(len(thresholds))
    width = 0.35

    # Panel 1: F1 lost vs timing gained
    bars_loss = ax1.bar(x - width / 2, frame["F1_lost"], width,
                        color="#ef4444", edgecolor="#ffffff22", label="F1 Lost")
    bars_gain = ax1.bar(x + width / 2, frame["timing_gained"], width,
                        color="#22c55e", edgecolor="#ffffff22", label="Timing Gained")
    for bar in bars_loss:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=8, color="#ef4444")
    for bar in bars_gain:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=8, color="#22c55e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Candidate Threshold", fontsize=11)
    ax1.set_ylabel("Metric Change", fontsize=11)
    ax1.set_title(f"F1 Lost vs Timing Gained (vs current {current:.2f})",
                  fontsize=12, fontweight="bold", color="white")
    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax1.grid(axis="y", alpha=0.15, color="#666")

    # Panel 2: Break-even F1 weight
    be = frame["breakeven_F1_weight"].to_numpy()
    colors = ["#3b82f6" if not np.isnan(v) and 0 < v < 1 else "#6b7280" for v in be]
    bars = ax2.barh(labels, np.where(np.isnan(be), 0, be), color=colors,
                    edgecolor="#ffffff22", height=0.5)
    for i, (bar, val) in enumerate(zip(bars, be)):
        if not np.isnan(val):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2%}", ha="left", va="center",
                     fontsize=10, color="#eeeeee", fontweight="bold")
    ax2.axvline(x=0.333, color="#f59e0b", linestyle="--", alpha=0.6,
                label="Equal weight (33%)")
    ax2.axvline(x=0.50, color="#a855f7", linestyle="--", alpha=0.6,
                label="Majority F1 (50%)")
    ax2.set_xlabel("Break-Even F1 Weight", fontsize=11)
    ax2.set_ylabel("Candidate Threshold", fontsize=11)
    ax2.set_title("Break-Even F1 Weight to Keep Current Threshold",
                  fontsize=12, fontweight="bold", color="white")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc",
               loc="lower right")
    ax2.grid(axis="x", alpha=0.15, color="#666")
    ax2.set_xlim(0, 1.0)

    fig.suptitle("Break-Even Threshold Analysis", fontsize=14, fontweight="bold",
                 color="white", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
