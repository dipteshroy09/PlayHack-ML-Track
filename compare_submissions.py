"""Side-by-side comparison of submission variants under plausible score aggregations.

Outputs:
    submission_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHART_PATH = Path("submission_comparison.png")

SWEEP = pd.read_csv("threshold_sweep.csv").set_index("threshold")
ARTIFACTS = Path("submission_artifacts")
CURRENT = round(float(Path("task_a_threshold.txt").read_text(encoding="utf-8").strip()), 2)

VARIANTS = {
    0.50: ARTIFACTS / "submission_threshold_0.50.csv",
    0.34: Path("submission.csv") if CURRENT == 0.34 else ARTIFACTS / "submission_threshold_0.34.csv",
    0.28: ARTIFACTS / "submission_threshold_0.28.csv",
}

SCENARIOS = {
    "f1_only": (1.00, 0.00, 0.00),
    "f1_heavy": (0.70, 0.15, 0.15),
    "equal_thirds": (1 / 3, 1 / 3, 1 / 3),
    "timing_heavy": (0.20, 0.40, 0.40),
}


def main() -> None:
    rows = []
    for threshold, path in sorted(VARIANTS.items(), reverse=True):
        metrics = SWEEP.loc[threshold]
        record = {
            "threshold": threshold,
            "F1": metrics["taskA_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "onset_skill": metrics["onset_skill"],
            "recovery_skill": metrics["recovery_skill"],
            "oof_flagged": metrics["predicted_positive_rate"],
        }
        if path.exists():
            submission = pd.read_csv(path)
            record["test_flagged"] = float(submission["injured_in_risk_window"].mean())
            record["file"] = path.name
        else:
            record["test_flagged"] = float("nan")
            record["file"] = "not generated"
        for name, (w_f1, w_on, w_rec) in SCENARIOS.items():
            record[name] = (
                w_f1 * metrics["taskA_f1"]
                + w_on * metrics["onset_skill"]
                + w_rec * metrics["recovery_skill"]
            )
        rows.append(record)

    frame = pd.DataFrame(rows)

    print("=== official metrics (validated OOF) and realized flag rate ===")
    print(
        frame[
            ["threshold", "F1", "precision", "recall", "onset_skill",
             "recovery_skill", "oof_flagged", "test_flagged", "file"]
        ].round(4).to_string(index=False)
    )

    print("\n=== score under each candidate aggregation (none official) ===")
    view = frame[["threshold", *SCENARIOS]].copy()
    print(view.round(4).to_string(index=False))

    print("\nwinner per scenario:")
    for name in SCENARIOS:
        best = frame.loc[frame[name].idxmax()]
        print(f"  {name:14s} -> {best['threshold']:.2f}")

    # Worst-case regret across the scenario set, restricted to these variants.
    regrets = {}
    for _, row in frame.iterrows():
        regrets[row["threshold"]] = max(
            frame[name].max() - row[name] for name in SCENARIOS
        )
    print("\nworst-case regret among these three variants:")
    for threshold, regret in sorted(regrets.items(), reverse=True):
        marker = "  <- current" if threshold == CURRENT else ""
        print(f"  {threshold:.2f} : {regret:.4f}{marker}")
    best_threshold = min(regrets, key=regrets.get)
    print(f"\nminimax-regret choice among the three: {best_threshold:.2f}")

    write_chart(frame, regrets, CURRENT)
    print(f"\nSaved {CHART_PATH}")


def write_chart(frame: pd.DataFrame, regrets: dict, current: float) -> None:
    scenario_names = list(SCENARIOS.keys())
    pretty_names = [n.replace("_", " ").title() for n in scenario_names]
    thresholds = frame["threshold"].to_numpy()
    labels = []
    for t in thresholds:
        lbl = f"{t:.2f}"
        if np.isclose(t, current):
            lbl += " (current)"
        labels.append(lbl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    # Panel 1: Grouped bars per scenario
    x = np.arange(len(scenario_names))
    width = 0.25
    colors = ["#3b82f6", "#22c55e", "#f59e0b"]
    for i, (t, lbl) in enumerate(zip(thresholds, labels)):
        vals = [float(frame.loc[frame["threshold"] == t, s].iloc[0]) for s in scenario_names]
        bars = ax1.bar(x + (i - 1) * width, vals, width, color=colors[i],
                       edgecolor="#ffffff22", label=lbl)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                     color=colors[i], fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(pretty_names, fontsize=10, rotation=15, ha="right")
    ax1.set_ylabel("Composite Score", fontsize=11)
    ax1.set_title("Score Under Each Aggregation Scenario",
                  fontsize=12, fontweight="bold", color="white")
    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax1.grid(axis="y", alpha=0.15, color="#666")

    # Panel 2: Worst-case regret
    regret_vals = [regrets.get(t, 0) for t in thresholds]
    bar_colors = ["#ef4444" if r == max(regret_vals) else "#6366f1" for r in regret_vals]
    bars2 = ax2.barh(labels, regret_vals, color=bar_colors, edgecolor="#ffffff22", height=0.5)
    for bar, val in zip(bars2, regret_vals):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", ha="left", va="center",
                 fontsize=10, color="#eeeeee", fontweight="bold")
    ax2.set_xlabel("Max Regret", fontsize=11)
    ax2.set_title("Worst-Case Regret",
                  fontsize=12, fontweight="bold", color="white")
    ax2.grid(axis="x", alpha=0.15, color="#666")

    fig.suptitle("Submission Variant Comparison", fontsize=14, fontweight="bold",
                 color="white", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
