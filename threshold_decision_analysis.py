"""Worst-case regret analysis for threshold selection.

Calculates the regret (shortfall from the best possible threshold) across
four plausible scoring aggregations, reporting the minimax-regret choice.

Writes one report; never modifies task_a_threshold.txt.

Outputs:
    threshold_decision_report.md
    regret_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHART_PATH = Path("regret_heatmap.png")

SWEEP_PATH = Path("threshold_sweep.csv")
REPORT_PATH = Path("threshold_decision_report.md")
CURRENT = float(Path("task_a_threshold.txt").read_text(encoding="utf-8").strip())

SCENARIOS = {
    "f1_only": (1.00, 0.00, 0.00),
    "f1_heavy": (0.70, 0.15, 0.15),
    "equal_thirds": (1 / 3, 1 / 3, 1 / 3),
    "timing_heavy": (0.20, 0.40, 0.40),
}


def main() -> None:
    sweep = pd.read_csv(SWEEP_PATH)
    table = sweep[["threshold", "taskA_f1", "onset_skill", "recovery_skill", "recall",
                   "predicted_positive_rate"]].copy()

    regrets = pd.DataFrame({"threshold": table["threshold"]})
    for name, (w_f1, w_onset, w_recovery) in SCENARIOS.items():
        value = (
            w_f1 * table["taskA_f1"]
            + w_onset * table["onset_skill"]
            + w_recovery * table["recovery_skill"]
        )
        regrets[name] = value.max() - value
    regrets["max_regret"] = regrets[list(SCENARIOS)].max(axis=1)

    merged = table.merge(regrets, on="threshold")
    best = merged.loc[merged["max_regret"].idxmin()]
    current = merged.loc[(merged["threshold"] - CURRENT).abs().idxmin()]

    per_scenario = {
        name: merged.loc[merged[name].idxmin(), "threshold"] for name in SCENARIOS
    }

    columns = ["threshold", "taskA_f1", "onset_skill", "recovery_skill", "recall",
               "predicted_positive_rate", *SCENARIOS, "max_regret"]
    shortlist = merged.sort_values("max_regret").head(10)[columns]

    print("optimal threshold per scenario:")
    for name, threshold in per_scenario.items():
        print(f"  {name:14s} -> {threshold:.2f}")
    print(f"\nminimax-regret threshold : {best['threshold']:.2f} (max regret {best['max_regret']:.4f})")
    print(f"current threshold {CURRENT:.2f}    : max regret {current['max_regret']:.4f}")
    print("\ntop 10 thresholds by worst-case regret:")
    print(shortlist.round(4).to_string(index=False))

    lines = [
        "# Threshold Choice Under Unknown Aggregation",
        "",
        "The challenge PDF (page 6) defines Task A F1, onset skill, and recovery skill",
        "separately and never states how they combine. Organizer clarification is not",
        "available, so the operating point is chosen under uncertainty.",
        "",
        "Metrics come from the validated leakage-free OOF sweep (`threshold_sweep.csv`).",
        "Regret is the shortfall against the best threshold *within each scenario*.",
        "",
        "## Optimal threshold per scenario",
        "",
        "| Scenario | Weights (F1 / onset / recovery) | Best threshold |",
        "| --- | --- | --- |",
    ]
    for name, weights in SCENARIOS.items():
        weight_text = " / ".join(f"{w:.2f}" for w in weights)
        lines.append(f"| {name} | {weight_text} | {per_scenario[name]:.2f} |")
    lines += [
        "",
        "## Worst-case regret shortlist",
        "",
        "| Threshold | F1 | Onset | Recovery | Recall | Flagged | Max regret |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in shortlist.iterrows():
        lines.append(
            f"| {row['threshold']:.2f} | {row['taskA_f1']:.4f} | {row['onset_skill']:.4f} "
            f"| {row['recovery_skill']:.4f} | {row['recall']:.4f} "
            f"| {row['predicted_positive_rate']:.4f} | {row['max_regret']:.4f} |"
        )
    lines += [
        "",
        f"- Minimax-regret threshold: **{best['threshold']:.2f}** (worst-case regret {best['max_regret']:.4f}).",
        f"- Current saved threshold {CURRENT:.2f} has worst-case regret {current['max_regret']:.4f}.",
        "",
        "This is a decision aid, not an official score. `task_a_threshold.txt` is unchanged.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT_PATH}")

    write_chart(shortlist, float(best["threshold"]), CURRENT)
    print(f"Saved {CHART_PATH}")


def write_chart(shortlist: pd.DataFrame, best_threshold: float, current: float) -> None:
    scenario_names = list(SCENARIOS.keys())
    pretty_names = [n.replace("_", " ").title() for n in scenario_names]
    thresholds = shortlist["threshold"].to_numpy()
    data = shortlist[scenario_names].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Regret", color="#cccccc", fontsize=11)
    cbar.ax.tick_params(colors="#cccccc")

    ax.set_xticks(range(len(pretty_names)))
    ax.set_xticklabels(pretty_names, fontsize=10, color="#cccccc", rotation=25, ha="right")
    ylabels = []
    for t in thresholds:
        label = f"{t:.2f}"
        if np.isclose(t, best_threshold):
            label += " ★"
        if np.isclose(t, current):
            label += " (current)"
        ylabels.append(label)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels(ylabels, fontsize=10, color="#cccccc")

    for i in range(len(thresholds)):
        for j in range(len(scenario_names)):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.5 else "#1a1a2e"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xlabel("Scoring Scenario", fontsize=11, color="#cccccc")
    ax.set_ylabel("Threshold", fontsize=11, color="#cccccc")
    ax.set_title("Worst-Case Regret Heatmap  (★ = minimax-regret choice)",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    for spine in ax.spines.values():
        spine.set_color("#333355")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
