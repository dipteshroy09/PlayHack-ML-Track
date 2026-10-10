"""Locate the Task A threshold achieving a target OOF recall.

Prints the full scorecard at the selected and comparison points.

Outputs:
    recall_sweep.png   metrics across the recall sweep grid
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_oof import score_oof
import scoring

CHART_PATH = Path("recall_sweep.png")

TARGET_RECALL = 0.90
OOF_PATH = "oof_predictions.csv"
COMPARISONS = [0.52, 0.50]


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    truth = oof[scoring.LABEL_A].astype(int).to_numpy()
    proba = oof["proba_injured"].to_numpy()

    injured_proba = np.sort(proba[truth == 1])
    # Highest threshold whose predictions still capture TARGET_RECALL of injured.
    index = int(np.floor((1.0 - TARGET_RECALL) * len(injured_proba)))
    exact = float(injured_proba[index])
    rounded = float(np.floor(exact * 100) / 100)

    print(f"injured athletes            : {len(injured_proba)}")
    print(f"exact threshold for {TARGET_RECALL:.0%} recall : {exact:.6f}")
    print(f"rounded down to 2 decimals  : {rounded:.2f}")

    rows = []
    for label, threshold in [
        ("target_recall_exact", exact),
        ("target_recall_rounded", rounded),
        *[(f"comparison_{t:.2f}", t) for t in COMPARISONS],
    ]:
        result = score_oof(oof, threshold)
        result["label"] = label
        result["predicted_positive_rate"] = float((proba >= threshold).mean())
        rows.append(result)

    frame = pd.DataFrame(rows)[
        [
            "label", "threshold", "taskA_f1", "precision", "recall",
            "true_positive", "false_negative", "false_positive", "true_negative",
            "onset_skill", "recovery_skill", "onset_mae_model", "recovery_mae_model",
            "predicted_positive_rate",
        ]
    ]
    print("\n=== official scorecard at each candidate ===")
    print(frame.round(4).to_string(index=False))

    print("\n=== recall sweep near the target ===")
    grid = np.round(np.arange(0.30, 0.45, 0.01), 2)
    near = []
    for threshold in grid:
        result = score_oof(oof, float(threshold))
        near.append(
            {
                "threshold": threshold,
                "recall": result["recall"],
                "taskA_f1": result["taskA_f1"],
                "precision": result["precision"],
                "onset_skill": result["onset_skill"],
                "recovery_skill": result["recovery_skill"],
                "flagged": float((proba >= threshold).mean()),
            }
        )
    near_df = pd.DataFrame(near)
    print(near_df.round(4).to_string(index=False))

    write_chart(near_df, rounded)
    print(f"\nSaved {CHART_PATH}")


def write_chart(near_df: pd.DataFrame, selected_threshold: float) -> None:
    thresholds = near_df["threshold"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    # Panel 1: Classification metrics
    ax1.plot(thresholds, near_df["taskA_f1"], "o-", color="#3b82f6",
             linewidth=2.2, markersize=6, label="F1", zorder=5)
    ax1.plot(thresholds, near_df["recall"], "s-", color="#22c55e",
             linewidth=2, markersize=5, label="Recall", alpha=0.85)
    ax1.plot(thresholds, near_df["precision"], "^-", color="#f59e0b",
             linewidth=2, markersize=5, label="Precision", alpha=0.85)
    ax1.axhline(y=TARGET_RECALL, color="#ef4444", linestyle=":", alpha=0.5,
                label=f"Target recall = {TARGET_RECALL:.0%}")
    ax1.axvline(x=selected_threshold, color="#8b5cf6", linestyle="--", alpha=0.6,
                linewidth=1.5, label=f"Selected = {selected_threshold:.2f}")
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Classification Metrics Across Recall Sweep",
                  fontsize=12, fontweight="bold", color="white")
    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555",
               labelcolor="#ccc", ncol=2, loc="lower left")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.15, color="#666")

    # Panel 2: Timing skills
    ax2.plot(thresholds, near_df["onset_skill"], "D-", color="#06b6d4",
             linewidth=2.2, markersize=6, label="Onset Skill", zorder=5)
    ax2.plot(thresholds, near_df["recovery_skill"], "p-", color="#d946ef",
             linewidth=2.2, markersize=6, label="Recovery Skill")
    ax2.axvline(x=selected_threshold, color="#8b5cf6", linestyle="--", alpha=0.6,
                linewidth=1.5, label=f"Selected = {selected_threshold:.2f}")
    ax2.set_xlabel("Threshold", fontsize=11)
    ax2.set_ylabel("Skill Score", fontsize=11)
    ax2.set_title("Timing Skills Across Recall Sweep",
                  fontsize=12, fontweight="bold", color="white")
    ax2.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.15, color="#666")

    fig.suptitle(f"Recall Sweep Analysis  (target = {TARGET_RECALL:.0%})",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
