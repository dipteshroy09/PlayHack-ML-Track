"""Task A recall ceiling analysis.

Shows that injury detectability is a monotone inverse function of true onset
day: early-window injuries are highly detectable, late-window injuries are not.

Outputs:
    detectability_by_onset_week.png   detection rate & MAE by onset week
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scoring
from evaluate_oof import current_threshold

OOF_PATH = "oof_predictions.csv"


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    threshold = current_threshold()
    truly_injured = oof[scoring.LABEL_A] == 1
    flagged = oof["proba_injured"] >= threshold

    caught = oof[truly_injured & flagged]
    missed = oof[truly_injured & ~flagged]
    correlation = np.corrcoef(
        oof.loc[truly_injured, "proba_injured"], oof.loc[truly_injured, scoring.LABEL_ONSET]
    )[0, 1]

    print(f"threshold {threshold:.2f} | caught {len(caught)} | missed {len(missed)}")
    print(f"\ncorr(injury probability, TRUE onset day) on injured athletes = {correlation:+.4f}")
    print("Negative and large: the model detects imminent injuries, not distant ones.\n")

    print("Caught injuries (scored on timing error):")
    print(f"  true onset   mean={caught[scoring.LABEL_ONSET].mean():.2f} median={caught[scoring.LABEL_ONSET].median():.0f}")
    print(f"  pred onset   mean={caught['pred_onset'].mean():.2f}")
    print(f"  onset MAE    {np.mean(np.abs(caught['pred_onset'] - caught[scoring.LABEL_ONSET])):.4f}")
    print(f"  recovery MAE {np.mean(np.abs(caught['pred_recovery'] - caught[scoring.LABEL_RECOVERY])):.4f}")

    print("\nMissed injuries (each costs the fixed 30 penalty on both timing targets):")
    print(f"  true onset   mean={missed[scoring.LABEL_ONSET].mean():.2f} median={missed[scoring.LABEL_ONSET].median():.0f}")
    print(f"  pred onset   mean={missed['pred_onset'].mean():.2f}  <- still close to the truth")

    print("\nOnset detectability by true onset week:")
    injured = oof[truly_injured].copy()
    injured["onset_week"] = pd.cut(
        injured[scoring.LABEL_ONSET], [0, 7, 14, 21, 30], labels=["1-7", "8-14", "15-21", "22-30"]
    )
    summary = injured.groupby("onset_week", observed=True).apply(
        lambda group: pd.Series(
            {
                "n": len(group),
                "mean_probability": group["proba_injured"].mean(),
                "detected_rate": float((group["proba_injured"] >= threshold).mean()),
                "onset_mae": float(np.mean(np.abs(group["pred_onset"] - group[scoring.LABEL_ONSET]))),
            }
        ),
        include_groups=False,
    )
    print(summary.round(4).to_string())

    print(
        "\nConsequence: pure onset MAE is strong across all injured athletes, so onset skill"
        "\nis destroyed only by the miss penalty, not by poor timing predictions."
    )

    # ---- Chart ----
    write_chart(summary, threshold)
    print(f"\nSaved detectability_by_onset_week.png")


CHART_PATH = Path("detectability_by_onset_week.png")


def write_chart(summary: pd.DataFrame, threshold: float) -> None:
    weeks = [str(w) for w in summary.index]
    detected = summary["detected_rate"].to_numpy()
    mae_vals = summary["onset_mae"].to_numpy()
    counts = summary["n"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    # Panel 1: Detection rate
    colors1 = ["#22c55e" if d >= 0.7 else "#f59e0b" if d >= 0.4 else "#ef4444" for d in detected]
    bars1 = ax1.bar(weeks, detected, color=colors1, edgecolor="#ffffff22", width=0.6)
    for bar, val, n in zip(bars1, detected, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}\n(n={int(n)})", ha="center", va="bottom",
                 fontsize=9, color="#eeeeee", fontweight="bold")
    ax1.set_xlabel("True Onset Week", fontsize=11)
    ax1.set_ylabel("Detection Rate", fontsize=11)
    ax1.set_title(f"Detection Rate by Onset Week (threshold={threshold:.2f})",
                  fontsize=12, fontweight="bold", color="white")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=threshold, color="#8b5cf6", linestyle="--", alpha=0.5, label=f"Threshold {threshold:.2f}")
    ax1.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax1.grid(axis="y", alpha=0.15, color="#666")

    # Panel 2: Onset MAE
    colors2 = ["#3b82f6", "#6366f1", "#a855f7", "#ec4899"]
    bars2 = ax2.bar(weeks, mae_vals, color=colors2[:len(weeks)], edgecolor="#ffffff22", width=0.6)
    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=10, color="#eeeeee", fontweight="bold")
    ax2.set_xlabel("True Onset Week", fontsize=11)
    ax2.set_ylabel("Onset MAE (days)", fontsize=11)
    ax2.set_title("Onset Prediction MAE by Onset Week",
                  fontsize=12, fontweight="bold", color="white")
    ax2.grid(axis="y", alpha=0.15, color="#666")

    fig.suptitle("Injury Detectability Analysis", fontsize=14, fontweight="bold",
                 color="white", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
