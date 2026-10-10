"""Recovery duration regressor diagnostics.

Establishes achievable floors: constant-predictor baselines, target shape,
feature signal strength, and OOF error profile.

Outputs:
    recovery_diagnostics.png   distribution + MAE by recovery bucket
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import scoring

CHART_PATH = Path("recovery_diagnostics.png")

SEED = 42
N_SPLITS = 5
TARGET = scoring.LABEL_RECOVERY
LABEL_A = scoring.LABEL_A
ID = "athlete_id"


def mae(truth: np.ndarray, prediction) -> float:
    return float(np.mean(np.abs(np.asarray(prediction, dtype=float) - truth)))


def main() -> None:
    data = pd.read_csv("train_features.csv")
    oof = pd.read_csv("oof_predictions.csv")
    injured = data[data[LABEL_A] == 1].copy()
    truth = injured[TARGET].to_numpy(dtype=float)

    print("=== target shape (injured training athletes) ===")
    print(f"rows            : {len(injured)}")
    print(f"mean / median   : {truth.mean():.4f} / {np.median(truth):.4f}")
    print(f"std             : {truth.std(ddof=1):.4f}")
    print(f"min / max       : {truth.min():.0f} / {truth.max():.0f}")
    print(f"integer-valued  : {bool(np.all(truth == np.rint(truth)))}")
    quantiles = np.percentile(truth, [1, 5, 25, 50, 75, 95, 99])
    print("percentiles 1/5/25/50/75/95/99:", np.round(quantiles, 2).tolist())
    print("\nvalue counts (top 12):")
    print(injured[TARGET].value_counts().head(12).to_string())

    print("\n=== constant-predictor floors on full training injured set ===")
    print(f"MAE @ mean            : {mae(truth, truth.mean()):.4f}   <- PDF baseline rule")
    print(f"MAE @ median          : {mae(truth, np.median(truth)):.4f}")
    print(f"MAE @ rounded median  : {mae(truth, np.rint(np.median(truth))):.4f}")
    grid = np.arange(truth.min(), truth.max() + 0.5, 0.5)
    best = min(grid, key=lambda value: mae(truth, value))
    print(f"MAE @ best constant   : {mae(truth, best):.4f} (at {best:.1f})")

    print("\n=== leakage-free constant baselines using the locked folds ===")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    mean_errors: list[float] = []
    median_errors: list[float] = []
    for train_idx, valid_idx in splitter.split(data, data[LABEL_A]):
        train_injured = data.iloc[train_idx]
        train_injured = train_injured[train_injured[LABEL_A] == 1][TARGET]
        valid_injured = data.iloc[valid_idx]
        valid_injured = valid_injured[valid_injured[LABEL_A] == 1][TARGET].to_numpy(dtype=float)
        mean_errors.extend(np.abs(valid_injured - train_injured.mean()).tolist())
        median_errors.extend(np.abs(valid_injured - train_injured.median()).tolist())
    baseline_mean_mae = float(np.mean(mean_errors))
    baseline_median_mae = float(np.mean(median_errors))
    print(f"OOF MAE @ fold train mean   : {baseline_mean_mae:.4f}  (official baseline)")
    print(f"OOF MAE @ fold train median : {baseline_median_mae:.4f}")
    print(f"skill of median-only model  : {max(0.0, 1 - baseline_median_mae / baseline_mean_mae):.4f}")

    print("\n=== existing OOF recovery model ===")
    oof_injured = oof[oof[LABEL_A] == 1]
    oof_truth = oof_injured[TARGET].to_numpy(dtype=float)
    oof_pred = oof_injured["pred_recovery"].to_numpy(dtype=float)
    model_mae = mae(oof_truth, oof_pred)
    print(f"pure MAE (rounded, saved)   : {model_mae:.4f}")
    print(f"skill vs official baseline  : {max(0.0, 1 - model_mae / baseline_mean_mae):.4f}")
    print(f"prediction mean / std       : {oof_pred.mean():.3f} / {oof_pred.std(ddof=1):.3f}")
    print(f"prediction min / max        : {oof_pred.min():.0f} / {oof_pred.max():.0f}")
    print(f"truth std / pred std ratio  : {oof_truth.std(ddof=1) / oof_pred.std(ddof=1):.3f}")
    print(f"corr(pred, truth)           : {np.corrcoef(oof_pred, oof_truth)[0, 1]:.4f}")
    residual = oof_truth - oof_pred
    print(f"residual mean (bias)        : {residual.mean():+.4f}")
    print("\nMAE by true-recovery bucket:")
    buckets = pd.cut(oof_truth, [0, 3, 5, 7, 10, 14, 100], right=True)
    frame = pd.DataFrame({"bucket": buckets, "abs_err": np.abs(residual), "truth": oof_truth})
    print(
        frame.groupby("bucket", observed=True)
        .agg(n=("truth", "size"), mean_truth=("truth", "mean"), mae=("abs_err", "mean"))
        .round(3)
        .to_string()
    )

    print("\n=== strongest linear signals vs recovery_duration (injured rows) ===")
    numeric = injured.select_dtypes(include=[np.number]).drop(
        columns=[ID, LABEL_A, TARGET, scoring.LABEL_ONSET], errors="ignore"
    )
    correlations = numeric.corrwith(injured[TARGET]).dropna().sort_values(
        key=lambda series: series.abs(), ascending=False
    )
    print(correlations.head(15).round(4).to_string())
    print(f"\nfeatures with |corr| > 0.10 : {int((correlations.abs() > 0.10).sum())}")
    print(f"features with |corr| > 0.05 : {int((correlations.abs() > 0.05).sum())}")
    print(
        "corr(onset_day_offset, recovery) : "
        f"{injured[[scoring.LABEL_ONSET, TARGET]].corr().iloc[0, 1]:.4f}"
    )

    # ---- Chart ----
    write_chart(truth, oof_truth, oof_pred, model_mae, baseline_mean_mae)
    print(f"\nSaved {CHART_PATH}")


def write_chart(
    truth: np.ndarray,
    oof_truth: np.ndarray,
    oof_pred: np.ndarray,
    model_mae: float,
    baseline_mae: float,
) -> None:
    residual = oof_truth - oof_pred
    buckets = pd.cut(oof_truth, [0, 3, 5, 7, 10, 14, 100], right=True)
    frame = pd.DataFrame({"bucket": buckets, "abs_err": np.abs(residual), "truth": oof_truth})
    bucket_stats = (
        frame.groupby("bucket", observed=True)
        .agg(n=("truth", "size"), mean_truth=("truth", "mean"), mae=("abs_err", "mean"))
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    # Panel 1: Distribution histogram
    ax1.hist(truth, bins=30, color="#6366f1", edgecolor="#1a1a2e", alpha=0.85)
    ax1.axvline(truth.mean(), color="#f59e0b", linestyle="--", linewidth=1.5,
                label=f"Mean = {truth.mean():.1f}")
    ax1.axvline(np.median(truth), color="#22c55e", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(truth):.1f}")
    ax1.set_xlabel("Recovery Duration (days)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Recovery Duration Distribution (Injured Athletes)",
                  fontsize=12, fontweight="bold", color="white")
    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax1.grid(axis="y", alpha=0.15, color="#666")

    # Panel 2: MAE by bucket
    labels = [str(b) for b in bucket_stats.index]
    mae_vals = bucket_stats["mae"].to_numpy()
    counts = bucket_stats["n"].to_numpy()
    gradient = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899"]
    bars = ax2.bar(labels, mae_vals, color=gradient[:len(labels)],
                   edgecolor="#ffffff22", width=0.6)
    for bar, val, n in zip(bars, mae_vals, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}\n(n={int(n)})", ha="center", va="bottom",
                 fontsize=9, color="#eeeeee", fontweight="bold")
    ax2.axhline(y=model_mae, color="#ef4444", linestyle="--", alpha=0.6,
                label=f"Overall MAE = {model_mae:.2f}")
    ax2.axhline(y=baseline_mae, color="#f59e0b", linestyle=":", alpha=0.6,
                label=f"Baseline MAE = {baseline_mae:.2f}")
    ax2.set_xlabel("Recovery Duration Bucket", fontsize=11)
    ax2.set_ylabel("MAE (days)", fontsize=11)
    ax2.set_title("OOF Recovery MAE by True Duration Bucket",
                  fontsize=12, fontweight="bold", color="white")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")
    ax2.grid(axis="y", alpha=0.15, color="#666")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    skill = max(0.0, 1 - model_mae / baseline_mae)
    fig.suptitle(f"Recovery Regressor Diagnostics  (skill = {skill:.4f})",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
