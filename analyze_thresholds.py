"""Threshold sweep and Pareto analysis from leakage-free OOF predictions.

Reports the non-dominated Pareto frontier and operating-point candidates
across F1, onset skill, and recovery skill. Does not modify task_a_threshold.txt.

Outputs:
    threshold_sweep.csv, pareto_thresholds.csv, threshold_candidates.csv,
    threshold_report.md, threshold_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_oof import score_oof

OOF_PATH = Path("oof_predictions.csv")
SWEEP_PATH = Path("threshold_sweep.csv")
PARETO_PATH = Path("pareto_thresholds.csv")
CANDIDATES_PATH = Path("threshold_candidates.csv")
REPORT_PATH = Path("threshold_report.md")
CHART_PATH = Path("threshold_tradeoff.png")
CURRENT_THRESHOLD_PATH = Path("task_a_threshold.txt")

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
OFFICIAL_METRICS = ["taskA_f1", "onset_skill", "recovery_skill"]
RECALL_F1_RETENTION = 0.90

ROLE_DEFINITIONS = {
    "current_reference": "Current temporary threshold; included for comparison only.",
    "f1_focused": "Highest Task A F1; ties prefer better timing skill, then recall.",
    "balanced": (
        "Pareto point maximizing the worst relative attainment across the three "
        "PDF-defined metrics. Relative attainment divides each metric by its best "
        "observed OOF value; this is a decision aid, not an official formula."
    ),
    "timing_focused": (
        "Highest mean of onset and recovery skill; ties prefer higher F1. This "
        "protects Task B but is not an official combined score."
    ),
    "recall_focused": (
        f"Highest recall among thresholds retaining at least {RECALL_F1_RETENTION:.0%} "
        "of the best observed F1; ties prefer higher F1 and timing skill."
    ),
    "maximum_recall_endpoint": (
        "Maximum possible recall on the sweep, shown as an extreme endpoint rather "
        "than a recommended operating point."
    ),
}


def build_sweep(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        result = score_oof(oof, float(threshold))
        result["predicted_positive_rate"] = float(
            (oof["proba_injured"] >= threshold).mean()
        )
        result["timing_skill_mean"] = float(
            np.mean([result["onset_skill"], result["recovery_skill"]])
        )
        rows.append(result)
    sweep = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_f1 = float(sweep["taskA_f1"].max())
    sweep["f1_retention"] = sweep["taskA_f1"] / best_f1
    return sweep


def pareto_frontier(sweep: pd.DataFrame) -> pd.DataFrame:
    """Return thresholds not dominated on all three PDF-defined metrics."""
    values = sweep[OFFICIAL_METRICS].to_numpy(dtype=float)
    keep = np.ones(len(sweep), dtype=bool)
    eps = 1e-12
    for i, current in enumerate(values):
        at_least_as_good = np.all(values >= current - eps, axis=1)
        strictly_better = np.any(values > current + eps, axis=1)
        if np.any(at_least_as_good & strictly_better):
            keep[i] = False
    return sweep.loc[keep].copy().sort_values("threshold").reset_index(drop=True)


def best_row(frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> pd.Series:
    return frame.sort_values(sort_columns, ascending=ascending).iloc[0]


def select_candidates(
    sweep: pd.DataFrame,
    pareto: pd.DataFrame,
    current_threshold: float,
) -> pd.DataFrame:
    candidates: list[dict[str, object]] = []

    def add(role: str, row: pd.Series) -> None:
        record = row.to_dict()
        record["role"] = role
        record["definition"] = ROLE_DEFINITIONS[role]
        candidates.append(record)

    current_index = (sweep["threshold"] - current_threshold).abs().idxmin()
    add("current_reference", sweep.loc[current_index])

    f1_row = best_row(
        sweep,
        ["taskA_f1", "timing_skill_mean", "recall"],
        [False, False, False],
    )
    add("f1_focused", f1_row)

    # Maximin on relative attainable values to avoid scale bias.
    balanced_pool = pareto.copy()
    relative_columns = []
    for metric in OFFICIAL_METRICS:
        best = float(sweep[metric].max())
        relative = f"relative_{metric}"
        balanced_pool[relative] = balanced_pool[metric] / best if best > 0 else 1.0
        relative_columns.append(relative)
    balanced_pool["relative_min"] = balanced_pool[relative_columns].min(axis=1)
    balanced_pool["relative_mean"] = balanced_pool[relative_columns].mean(axis=1)
    balanced_row = best_row(
        balanced_pool,
        ["relative_min", "relative_mean", "taskA_f1"],
        [False, False, False],
    )
    add("balanced", balanced_row)

    timing_row = best_row(
        sweep,
        ["timing_skill_mean", "taskA_f1", "recall"],
        [False, False, False],
    )
    add("timing_focused", timing_row)

    recall_pool = sweep[sweep["f1_retention"] >= RECALL_F1_RETENTION].copy()
    recall_row = best_row(
        recall_pool,
        ["recall", "taskA_f1", "timing_skill_mean"],
        [False, False, False],
    )
    add("recall_focused", recall_row)

    maximum_recall_row = best_row(
        sweep,
        ["recall", "taskA_f1", "timing_skill_mean"],
        [False, False, False],
    )
    add("maximum_recall_endpoint", maximum_recall_row)

    result = pd.DataFrame(candidates)
    first = ["role", "definition", "threshold", *OFFICIAL_METRICS, "precision", "recall"]
    remaining = [column for column in result.columns if column not in first]
    return result[first + remaining]


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    headers = {
        "role": "Candidate",
        "threshold": "Threshold",
        "taskA_f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "onset_skill": "Onset skill",
        "recovery_skill": "Recovery skill",
        "predicted_positive_rate": "Predicted injured",
    }
    rows = ["| " + " | ".join(headers.get(c, c) for c in columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "role":
                values.append(str(value).replace("_", " ").title())
            elif column == "threshold":
                values.append(f"{float(value):.2f}")
            else:
                values.append(f"{float(value):.4f}")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    sweep: pd.DataFrame,
    pareto: pd.DataFrame,
    candidates: pd.DataFrame,
    current_threshold: float,
) -> None:
    columns = [
        "role",
        "threshold",
        "taskA_f1",
        "precision",
        "recall",
        "onset_skill",
        "recovery_skill",
        "predicted_positive_rate",
    ]
    candidate_table = markdown_table(candidates, columns)

    frontier_view = pareto[
        ["threshold", "taskA_f1", "precision", "recall", "onset_skill", "recovery_skill"]
    ].copy()
    frontier_view.insert(0, "role", "pareto")
    frontier_table = markdown_table(
        frontier_view,
        ["role", "threshold", "taskA_f1", "precision", "recall", "onset_skill", "recovery_skill"],
    )

    definitions = "\n".join(
        f"- **{role.replace('_', ' ').title()}:** {definition}"
        for role, definition in ROLE_DEFINITIONS.items()
    )
    report = f"""# Leakage-Free OOF Threshold and Pareto Report

## Scope

This report uses only `oof_predictions.csv`. Every athlete was predicted by models that did not train on that athlete. Task B skill uses the baseline mean learned from that athlete's outer-training fold.

The PDF defines **F1**, **onset skill**, and **recovery skill**, but does not define how they are weighted into one leaderboard number. Therefore this report presents trade-offs rather than claiming one official winner. The current threshold remains `{current_threshold:.2f}` and was not changed.

## Operating-point candidates

{candidate_table}

## Candidate definitions

{definitions}

## Pareto frontier

A threshold is on the Pareto frontier when no other tested threshold is at least as good on all three PDF-defined metrics and strictly better on at least one.

{frontier_table}

## Interpretation rules

- F1-focused is appropriate only if Task A F1 is the dominant priority.
- Timing-focused maximizes protection from the 30-point miss penalty but may create many false alarms.
- Recall-focused preserves at least 90% of maximum F1 while catching as many injuries as possible.
- Balanced is a transparent heuristic based on relative attainable performance; it is **not** an official competition score.
- The maximum-recall endpoint is shown to expose the extreme, not as an automatic recommendation.

## Artifacts

- `threshold_sweep.csv`: all 99 thresholds from 0.01 through 0.99.
- `pareto_thresholds.csv`: non-dominated thresholds.
- `threshold_candidates.csv`: candidate details and definitions.
- `threshold_tradeoff.png`: metric curves and candidate locations.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_chart(sweep: pd.DataFrame, candidates: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.plot(sweep["threshold"], sweep["taskA_f1"], label="Task A F1", linewidth=2.2)
    ax1.plot(sweep["threshold"], sweep["precision"], label="Precision", alpha=0.85)
    ax1.plot(sweep["threshold"], sweep["recall"], label="Recall", alpha=0.85)
    ax1.set_ylabel("Classification metric")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.25)

    ax2.plot(sweep["threshold"], sweep["onset_skill"], label="Onset skill", linewidth=2.2)
    ax2.plot(sweep["threshold"], sweep["recovery_skill"], label="Recovery skill", linewidth=2.2)
    ax2.set_xlabel("Task A probability threshold")
    ax2.set_ylabel("Timing skill")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    colors = {
        "current_reference": "black",
        "f1_focused": "tab:blue",
        "balanced": "tab:purple",
        "timing_focused": "tab:orange",
        "recall_focused": "tab:green",
        "maximum_recall_endpoint": "tab:red",
    }
    legend_roles: set[str] = set()
    for _, row in candidates.iterrows():
        role = str(row["role"])
        threshold = float(row["threshold"])
        for axis in (ax1, ax2):
            label = None
            if axis is ax1 and role not in legend_roles:
                label = role.replace("_", " ")
                legend_roles.add(role)
            axis.axvline(
                threshold,
                color=colors.get(role, "gray"),
                linestyle="--",
                alpha=0.55,
                linewidth=1.2,
                label=label,
            )

    ax1.legend(loc="lower left", ncol=2, fontsize=8)
    ax2.legend(loc="best")
    fig.suptitle("Leakage-Free OOF Threshold Trade-offs")
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def validate_outputs(
    sweep: pd.DataFrame,
    pareto: pd.DataFrame,
    candidates: pd.DataFrame,
    current_threshold: float,
) -> None:
    if len(sweep) != len(THRESHOLDS):
        raise AssertionError("Threshold sweep does not contain all requested thresholds")
    if sweep["threshold"].duplicated().any():
        raise AssertionError("Threshold sweep contains duplicate thresholds")
    if not sweep[OFFICIAL_METRICS + ["precision", "recall"]].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise AssertionError("A reported score is outside [0, 1]")
    if pareto.empty:
        raise AssertionError("Pareto frontier is unexpectedly empty")
    expected_roles = set(ROLE_DEFINITIONS)
    if set(candidates["role"]) != expected_roles:
        raise AssertionError("Candidate roles are incomplete")
    current = candidates[candidates["role"] == "current_reference"].iloc[0]
    if not np.isclose(float(current["threshold"]), current_threshold, atol=0.005):
        raise AssertionError("Current-reference candidate does not match task_a_threshold.txt")

    # Independently verify that no reported Pareto point is dominated.
    values = sweep[OFFICIAL_METRICS].to_numpy(dtype=float)
    for _, point in pareto.iterrows():
        current_values = point[OFFICIAL_METRICS].to_numpy(dtype=float)
        dominated = np.any(
            np.all(values >= current_values - 1e-12, axis=1)
            & np.any(values > current_values + 1e-12, axis=1)
        )
        if dominated:
            raise AssertionError(f"Dominated threshold reported as Pareto: {point['threshold']}")


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    current_threshold = float(CURRENT_THRESHOLD_PATH.read_text(encoding="utf-8").strip())

    sweep = build_sweep(oof)
    pareto = pareto_frontier(sweep)
    candidates = select_candidates(sweep, pareto, current_threshold)
    validate_outputs(sweep, pareto, candidates, current_threshold)

    sweep.to_csv(SWEEP_PATH, index=False)
    pareto.to_csv(PARETO_PATH, index=False)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    write_report(sweep, pareto, candidates, current_threshold)
    write_chart(sweep, candidates)

    display_columns = [
        "role", "threshold", "taskA_f1", "precision", "recall",
        "onset_skill", "recovery_skill", "predicted_positive_rate",
    ]
    print("Leakage-free OOF operating-point candidates:\n")
    print(candidates[display_columns].round(4).to_string(index=False))
    print(f"\nPareto thresholds: {len(pareto)}")
    print(
        f"Saved {SWEEP_PATH}, {PARETO_PATH}, {CANDIDATES_PATH}, "
        f"{REPORT_PATH}, and {CHART_PATH}"
    )
    print(f"task_a_threshold.txt remains unchanged at {current_threshold:.2f}")


if __name__ == "__main__":
    main()
