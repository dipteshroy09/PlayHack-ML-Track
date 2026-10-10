"""Leakage-free five-fold out-of-fold evaluation for the PlayHack pipeline.

Each athlete's predictions come from models that never trained on that athlete.
Inner splits select CatBoost iteration counts without exposing outer-fold labels.

Outputs:
    oof_predictions.csv       per-athlete OOF predictions
    oof_fold_summary.csv      per-fold iteration counts and diagnostics
    oof_report.txt            aggregate metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import scoring

DATA_PATH = Path("train_features.csv")
OOF_PATH = Path("oof_predictions.csv")
FOLD_SUMMARY_PATH = Path("oof_fold_summary.csv")
REPORT_PATH = Path("oof_report.txt")

THRESHOLD_PATH = Path("task_a_threshold.txt")
SEED = 42
N_SPLITS = 5
INNER_SIZE = 0.18
MAX_ITERATIONS = 1500
MIN_ITERATIONS = 50

TARGET_A = scoring.LABEL_A
TARGET_ONSET = scoring.LABEL_ONSET
TARGET_RECOVERY = scoring.LABEL_RECOVERY
LABEL_COLS = [TARGET_A, TARGET_ONSET, TARGET_RECOVERY]
ID_COL = "athlete_id"
DROP_COLS = [ID_COL, *LABEL_COLS]
CAT_COLS = ["sport", "gender", "dominant_side", "position", "team_id"]


@dataclass(frozen=True)
class SelectedIterations:
    classifier: int
    onset: int
    recovery: int


def current_threshold() -> float:
    """Read the single source of truth for the Task A decision threshold."""
    return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model inputs in a CatBoost-safe, stable column order."""
    features = frame[[c for c in frame.columns if c not in DROP_COLS]].copy()
    for column in CAT_COLS:
        features[column] = features[column].fillna("missing").astype(str)
    return features


def categorical_indices(feature_columns: list[str]) -> list[int]:
    missing = [column for column in CAT_COLS if column not in feature_columns]
    if missing:
        raise ValueError(f"Missing categorical features: {missing}")
    return [feature_columns.index(column) for column in CAT_COLS]


def classifier_model(iterations: int, seed: int, *, early_stopping: bool) -> CatBoostClassifier:
    params = dict(
        loss_function="Logloss",
        eval_metric="F1",
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 100
    return CatBoostClassifier(**params)


def regressor_model(iterations: int, seed: int, *, early_stopping: bool) -> CatBoostRegressor:
    params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 100
    return CatBoostRegressor(**params)


def safe_best_iterations(model, default: int = 500) -> int:
    """Convert CatBoost's zero-based best iteration to a safe tree count."""
    best = model.get_best_iteration()
    if best is None or best < 0:
        return default
    return max(MIN_ITERATIONS, int(best) + 1)


def select_classifier_iterations(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    cat_idx: list[int],
    seed: int,
) -> int:
    """Choose tree count without exposing the untouched outer fold."""
    inner_train, inner_stop = train_test_split(
        outer_train,
        test_size=INNER_SIZE,
        random_state=seed,
        stratify=outer_train[TARGET_A],
    )
    x_train = prepare_features(inner_train)[feature_columns]
    x_stop = prepare_features(inner_stop)[feature_columns]
    model = classifier_model(MAX_ITERATIONS, seed, early_stopping=True)
    model.fit(
        Pool(x_train, inner_train[TARGET_A].astype(int), cat_features=cat_idx),
        eval_set=Pool(x_stop, inner_stop[TARGET_A].astype(int), cat_features=cat_idx),
        use_best_model=True,
    )
    return safe_best_iterations(model)


def select_regressor_iterations(
    injured_outer_train: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    cat_idx: list[int],
    seed: int,
) -> int:
    """Choose a regressor tree count using injured outer-training rows only."""
    inner_train, inner_stop = train_test_split(
        injured_outer_train,
        test_size=INNER_SIZE,
        random_state=seed,
    )
    x_train = prepare_features(inner_train)[feature_columns]
    x_stop = prepare_features(inner_stop)[feature_columns]
    model = regressor_model(MAX_ITERATIONS, seed, early_stopping=True)
    model.fit(
        Pool(x_train, inner_train[target].astype(float), cat_features=cat_idx),
        eval_set=Pool(x_stop, inner_stop[target].astype(float), cat_features=cat_idx),
        use_best_model=True,
    )
    return safe_best_iterations(model)


def fit_predict_fold(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    fold: int,
    feature_columns: list[str],
    cat_idx: list[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train entirely inside one outer fold and predict its untouched rows."""
    seed = SEED + fold
    injured_train = outer_train[outer_train[TARGET_A] == 1].copy()

    selected = SelectedIterations(
        classifier=select_classifier_iterations(outer_train, feature_columns, cat_idx, seed),
        onset=select_regressor_iterations(
            injured_train, TARGET_ONSET, feature_columns, cat_idx, seed + 100
        ),
        recovery=select_regressor_iterations(
            injured_train, TARGET_RECOVERY, feature_columns, cat_idx, seed + 200
        ),
    )

    x_train = prepare_features(outer_train)[feature_columns]
    x_injured = prepare_features(injured_train)[feature_columns]
    x_valid = prepare_features(outer_valid)[feature_columns]

    classifier = classifier_model(selected.classifier, seed, early_stopping=False)
    classifier.fit(
        Pool(x_train, outer_train[TARGET_A].astype(int), cat_features=cat_idx)
    )

    onset_model = regressor_model(selected.onset, seed + 100, early_stopping=False)
    onset_model.fit(
        Pool(x_injured, injured_train[TARGET_ONSET].astype(float), cat_features=cat_idx)
    )

    recovery_model = regressor_model(selected.recovery, seed + 200, early_stopping=False)
    recovery_model.fit(
        Pool(x_injured, injured_train[TARGET_RECOVERY].astype(float), cat_features=cat_idx)
    )

    predictions = outer_valid[[ID_COL, *LABEL_COLS]].copy()
    predictions["fold"] = fold
    predictions["proba_injured"] = classifier.predict_proba(x_valid)[:, 1]
    predictions["pred_onset"] = np.clip(
        np.rint(onset_model.predict(x_valid)), 1, 30
    ).astype(int)
    predictions["pred_recovery"] = np.clip(
        np.rint(recovery_model.predict(x_valid)), 1, None
    ).astype(int)

    baseline = scoring.baseline_means(outer_train)
    predictions["fold_baseline_onset"] = baseline["onset"]
    predictions["fold_baseline_recovery"] = baseline["recovery"]
    predictions["outer_train_rows"] = len(outer_train)
    predictions["outer_train_injured_rows"] = len(injured_train)

    fold_injured = predictions[TARGET_A] == 1
    fold_metrics = {
        "fold": fold,
        "outer_train_rows": len(outer_train),
        "outer_valid_rows": len(outer_valid),
        "outer_train_injured_rows": len(injured_train),
        "outer_valid_injured_rows": int(fold_injured.sum()),
        "classifier_iterations": selected.classifier,
        "onset_iterations": selected.onset,
        "recovery_iterations": selected.recovery,
        "pure_onset_mae": mean_absolute_error(
            predictions.loc[fold_injured, TARGET_ONSET],
            predictions.loc[fold_injured, "pred_onset"],
        ),
        "pure_recovery_mae": mean_absolute_error(
            predictions.loc[fold_injured, TARGET_RECOVERY],
            predictions.loc[fold_injured, "pred_recovery"],
        ),
    }
    return predictions, fold_metrics


def score_oof(oof: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Score OOF predictions with each row's fold-training baseline mean."""
    true_a = oof[TARGET_A].astype(int).to_numpy()
    pred_a = (oof["proba_injured"].to_numpy() >= threshold).astype(int)
    injured = true_a == 1

    onset_true = oof.loc[injured, TARGET_ONSET].to_numpy(dtype=float)
    onset_pred = oof.loc[injured, "pred_onset"].to_numpy(dtype=float)
    recovery_true = oof.loc[injured, TARGET_RECOVERY].to_numpy(dtype=float)
    recovery_pred = oof.loc[injured, "pred_recovery"].to_numpy(dtype=float)
    injured_call = pred_a[injured]

    onset_model_error = np.where(
        injured_call == 1,
        np.abs(onset_pred - onset_true),
        scoring.MISS_PENALTY,
    )
    recovery_model_error = np.where(
        injured_call == 1,
        np.abs(recovery_pred - recovery_true),
        scoring.MISS_PENALTY,
    )
    onset_baseline_error = np.abs(
        oof.loc[injured, "fold_baseline_onset"].to_numpy(dtype=float) - onset_true
    )
    recovery_baseline_error = np.abs(
        oof.loc[injured, "fold_baseline_recovery"].to_numpy(dtype=float) - recovery_true
    )

    onset_mae = float(onset_model_error.mean())
    recovery_mae = float(recovery_model_error.mean())
    onset_baseline_mae = float(onset_baseline_error.mean())
    recovery_baseline_mae = float(recovery_baseline_error.mean())
    onset_skill = max(0.0, 1.0 - onset_mae / onset_baseline_mae)
    recovery_skill = max(0.0, 1.0 - recovery_mae / recovery_baseline_mae)

    tn, fp, fn, tp = confusion_matrix(true_a, pred_a).ravel()
    task_a_f1 = f1_score(true_a, pred_a)
    return {
        "threshold": threshold,
        "taskA_f1": float(task_a_f1),
        "precision": float(precision_score(true_a, pred_a, zero_division=0)),
        "recall": float(recall_score(true_a, pred_a, zero_division=0)),
        "true_positive": int(tp),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "pure_onset_mae": float(mean_absolute_error(onset_true, onset_pred)),
        "pure_recovery_mae": float(mean_absolute_error(recovery_true, recovery_pred)),
        "onset_mae_model": onset_mae,
        "onset_mae_baseline": onset_baseline_mae,
        "onset_skill": onset_skill,
        "recovery_mae_model": recovery_mae,
        "recovery_mae_baseline": recovery_baseline_mae,
        "recovery_skill": recovery_skill,
        "combined_unofficial": float(np.mean([task_a_f1, onset_skill, recovery_skill])),
        "injured_athletes_scored": int(injured.sum()),
    }


def format_report(metrics: dict[str, float], rows: int) -> str:
    return "\n".join(
        [
            "LEAKAGE-FREE 5-FOLD OOF REPORT",
            "=" * 54,
            f"OOF athletes                 : {rows}",
            f"Task A threshold (from file) : {metrics['threshold']:.2f}",
            "",
            f"Task A F1                    : {metrics['taskA_f1']:.4f}",
            f"Task A precision             : {metrics['precision']:.4f}",
            f"Task A recall                : {metrics['recall']:.4f}",
            f"Caught / missed injuries     : {metrics['true_positive']} / {metrics['false_negative']}",
            f"False alarms / healthy       : {metrics['false_positive']} / {metrics['true_negative']}",
            "",
            f"Pure onset MAE               : {metrics['pure_onset_mae']:.3f}",
            f"Scored onset MAE             : {metrics['onset_mae_model']:.3f}",
            f"OOF onset baseline MAE       : {metrics['onset_mae_baseline']:.3f}",
            f"Onset skill                  : {metrics['onset_skill']:.4f}",
            "",
            f"Pure recovery MAE            : {metrics['pure_recovery_mae']:.3f}",
            f"Scored recovery MAE          : {metrics['recovery_mae_model']:.3f}",
            f"OOF recovery baseline MAE    : {metrics['recovery_mae_baseline']:.3f}",
            f"Recovery skill               : {metrics['recovery_skill']:.4f}",
            "",
            f"Unofficial equal-weight mean : {metrics['combined_unofficial']:.4f}",
            "(The PDF does not define an official combined weighting.)",
        ]
    )


def validate_oof(oof: pd.DataFrame, source: pd.DataFrame) -> None:
    """Fail loudly if OOF coverage, uniqueness, or fold isolation is broken."""
    if len(oof) != len(source):
        raise AssertionError(f"Expected {len(source)} OOF rows, got {len(oof)}")
    if oof[ID_COL].duplicated().any():
        duplicates = oof.loc[oof[ID_COL].duplicated(), ID_COL].tolist()[:10]
        raise AssertionError(f"Duplicate OOF athlete IDs: {duplicates}")
    if set(oof[ID_COL]) != set(source[ID_COL]):
        raise AssertionError("OOF athlete IDs do not exactly match source athlete IDs")
    required = ["proba_injured", "pred_onset", "pred_recovery", "fold"]
    if oof[required].isna().any().any():
        raise AssertionError("OOF prediction columns contain missing values")
    if sorted(oof["fold"].unique().tolist()) != list(range(1, N_SPLITS + 1)):
        raise AssertionError("OOF output does not contain exactly folds 1..5")
    if not oof["proba_injured"].between(0, 1).all():
        raise AssertionError("Task A OOF probabilities are outside [0, 1]")
    if not oof["pred_onset"].between(1, 30).all():
        raise AssertionError("OOF onset predictions are outside [1, 30]")
    if (oof["pred_recovery"] < 1).any():
        raise AssertionError("OOF recovery predictions must be positive")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    feature_columns = prepare_features(data).columns.tolist()
    cat_idx = categorical_indices(feature_columns)

    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []

    print("Building strict leakage-free OOF predictions...")
    for fold, (train_idx, valid_idx) in enumerate(
        outer_cv.split(data, data[TARGET_A]), start=1
    ):
        outer_train = data.iloc[train_idx].copy()
        outer_valid = data.iloc[valid_idx].copy()
        predictions, summary = fit_predict_fold(
            outer_train, outer_valid, fold, feature_columns, cat_idx
        )
        oof_parts.append(predictions)
        fold_metrics.append(summary)
        print(
            f"  fold {fold}: train={len(outer_train)}, valid={len(outer_valid)}, "
            f"iterations A/B1/B2={summary['classifier_iterations']}/"
            f"{summary['onset_iterations']}/{summary['recovery_iterations']}, "
            f"pure MAE onset={summary['pure_onset_mae']:.3f}, "
            f"recovery={summary['pure_recovery_mae']:.3f}"
        )

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(ID_COL).reset_index(drop=True)
    folds = pd.DataFrame(fold_metrics)
    validate_oof(oof, data)

    oof.to_csv(OOF_PATH, index=False)
    folds.to_csv(FOLD_SUMMARY_PATH, index=False)

    metrics = score_oof(oof, current_threshold())
    report = format_report(metrics, len(oof))
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    print("\nValidation checks passed:")
    print("  - exactly one OOF row per athlete")
    print("  - all five folds present")
    print("  - no missing predictions")
    print("  - predictions satisfy required ranges")
    print(f"\nSaved {OOF_PATH}, {FOLD_SUMMARY_PATH}, and {REPORT_PATH}")


if __name__ == "__main__":
    main()
