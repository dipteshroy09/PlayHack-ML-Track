"""Generate the competition submission by refitting on all training data.

Uses the same hyperparameters and post-processing as the OOF evaluation.
Iteration counts are medians from the validated OOF fold summary.

Outputs:
    submission.csv                      the deliverable
    submission_artifacts/               refit models and validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

import scoring
from evaluate_oof import (
    ID_COL,
    LABEL_COLS,
    SEED,
    categorical_indices,
    classifier_model,
    current_threshold,
    prepare_features,
    regressor_model,
    score_oof,
)

TRAIN_PATH = Path("train_features.csv")
TEST_PATH = Path("test_features.csv")
FOLD_SUMMARY_PATH = Path("oof_fold_summary.csv")
OOF_PATH = Path("oof_predictions.csv")
ARTIFACT_DIR = Path("submission_artifacts")

TARGET_A = scoring.LABEL_A
TARGET_ONSET = scoring.LABEL_ONSET
TARGET_RECOVERY = scoring.LABEL_RECOVERY
SUBMISSION_COLUMNS = [ID_COL, TARGET_A, TARGET_ONSET, TARGET_RECOVERY]

ONSET_MIN, ONSET_MAX = 1, 30
RECOVERY_MIN = 1


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_predictors = [c for c in train.columns if c not in [ID_COL, *LABEL_COLS]]
    test_predictors = [c for c in test.columns if c != ID_COL]
    if train_predictors != test_predictors:
        raise AssertionError("Train and test predictors differ in name or order")
    leaked = [c for c in LABEL_COLS if c in test.columns]
    if leaked:
        raise AssertionError(f"Test features unexpectedly contain labels: {leaked}")
    if not train[ID_COL].is_unique or not test[ID_COL].is_unique:
        raise AssertionError("Athlete IDs must be unique in both tables")
    if train[LABEL_COLS].isna().all(axis=1).any():
        raise AssertionError("A training athlete is missing every label")
    return train, test, train_predictors


def selected_iterations() -> dict[str, int]:
    """Median iteration count each outer fold chose during the validated run."""
    folds = pd.read_csv(FOLD_SUMMARY_PATH)
    return {
        "classifier": int(np.median(folds["classifier_iterations"])),
        "onset": int(np.median(folds["onset_iterations"])),
        "recovery": int(np.median(folds["recovery_iterations"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = current_threshold()
    train, test, predictors = load_tables()
    iterations = selected_iterations()

    features_train = prepare_features(train)[predictors]
    features_test = prepare_features(test)[predictors]
    cat_idx = categorical_indices(predictors)

    injured = train[train[TARGET_A] == 1]
    features_injured = prepare_features(injured)[predictors]

    print(f"train athletes   : {len(train)}  (injured {len(injured)})")
    print(f"test athletes    : {len(test)}")
    print(f"predictors       : {len(predictors)}")
    print(f"threshold        : {threshold:.2f} (from task_a_threshold.txt)")
    print(f"iterations       : {iterations}")

    classifier = classifier_model(iterations["classifier"], SEED, early_stopping=False)
    classifier.fit(Pool(features_train, train[TARGET_A].astype(int), cat_features=cat_idx))

    onset_model = regressor_model(iterations["onset"], SEED + 100, early_stopping=False)
    onset_model.fit(
        Pool(features_injured, injured[TARGET_ONSET].astype(float), cat_features=cat_idx)
    )

    recovery_model = regressor_model(iterations["recovery"], SEED + 200, early_stopping=False)
    recovery_model.fit(
        Pool(features_injured, injured[TARGET_RECOVERY].astype(float), cat_features=cat_idx)
    )

    probability = classifier.predict_proba(features_test)[:, 1]
    onset = np.clip(np.rint(onset_model.predict(features_test)), ONSET_MIN, ONSET_MAX).astype(int)
    recovery = np.clip(np.rint(recovery_model.predict(features_test)), RECOVERY_MIN, None).astype(int)

    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL].to_numpy(),
            TARGET_A: (probability >= threshold).astype(int),
            TARGET_ONSET: onset,
            TARGET_RECOVERY: recovery,
        }
    )[SUBMISSION_COLUMNS]

    validate_submission(submission, test)
    submission.to_csv(args.output, index=False)

    oof = pd.read_csv(OOF_PATH)
    expected = score_oof(oof, threshold)
    positive_rate = float(submission[TARGET_A].mean())

    report = {
        "status": "PASS",
        "output": args.output,
        "threshold": threshold,
        "iterations": iterations,
        "test_rows": int(len(submission)),
        "predicted_positive_rate": positive_rate,
        "oof_predicted_positive_rate": float((oof["proba_injured"] >= threshold).mean()),
        "train_injury_rate": float(train[TARGET_A].mean()),
        "expected_from_oof": {
            "taskA_f1": expected["taskA_f1"],
            "precision": expected["precision"],
            "recall": expected["recall"],
            "onset_skill": expected["onset_skill"],
            "recovery_skill": expected["recovery_skill"],
            "pure_onset_mae": expected["pure_onset_mae"],
            "pure_recovery_mae": expected["pure_recovery_mae"],
        },
        "onset_summary": summarize(submission[TARGET_ONSET]),
        "recovery_summary": summarize(submission[TARGET_RECOVERY]),
    }
    (ARTIFACT_DIR / "submission_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    classifier.save_model(str(ARTIFACT_DIR / "final_task_a_classifier.cbm"))
    onset_model.save_model(str(ARTIFACT_DIR / "final_onset_regressor.cbm"))
    recovery_model.save_model(str(ARTIFACT_DIR / "final_recovery_regressor.cbm"))

    print("\n=== submission summary ===")
    print(submission.head(8).to_string(index=False))
    print(f"\nrows                        : {len(submission)}")
    print(f"flagged injured             : {submission[TARGET_A].sum()} ({positive_rate:.2%})")
    print(f"OOF flagged rate at {threshold:.2f}    : {report['oof_predicted_positive_rate']:.2%}")
    print(f"onset  min/median/max       : {submission[TARGET_ONSET].min()}/"
          f"{int(submission[TARGET_ONSET].median())}/{submission[TARGET_ONSET].max()}")
    print(f"recovery min/median/max     : {submission[TARGET_RECOVERY].min()}/"
          f"{int(submission[TARGET_RECOVERY].median())}/{submission[TARGET_RECOVERY].max()}")
    print("\nexpected performance (from validated OOF, not a test-set guarantee):")
    print(f"  Task A F1      : {expected['taskA_f1']:.4f}")
    print(f"  precision      : {expected['precision']:.4f}")
    print(f"  recall         : {expected['recall']:.4f}")
    print(f"  onset skill    : {expected['onset_skill']:.4f}")
    print(f"  recovery skill : {expected['recovery_skill']:.4f}")
    print(f"\nwrote {args.output} and {ARTIFACT_DIR}/")
    print("SUBMISSION_VALIDATION=PASS")


def summarize(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": float(series.max()),
    }


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if submission.columns.tolist() != SUBMISSION_COLUMNS:
        raise AssertionError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
    if len(submission) != len(test):
        raise AssertionError("Submission row count does not match the test set")
    if not submission[ID_COL].is_unique:
        raise AssertionError("Duplicate athlete_id in submission")
    if set(submission[ID_COL]) != set(test[ID_COL]):
        raise AssertionError("Submission athlete IDs do not match the test set")
    if submission.isna().any().any():
        raise AssertionError("Submission contains missing values")
    if not submission[TARGET_A].isin([0, 1]).all():
        raise AssertionError("injured_in_risk_window must be 0 or 1")
    if not submission[TARGET_ONSET].between(ONSET_MIN, ONSET_MAX).all():
        raise AssertionError(f"onset_day_offset outside [{ONSET_MIN}, {ONSET_MAX}]")
    if (submission[TARGET_RECOVERY] < RECOVERY_MIN).any():
        raise AssertionError("recovery_duration must be positive")
    for column in [TARGET_A, TARGET_ONSET, TARGET_RECOVERY]:
        values = submission[column].to_numpy()
        if not np.all(values == np.rint(values)):
            raise AssertionError(f"{column} must contain whole numbers")
    # Page 5: timing predictions are required even when the classification is 0.
    healthy = submission[submission[TARGET_A] == 0]
    if len(healthy) and (healthy[TARGET_ONSET].isna().any() or healthy[TARGET_RECOVERY].isna().any()):
        raise AssertionError("Timing predictions must be present for not-injured rows too")


if __name__ == "__main__":
    main()
