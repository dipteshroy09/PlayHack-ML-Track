"""Scoring for the PlayHack injury prediction task.

Scoring rules (from PDF page 06/06):
  Task A: F1-score on injured_in_risk_window.
  Task B: onset_day_offset & recovery_duration, evaluated on truly injured athletes.
      - Hit (predicted injured): MAE against true values.
      - Miss (predicted not injured): fixed penalty of 30 on both predictions.
      - Skill = max(0, 1 - MAE_model / MAE_baseline).
      - Baseline: training-set mean over injured athletes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

MISS_PENALTY = 30.0            # fixed error when we miss a real injury
LABEL_A = "injured_in_risk_window"
LABEL_ONSET = "onset_day_offset"
LABEL_RECOVERY = "recovery_duration"


def make_split(df, val_size=0.20, seed=42):
    """Stratified split of the training athletes into train / validation.

    Stratifying on the injury label keeps the ~35% injured rate identical in
    both halves, so the validation score is a fair stand-in for the test set.
    Returns (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed,
        stratify=df[LABEL_A],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _regression_skill(true_vals, pred_vals, pred_injured, train_mean):
    """Skill score for one Task-B target on the truly-injured athletes.

    true_vals    : actual target values for injured athletes
    pred_vals    : model's predicted values for those same athletes
    pred_injured : model's yes/no injury call for those athletes (1/0)
    train_mean   : mean of the target over injured athletes in TRAIN (baseline)
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    pred_injured = np.asarray(pred_injured).astype(int)

    # Model error: if we said "not injured" for a truly injured athlete -> penalty.
    model_err = np.where(pred_injured == 1,
                         np.abs(pred_vals - true_vals),
                         MISS_PENALTY)
    mae_model = model_err.mean()

    # Baseline error: always predict the training mean (and it never "misses").
    mae_base = np.abs(train_mean - true_vals).mean()

    skill = max(0.0, 1.0 - mae_model / mae_base) if mae_base > 0 else 0.0
    return {"mae_model": mae_model, "mae_baseline": mae_base, "skill": skill}


def score(y_true, y_pred, train_means, verbose=True):
    """Full leaderboard score.

    y_true : DataFrame with columns injured_in_risk_window, onset_day_offset,
             recovery_duration (onset/recovery NaN when not injured).
    y_pred : DataFrame with the same three columns (model predictions).
    train_means : dict with 'onset' and 'recovery' = training means over the
             injured athletes (the baseline the skill score is measured against).
    Returns a dict of all metrics.
    """
    a_true = y_true[LABEL_A].astype(int).to_numpy()
    a_pred = y_pred[LABEL_A].astype(int).to_numpy()
    f1 = f1_score(a_true, a_pred)

    # --- Task B: only truly-injured athletes are scored ---
    injured_mask = a_true == 1
    onset = _regression_skill(
        y_true.loc[injured_mask, LABEL_ONSET],
        y_pred.loc[injured_mask, LABEL_ONSET],
        y_pred.loc[injured_mask, LABEL_A],
        train_means["onset"],
    )
    recovery = _regression_skill(
        y_true.loc[injured_mask, LABEL_RECOVERY],
        y_pred.loc[injured_mask, LABEL_RECOVERY],
        y_pred.loc[injured_mask, LABEL_A],
        train_means["recovery"],
    )

    combined = np.mean([f1, onset["skill"], recovery["skill"]])
    result = {
        "taskA_f1": f1,
        "onset_skill": onset["skill"],
        "onset_mae_model": onset["mae_model"],
        "onset_mae_baseline": onset["mae_baseline"],
        "recovery_skill": recovery["skill"],
        "recovery_mae_model": recovery["mae_model"],
        "recovery_mae_baseline": recovery["mae_baseline"],
        "combined": combined,
        "n_injured_scored": int(injured_mask.sum()),
    }
    if verbose:
        print("=" * 52)
        print("SCORECARD  (metrics defined by PDF page 06/06)")
        print("=" * 52)
        print(f"  Task A   F1-score          : {result['taskA_f1']:.4f}   [0-1, higher better]")
        print(f"  Task B   onset skill       : {result['onset_skill']:.4f}   "
              f"(MAE {onset['mae_model']:.2f} vs baseline {onset['mae_baseline']:.2f})")
        print(f"  Task B   recovery skill    : {result['recovery_skill']:.4f}   "
              f"(MAE {recovery['mae_model']:.2f} vs baseline {recovery['mae_baseline']:.2f})")
        print("-" * 52)
        print(f"  combined (unofficial mean) : {result['combined']:.4f}")
        print(f"  injured athletes scored    : {result['n_injured_scored']}")
        print("=" * 52)
    return result


def baseline_means(train_df):
    """Training means over injured athletes -> the Task B baseline predictions."""
    inj = train_df[train_df[LABEL_A] == 1]
    return {"onset": inj[LABEL_ONSET].mean(),
            "recovery": inj[LABEL_RECOVERY].mean()}


if __name__ == "__main__":
    df = pd.read_csv("train_features.csv")
    train_df, val_df = make_split(df)
    print(f"split: train={len(train_df)}  val={len(val_df)}")
    print(f"injury rate  train={train_df[LABEL_A].mean():.3f}  "
          f"val={val_df[LABEL_A].mean():.3f}  (should match ~0.35)")
    print()

    means = baseline_means(train_df)
    print(f"baseline onset mean   = {means['onset']:.2f}")
    print(f"baseline recovery mean= {means['recovery']:.2f}")
    print()

    naive_pred = pd.DataFrame({
        LABEL_A: np.zeros(len(val_df), dtype=int),
        LABEL_ONSET: means["onset"],
        LABEL_RECOVERY: means["recovery"],
    }, index=val_df.index)

    print("--- naive baseline (predict all healthy) on validation ---")
    score(val_df, naive_pred, means)
