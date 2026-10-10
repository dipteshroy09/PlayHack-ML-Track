# PPT Placeholders — Values to Update


## Slide 1 — Title Slide

| # | Placeholder | What to Put | Status |
|---|-------------|-------------|--------|
| 1 | `Ctrl Freaks` | Your team name | ☐ |
| 2 | `Diptesh Roy( Team Leader), Arpan Chatterjee, Ratul Bhattacharjee, Prithvi Dey` | Team member names | ☐ |

---

## Slide 6 — Feature Engineering

| # | Placeholder | Value | Source | Status |
|---|-------------|-------|--------|--------|
| 3 | `[___] total features per athlete` | **347 features** | `build_features.py` — 7 source families (athlete metadata, daily activity, sleep, heart rate, hourly calories, hourly intensity, hourly steps, training sessions, weight log, cross-source interactions) | ✅ |

---

## Slide 9 — Results (Most Important)

### Performance Metrics

| # | Placeholder | Value | Source | Status |
|---|-------------|-------|--------|--------|
| 4 | `CV F1 Score` | **0.5261** | `oof_report.txt` — Task A F1 at threshold 0.28 | ✅ |
| 5 | `ROC AUC` | **0.7655** | `SUBMISSION.md` Section 1 — threshold-free ranking quality | ✅ |
| 6 | `Skill Score — Onset` | **0.6416** | `oof_report.txt` — scored onset MAE 2.730 vs baseline 7.616 | ✅ |
| 7 | `Skill Score — Recovery` | **0.0857** | `oof_report.txt` — scored recovery MAE 2.970 vs baseline 3.249 | ✅ |
| — | `Pure Onset MAE` | **2.705 days** | `oof_report.txt` — MAE on truly injured athletes only | ✅ |
| — | `Pure Recovery MAE` | **2.943 days** | `oof_report.txt` — MAE on truly injured athletes only | ✅ |
| — | `Precision / Recall` | **0.3570 / 0.9990** | `oof_report.txt` — at threshold 0.28 | ✅ |

### Confusion Matrix (at threshold 0.28, OOF over 3,000 athletes)

| # | Placeholder | Value | Status |
|---|-------------|-------|--------|
| 8 | `Actual 0, Predicted 0` (TN) | **61** | ✅ |
| 9 | `Actual 0, Predicted 1` (FP) | **1,889** | ✅ |
| 10 | `Actual 1, Predicted 0` (FN) | **1** | ✅ |
| 11 | `Actual 1, Predicted 1` (TP) | **1,049** | ✅ |

> Source: `oof_report.txt` — Caught / missed injuries: 1049 / 1, False alarms / healthy: 1889 / 61

### Feature Importance Chart

| # | Placeholder | What to Put | How to Get It | Status |
|---|-------------|-------------|---------------|--------|
| 12 | Top 10 Feature Importance (bar chart) | Horizontal bar chart image | Extract from saved CatBoost models in `submission_artifacts/` using `model.get_feature_importance()` | ☐ |

---

## Slide — Detection Horizon (Key Insight for Presentation)

> This is the strongest differentiator. Lead with this.

| True Onset (days into risk window) | Athletes | Mean Predicted Prob | Detected at 0.50 | Onset MAE |
|---|---|---|---|---|
| 1–7 | 252 | 0.9335 | **94.4%** | 1.62 days |
| 8–14 | 251 | 0.9092 | **97.2%** | 0.97 days |
| 15–21 | 246 | 0.4237 | 15.5% | 3.43 days |
| 22–30 | 301 | 0.3760 | **2.3%** | 4.47 days |

> Correlation between predicted probability and true onset day = **−0.852**

---

## Slide — Threshold Decision

| Threshold | F1 | Onset Skill | Recovery Skill | Recall | Flagged % |
|---|---|---|---|---|---|
| **0.28 (chosen)** | **0.5261** | **0.6416** | **0.0857** | **99.9%** | **97.9%** |
| 0.34 (minimax-regret) | 0.5416 | 0.4717 | 0.0000 | 94.9% | 87.6% |
| 0.50 (F1 max) | 0.6466 | 0.0000 | 0.0000 | 50.2% | 19.3% |

> Break-even: F1 must carry >75% of total score weight for threshold 0.50 to beat 0.28.

---

## Slide — Model Architecture

| Model | Type | Loss | Trained On | Trees | Depth | LR |
|---|---|---|---|---|---|---|
| Task A Classifier | CatBoost | Logloss (balanced) | 3,000 athletes | 50 | 6 | 0.03 |
| Onset Regressor | CatBoost | MAE | 1,050 injured only | 316 | 6 | 0.03 |
| Recovery Regressor | CatBoost | MAE | 1,050 injured only | 150 | 6 | 0.03 |

---

## Slide — Fold Stability

| Fold | F1 at 0.28 | Onset MAE | Recovery MAE |
|---|---|---|---|
| 1 | 0.5303 | 2.452 | 2.838 |
| 2 | 0.5243 | 2.400 | 3.029 |
| 3 | 0.5276 | 2.810 | 3.057 |
| 4 | 0.5237 | 2.957 | 2.857 |
| 5 | 0.5245 | 2.905 | 2.933 |
| **Mean** | **0.5261** | **2.705** | **2.943** |

> F1 spread: only 0.0066 — results are not from a lucky fold.

---

## Slide 12 — Thank You / Q&A

| # | Placeholder | What to Put | Status |
|---|-------------|-------------|--------|
| 13 | `dipteshroy2409@gmail.com` | Your contact info | ☐ |

---

## Summary — What's Done vs What You Still Need

### ✅ Filled in from pipeline results
- Feature count: **347** (#3)
- CV F1 Score: **0.5261** (#4)
- ROC AUC: **0.7655** (#5)
- Onset Skill: **0.6416** (#6)
- Recovery Skill: **0.0857** (#7)
- Confusion Matrix: TP=1049, FN=1, FP=1889, TN=61 (#8–11)
- Detection horizon table, threshold trade-off, model architecture, fold stability


---

