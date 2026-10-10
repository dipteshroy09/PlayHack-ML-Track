# ⚡ FaultLine — Detecting the Fracture Point, 14 Days Before It Happens

<div align="center">

**Team Ctrl Freaks** &nbsp;|&nbsp; PlayHack 2026 ML Track &nbsp;|&nbsp; IIT Guwahati — Sports Board × Technical Board

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-Academic-blue)](LICENSE)

</div>

---

> **FaultLine** is not just a classifier — it is a deeply calibrated early-warning system with a mathematically validated detection range and an optimized risk threshold. From 30 days of wearable data, it predicts whether a sports athlete will be injured in the next 30 days, and if so, exactly **when** and **for how long**.

---

## 👥 Team

| Name | Role |
|---|---|
| Diptesh Roy | Team Leader |
| Arpan Chatterjee | Member |
| Ratul Bhattacharjee | Member |
| Prithvi Dey | Member |

---

## 🎯 The Challenge

**Given:** 30 days of wearable sensor data (Days 1–30) for each athlete.  
**Predict:** What happens in the hidden risk window (Days 31–60)?

| Task | Type | Output |
|---|---|---|
| **Task A** | Binary Classification | Will the athlete be injured in days 31–60? (0 or 1) |
| **Task B** | Regression | If injured — onset day offset (1–30) and recovery duration (days) |

**Dataset:** 3,000 training athletes · 1,100 test athletes · 7 disparate wearable data sources

```
|--- Days 1-30 ---|--- Days 31-60 ---|
  Observation        Hidden Risk
  Window (Known)     Window (Predict)
```

---

## 🏆 Performance Results

<div align="center">

| F1 Score | ROC AUC | Onset Skill | Recovery Skill |
|:---:|:---:|:---:|:---:|
| **0.5261** | **0.7655** | **0.6416** | **0.0857** |

</div>

### Confusion Matrix (at threshold 0.28)

|  | Predicted Healthy | Predicted Injured |
|---|---|---|
| **Actually Healthy** | TN = 61 | FP = 1,889 |
| **Actually Injured** | **FN = 1** ⚠️ | TP = 1,049 |

> **Precision:** 0.3570 &nbsp;|&nbsp; **Recall:** 0.9990 &nbsp;|&nbsp; **Onset MAE:** 2.705 days &nbsp;|&nbsp; **Recovery MAE:** 2.943 days

> 🎯 The system flagged 97.9% of athletes for review, missing exactly **ONE** injured athlete out of 1,050.

### Per-Fold Stability (5-Fold CV)

| Fold | F1 @ 0.28 | Onset MAE | Recovery MAE |
|---|---|---|---|
| 1 | 0.5303 | 2.452 | 2.838 |
| 2 | 0.5243 | 2.400 | 3.029 |
| 3 | 0.5276 | 2.810 | 3.057 |
| 4 | 0.5237 | 2.957 | 2.857 |
| 5 | 0.5245 | 2.905 | 2.933 |
| **Mean** | **0.5261** | **2.705** | **2.943** |

> F1 spread across all 5 folds = **0.0066** — proving the headline figure is not a lucky fold.

---

## 🔑 Key Discovery: The Detection Horizon

> This is the most significant finding — structural, not algorithmic.

**Correlation between predicted injury probability and true onset day = −0.852**

| True Onset (Risk Window) | Athletes | Detected @ 0.50 | Onset MAE |
|---|---|---|---|
| Days 1–7 | 252 | **94.4%** | 1.62 days |
| Days 8–14 | 251 | **97.2%** | 0.97 days |
| Days 15–21 | 246 | 15.5% | 3.43 days |
| Days 22–30 | 301 | **2.3%** | 4.47 days |

Athletes injured after day 21 have a mean predicted probability of **0.376** — statistically identical to healthy athletes at **0.377**. Late-onset unpredictability is **biology, not a tuning failure**.

**FaultLine is a genuine 14-day early-warning system.** No model trained on this observation window can be a 30-day one.

---

## ⚖️ The Threshold Decision (Game Theory)

**Strategic Asymmetry:** Missing an injury incurs a massive **30-day fixed penalty** on both timing tasks. Missing an injury is far worse than a false alarm.

This was treated as a formal game-theory optimization, not a heuristic choice:

| Threshold | F1 | Onset Skill | Recovery Skill | Recall |
|---|---|---|---|---|
| 0.50 (Standard) | 0.6466 | 0.0000 | 0.0000 | 50.2% |
| 0.34 (Minimax) | 0.5416 | 0.4717 | 0.0000 | 94.9% |
| **0.28 (Selected ✓)** | **0.5261** | **0.6416** | **0.0857** | **99.9%** |

**Synthesis:** Moving the threshold from 0.50 → 0.28 sacrifices **0.12** in F1, but unlocks a massive **+0.73** gain across timing skills. F1 would need to carry **>75%** of the total scoring weight for 0.50 to be the better choice.

> The 98% flag rate is **not** a claim that 98% of athletes will be injured — it is the scoring-optimal action under the competition's penalty structure. As a ranked risk list (how such a model would actually be used), the **AUC of 0.7655** and **14-day detection rate of 94–97%** are the operationally meaningful numbers.

---

## 🧱 Pipeline Architecture

```
                        ┌─────────────────┐
                        │  347 Features   │
                        │  (7 sources)    │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Task A Classifier     │
                    │   (CatBoost, 50 trees)  │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                                   ▼
   ┌─────────────────────┐           ┌──────────────────────┐
   │   Onset Regressor   │           │  Recovery Regressor   │
   │  (CatBoost, 316 t.) │           │  (CatBoost, 150 t.)  │
   └──────────┬──────────┘           └──────────┬───────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                    Final Submission CSV
```

### Models

| Model | Loss | Trained On | Trees | Depth | LR |
|---|---|---|---|---|---|
| Task A Classifier | Logloss (balanced) | 3,000 athletes | 50 | 6 | 0.03 |
| Onset Regressor | MAE | 1,050 injured only | 316 | 6 | 0.03 |
| Recovery Regressor | MAE | 1,050 injured only | 150 | 6 | 0.03 |

> Tree counts are **medians from 5-fold cross-validation**, never tuned on final fit data.

---

## 🔬 Exploratory Data Analysis

- **7 data sources:** daily activity, sleep, heart rate, hourly calories/intensity/steps, training sessions, weight log
- **Class split:** ~35% injured — 1,050 injured vs 1,950 healthy athletes
- **Missingness:** Only **0.41%** overall, concentrated in weight/body-fat fields
- **Key signal:** Wearable data shows clear physiological degradation patterns in the **final 7–14 days** before injury onset

---

## 🛡️ Zero-Leakage Guarantee

Every design choice was made to prevent information from the risk window contaminating the observation window:

- **Strict half-open intervals:** Days 1–30 only; includes all 24 hours of Day 30, excludes Day 31+
- **Critical bug fix:** Corrected an inclusive end-date issue that silently dropped hours 01–23 on Day 30 for all four hourly sources
- **Native NaN handling:** No global imputation — CatBoost handles NaNs natively, eliminating imputer leakage across CV folds
- **Categorical integrity:** Sport, gender, dominant side, position, and team passed as native CatBoost categoricals
- **Nested CV:** Baselines computed per-fold from the training split only — never from global statistics

```
FEATURE_TABLE_VALIDATION  = ✅ PASS
SOURCE_WINDOW_VALIDATION  = ✅ PASS
```

---

## ⚙️ Feature Engineering — 347 Predictors

**The Athlete-Relative Philosophy:** Every feature compares an athlete against their own prior behavior inside disjoint time windows — never against population-fitted statistics. This makes features portable, leakage-free, and genuinely personalized.

| Family | Count | What It Captures |
|---|---|---|
| Athlete Metadata | 10 | Age, height, weight, years playing, prior injuries, sport, position |
| Daily Activity (`da_`) | 107 | Steps, distance, active-minute bands, load ratios, ACWR |
| Sleep (`sleep_`) | 42 | Duration, efficiency, awake-in-bed, sleep debt |
| Heart Rate (`hr_`) | 42 | Daily avg, resting proxy (5th %ile), night HR, coverage |
| Hourly Calories (`cal_`) | 31 | Coverage-adjusted daily load, active hours, circadian share |
| Hourly Intensity (`intensity_`) | 31 | Structural intensity load, evening share |
| Hourly Steps (`hsteps_`) | 31 | Structural step load, disjoint window shifts |
| Training Sessions (`sess_`) | 45 | Load, streaks, monotony, strain, session-type mix |
| Weight Log + Interactions | 8 | BMI, body fat, load-vs-sleep, active-vs-sleep-debt |

### Top 10 Most Important Features (Task A Classifier)

1. `da_moderate_vigorous_minutes_mean_last3`
2. `intensity_daily_total_adjusted_mean_last3`
3. `da_vigorous_share_mean_last3`
4. `da_Calories_mean_last3`
5. `hsteps_daily_total_adjusted_zshift7_prev21`
6. `hr_daily_avg_mean_last3`
7. `da_VeryActiveMinutes_max`
8. `interaction_load_minus_sleep`
9. `da_moderate_vigorous_minutes_mean_last14`
10. `intensity_hourly_std`

> The top features are overwhelmingly **recent-window activity metrics** — confirming the 14-day signal horizon empirically.

---

## 🚀 What Sets FaultLine Apart

### 1. 🔭 Discovery, Not Just Prediction
We uncovered a fundamental physiological truth — injury detectability possesses a rigid **~14-day horizon**. Late-onset unpredictability is biology, not a tuning failure. This reframes the problem for anyone using such a model operationally.

### 2. 🎯 Game-Theory Threshold Optimization
We formally analyzed the asymmetric 30-day timing penalty. Deriving the optimal 0.28 threshold via break-even analysis delivered **+0.20 to the composite score** — outperforming any model tuning we attempted.

### 3. 🛡️ Zero-Leakage Guarantee
Strict nested 5-fold CV with per-fold baselines and zero global imputation. A microscopic F1 spread of **0.0066** guarantees high out-of-sample reliability.

### 4. 🧬 347 Athlete-Relative Features
Features explicitly compare athletes against their own historical disjoint time windows, creating a genuinely **personalized, population-independent** warning system.

---

## 🚀 Reproducing from Scratch

### Prerequisites

```bash
pip install catboost pandas numpy scikit-learn
```

Python 3.10+ recommended. All scripts are **deterministic** (seed 42 throughout).

### Run Order

```powershell
# 1. Engineer 347 features from raw wearable data
python build_features.py

# 2. Validate — expect both checks to PASS
python validate_features.py

# 3. Leakage-free 5-fold cross-validation (takes a few minutes)
python evaluate_oof.py

# 4. Sweep thresholds and generate Pareto analysis
python analyze_thresholds.py

# 5. Refit on all 3,000 athletes → write submission.csv
python make_submission.py
```

### Expected Dataset Layout

```
Dataset/
├── athlete_metadata.csv
├── dailyActivity_merged.csv
├── sleepDay_merged.csv
├── hourlyHeartrate_merged.csv
├── hourlyCalories_merged.csv
├── hourlyIntensities_merged.csv
├── hourlySteps_merged.csv
├── training_sessions.csv
├── train_labels.csv
└── Test data/
    └── (same sources, no labels)
```

---

## 📁 Repository Structure

```
training_pipeline/
│
├── build_features.py            # 347-feature engineering pipeline
├── validate_features.py         # Schema, window & parity checks
├── evaluate_oof.py              # Strict nested 5-fold OOF evaluation
├── analyze_thresholds.py        # Threshold sweep + Pareto + regret heatmap
├── make_submission.py           # Final refit → submission.csv
├── scoring.py                   # Official metric implementation
├── task_a_threshold.txt         # Chosen threshold (0.28)
│
├── diagnose_detectability.py    # The 14-day horizon discovery
├── diagnose_recovery.py         # Recovery ceiling analysis
├── threshold_breakeven.py       # Game-theory arithmetic
├── threshold_decision_analysis.py
├── compare_submissions.py
├── select_recall_threshold.py
│
├── oof_predictions.csv          # Per-athlete OOF predictions
├── oof_fold_summary.csv         # Per-fold diagnostics
├── oof_report.txt               # Headline validated scorecard
├── train_features.csv           # Pre-built train feature table
├── test_features.csv            # Pre-built test feature table
├── submission.csv               # Final submission (threshold 0.28)
│
├── submission_artifacts/
│   ├── final_task_a_classifier.cbm
│   ├── final_onset_regressor.cbm
│   ├── final_recovery_regressor.cbm
│   ├── submission_threshold_0.28.csv   ← submitted
│   ├── submission_threshold_0.34.csv   ← minimax-regret alternative
│   └── submission_threshold_0.50.csv   ← high-F1 alternative
│
├── Ctrl Freaks.pdf              # Team presentation slides
├── SUBMISSION.md                # Full technical write-up
└── README.md
```

---

## 🧪 What Didn't Work (Honest Negatives)

We report these because they explain why the final model is deliberately simple:

- **Feature family ablation** — Six leave-one-family-out variants; no family produced a repeatable OOF improvement.
- **Model configuration tuning** — Five CatBoost configs selected by inner-fold AUC. The existing config won. The model genuinely saturates in ~50 trees.
- **Recovery model search** — Eight configurations across tree, linear, and blend families. Nothing beat the production config; ElasticNet (3.019), Ridge (3.045), and Huber (3.288) all performed worse. The ceiling is structural.

> The threshold decision alone delivered **+0.20 composite metric points**. Remaining modelling headroom is ~+0.03.

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| No official metric weighting | The threshold is a documented bet; the 0.50 alternative is pre-built |
| Horizon-bound recall | Injuries starting after day 21 are physiologically undetectable from days 1–30 |
| Recovery barely predictable | MAE 2.943 vs baseline 3.249; ceiling confirmed across 8 model families |
| Mild test-set shift | Recent HR features +0.10–0.12 SMD; no feature exceeds 0.25 SMD |
| Team-level generalisation untested | Folds are athlete-stratified, not grouped by `team_id` |

---

## 📄 License

Developed for **PlayHack 2026 ML Track**, IIT Guwahati.  
Dataset © IIT Guwahati Sports Board. Code available for educational and research purposes.
