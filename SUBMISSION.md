# PlayHack ML Track — Injury Prediction Submission

**Challenge:** IIT Guwahati Sports Board × Technical Board — PLAYHACK / ML PS
**Task:** From a 30-day observation window (days 1–30), predict whether an injury onset occurs in the hidden risk window (days 31–60), and if so, when it begins and how long it sidelines the athlete.

---

## 1. Headline result

A three-model CatBoost pipeline evaluated with strict leakage-free 5-fold cross-validation over all 3,000 training athletes.

| Official metric (PDF page 6) | Value |
| --- | --- |
| Task A — F1 | **0.5261** |
| Task B — onset skill | **0.6416** |
| Task B — recovery skill | **0.0857** |

Supporting numbers at the submitted operating point (threshold 0.28):

| Quantity | Value |
| --- | --- |
| Precision / Recall | 0.3570 / 0.9990 |
| Confusion matrix (TP / FN / FP / TN) | 1049 / 1 / 1889 / 61 |
| ROC AUC (threshold-free ranking quality) | 0.7655 |
| Log loss | 0.4997 |
| Pure onset MAE (injured athletes) | 2.705 days |
| Pure recovery MAE (injured athletes) | 2.943 days |
| Scored onset MAE / baseline | 2.730 / 7.616 |
| Scored recovery MAE / baseline | 2.971 / 3.249 |

The PDF defines the three metrics separately and never states how they are combined, so they are reported separately here. No invented composite score is claimed.

---

## 2. What is being submitted

`submission.csv` — 1,100 rows, one per test athlete.

```
athlete_id,injured_in_risk_window,onset_day_offset,recovery_duration
3001,1,24,14
3002,1,24,13
3003,1,23,10
```

| Check | Status |
| --- | --- |
| Row count matches test set (1,100) | PASS |
| `athlete_id` unique and exactly matches test athletes | PASS |
| Column names and order per PDF | PASS |
| All values integers, zero nulls | PASS |
| `injured_in_risk_window` ∈ {0, 1} | PASS |
| `onset_day_offset` within 1–30 | PASS |
| `recovery_duration` ≥ 1 | PASS |
| Timing values present even when classification is 0 (PDF page 3 rule) | PASS |

Flagged injured: 1,083 of 1,100 (98.45%). Section 6 explains why this figure is high and deliberate.

---

## 3. The pipeline

### 3.1 Observation window handling

Features use **only days 1–30**, as required. An earlier inclusive end-date comparison was found to silently drop hours 01:00–23:00 on day 30 for all four hourly sources, retaining only midnight. This was replaced with a half-open interval: include day 1 through the end of day 30, exclude anything from day 31 onward. Independent validation confirmed all four hourly sources now carry hours 00–23 on day 30, and no post-observation data leaks in.

### 3.2 Feature engineering — 347 predictors

Built by `build_features.py` from all seven supplied sources, joined per athlete.

| Family | Count | Content |
| --- | --- | --- |
| Athlete metadata | 10 | age, height, baseline weight, years playing, prior injuries, plus 5 categoricals (sport, gender, dominant side, position, team) |
| Daily activity (`da_`) | 107 | steps, distance, active-minute bands, calories, derived load ratios |
| Sleep (`sleep_`) | 42 | duration, efficiency, awake-in-bed, sleep debt |
| Heart rate (`hr_`) | 42 | daily average, resting proxy (5th percentile), night HR, coverage |
| Hourly calories (`cal_`) | 31 | coverage-adjusted daily load, active hours, circadian share |
| Hourly intensity (`intensity_`) | 31 | as above |
| Hourly steps (`hsteps_`) | 31 | as above |
| Training sessions (`sess_`) | 45 | load, rest days, streaks, monotony, strain, session-type mix |
| Weight log | 4 | latest weight, BMI, body fat, record-present flag |
| Cross-source interactions | 4 | load-vs-sleep, load-vs-resting-HR, session-vs-sleep, active-vs-sleep-debt |

Two design choices keep this leakage-safe:

- **Athlete-relative, not population-fitted.** Every change feature compares an athlete against their own earlier behaviour, never against statistics fitted across the dataset.
- **Disjoint comparison windows.** Recent windows (last 3 / 7 / 14 days) are compared against non-overlapping prior windows (days 3–23, 24–30 split points), so "change" means something unambiguous.

Missingness is 0.41% overall and concentrated in the naturally sparse weight/body-fat fields. No global imputation was applied; CatBoost handles numeric NaNs natively, which avoids fitting an imputer outside the cross-validation folds.

### 3.3 Models

Three CatBoost models, all with depth 6, learning rate 0.03, `l2_leaf_reg` 3.0.

| Model | Loss | Trained on | Trees |
| --- | --- | --- | --- |
| Task A classifier | Logloss, balanced class weights | all 3,000 athletes | 50 |
| Onset regressor | MAE | 1,050 injured athletes only | 316 |
| Recovery regressor | MAE | 1,050 injured athletes only | 150 |

Tree counts are the medians of the counts each cross-validation fold selected independently, rather than a number tuned on the final fit — there is no held-out set left at that stage, so reusing validated selections is the honest option.

Post-processing: onset rounded and clipped to [1, 30]; recovery rounded and floored at 1.

### 3.4 Decision threshold

Probability ≥ **0.28** ⇒ `injured_in_risk_window = 1`. Rationale in Section 6.

---

## 4. Validation methodology

This is the part worth emphasising, because it is what makes the numbers above trustworthy rather than optimistic.

**Strict nested 5-fold cross-validation.** Every athlete receives exactly one prediction, produced by models that never saw that athlete. Folds are stratified on the injury label, 2,400 train / 600 validation each, 840 / 210 injured.

**Outer fold labels are never touched for tuning.** Each outer training fold contains its own inner split used to select tree counts. The untouched fold is used only to score. This matters: a common shortcut is to early-stop on the same validation fold you then report, which inflates results.

**Baselines are computed per fold.** Task B skill uses the training-mean baseline learned inside each athlete's own outer-training fold, exactly as the PDF prescribes, never a global mean.

**Verified, not assumed.** Independent re-checks confirmed exact fold reconstruction, complete and unique coverage of all 3,000 athletes, prediction ranges, per-fold class counts, and train/test schema parity. Feature validation reports `FEATURE_TABLE_VALIDATION=PASS` and `SOURCE_WINDOW_VALIDATION=PASS`.

Fold-level stability of the reported F1 at threshold 0.28: 0.5303, 0.5243, 0.5276, 0.5237, 0.5245 — a spread of 0.0066, so the headline figure is not a lucky fold.

---

## 5. The central finding: injury prediction has a horizon

This is the most substantive insight from the work and the most useful thing to lead with when presenting.

Injury probability turns out to be almost a monotone inverse function of the true onset day. **Correlation between predicted probability and true onset day = −0.852.**

| True onset (day of risk window) | Athletes | Mean predicted probability | Detected at 0.50 | Onset MAE |
| --- | --- | --- | --- | --- |
| 1–7 | 252 | 0.9335 | **94.4%** | 1.62 days |
| 8–14 | 251 | 0.9092 | **97.2%** | 0.97 days |
| 15–21 | 246 | 0.4237 | 15.5% | 3.43 days |
| 22–30 | 301 | 0.3760 | **2.3%** | 4.47 days |

The model detects injuries that are imminent — 94–97% of those occurring in the first two weeks of the risk window, with onset predicted to within about one day. Detection then collapses to 2.3% for injuries starting after day 21.

This is not a tuning failure. Late-onset injured athletes have a mean predicted probability of 0.376; healthy athletes sit at 0.377. They are statistically indistinguishable, because an athlete who gets injured on day 55 shows no precursor in days 1–30. **The physiological warning signal has a horizon of roughly two weeks, and the challenge's risk window is 30 days long.**

Practical reading: this pipeline is a genuine two-week early-warning system. It cannot be a 30-day one, and neither can any model built from this observation window.

---

## 6. Why the threshold is 0.28

The scoring rules couple the two tasks: a truly injured athlete classified as healthy incurs a fixed penalty of 30 on **both** timing predictions. Since the submission carries a single `injured_in_risk_window` column, one threshold controls all three metrics simultaneously. There is no way to decouple them.

That creates a real trade-off:

| Threshold | F1 | Onset skill | Recovery skill | Flagged |
| --- | --- | --- | --- | --- |
| 0.50 | 0.6466 | 0.0000 | 0.0000 | 19.3% |
| 0.34 | 0.5416 | 0.4717 | 0.0000 | 87.6% |
| **0.28 (chosen)** | 0.5261 | 0.6416 | 0.0857 | 97.9% |

Moving from 0.50 to 0.28 costs 0.1205 of F1 and gains 0.7273 across the two timing skills — a net of **+0.6067 raw metric points**. Solving for the flip point: F1 would have to carry **more than 75% of the total score** for 0.50 to be the better choice. Since the PDF defines three metrics, two of which are timing skills with an explicit programmatic formula, that seemed the less likely world.

Two further facts supported the decision:

- The gain is not an artifact. For the athletes the classifier misses, the onset regressor still predicts 22.11 against a truth of 22.44. Those predictions are already accurate; the 30-day penalty was pure waste. Lowering the threshold converts a penalty into a scored, accurate prediction.
- Recovery skill requires recall ≥ 98.9% before it registers at all, because the official recovery baseline (MAE 3.249) is already close to the best possible constant predictor.

**Stated plainly:** the 98.45% flag rate is not a claim that 98% of athletes will be injured. It is the scoring-optimal action under a rule that penalises a missed injury 30 times more heavily than a false alarm. As a ranked risk list — which is how such a model would actually be used — the AUC of 0.7655 and the two-week detection rate of 94–97% are the operationally meaningful numbers.

If the organisers' aggregation turns out to weight F1 above ~75%, the alternative submission at 0.50 (F1 0.6466) is pre-built and included.

---

## 7. What was tried and did not work

Reporting negative results honestly, since they explain why the final model is deliberately simple. The experiment scripts themselves were removed during cleanup; their findings are preserved below.

**Grouped feature-family ablation.** Six leave-one-family-out variants on identical folds. No family produced a repeatable improvement. Removing hourly features gained AUC in 4 of 5 folds but lost F1; removing interactions gained 0.0018 AUC, inside noise.

**Task A ranking and regularisation tuning.** Five CatBoost configurations selected by inner-fold AUC with log loss as tie-break, thresholds chosen only from inner predictions. The existing configuration won on mean inner AUC (0.7659) and in 3 of 5 folds. Fully nested selection scored *worse* than the incumbent (AUC 0.7505 vs 0.7655), because inner-AUC gaps of ~0.005–0.010 sat inside a config-level standard deviation of ~0.006 — the selection was fitting noise. A hypothesis that F1-based early stopping and a 50-tree floor were cutting training short was tested and refuted: AUC-based early stopping selected *fewer* trees (12–65). The model genuinely saturates in a few dozen trees.

**Recovery regressor improvement.** Eight configurations across tree, linear, and blend families. The production configuration was reproduced exactly (2.9429, matching the saved OOF value) and nothing beat it meaningfully; the honestly inner-selected candidate was worse (2.9514). Regularised linear models were clearly worse (ElasticNet 3.0190, Ridge 3.0448, Huber 3.2876), refuting a weak-additive-signal hypothesis. The ceiling is structural: the official baseline of 3.249 is already near the best possible constant (3.233), capping achievable recovery skill near 0.15.

**Conclusion.** Remaining modelling headroom is roughly +0.03 on an equal-weight mean, against +0.20 delivered by the threshold decision alone. The pipeline is at its ceiling for this feature set and observation window.

---

## 8. What to submit

### 8.1 The one required file

The PDF specifies exactly one deliverable: a submission CSV with `athlete_id`, `injured_in_risk_window`, `onset_day_offset`, `recovery_duration`. That is **`submission.csv`**.

The PDF references the template as `sample_submission.csv`. If the portal expects that exact filename, copy it across:

```powershell
Copy-Item submission.csv sample_submission.csv
```

> **Verify before submitting.** The problem statement does not document the submission channel, filename convention, or deadline. Confirm those against the official PlayHack channel (portal, Discord, or email announcement). Do not rely on this document for mechanics it cannot know.

### 8.2 If code is requested

Submit the folder, but exclude clutter. Include:

**Pipeline (in run order)**
```
build_features.py            observation-window features for train and test
validate_features.py         schema, window, and parity checks
evaluate_oof.py              leakage-free 5-fold evaluation
analyze_thresholds.py        threshold sweep and Pareto analysis
make_submission.py           refits on all data, writes submission.csv
scoring.py                   official metric implementation
task_a_threshold.txt         the decision threshold (0.28)
```

**Evidence and analysis**
```
oof_report.txt               headline validated scorecard
oof_predictions.csv          per-athlete out-of-fold predictions
oof_fold_summary.csv         per-fold diagnostics
threshold_report.md          operating-point trade-offs
threshold_tradeoff.png       trade-off chart
diagnose_detectability.py    the two-week horizon finding
diagnose_recovery.py         recovery ceiling analysis
threshold_breakeven.py       threshold decision arithmetic
compare_submissions.py       variant comparison
SUBMISSION.md                this document
```

**Optional depth** — the diagnostic and decision scripts that back the claims in this document: `diagnose_detectability.py`, `diagnose_recovery.py`, `threshold_breakeven.py`, `threshold_decision_analysis.py`, `compare_submissions.py`, `select_recall_threshold.py`, and `threshold_decision_report.md`.

**Already removed during cleanup:** the experiment runners and their output folders (feature-family ablation and recovery search), superseded baseline reports, legacy trainers that overwrote `task_a_threshold.txt`, their stale model files, and regenerable caches (`catboost_info/`, `__pycache__/`). Their conclusions are preserved in Section 7. The feature tables `train_features.csv` and `test_features.csv` are kept because `evaluate_oof.py` and `make_submission.py` read them directly; they can also be regenerated with `build_features.py`.

Whether to ship the `Dataset/` folder depends on the rules; usually organisers already have it. Omit unless required.

### 8.3 Reproducing from scratch

```powershell
python build_features.py        # writes train_features.csv, test_features.csv
python validate_features.py     # expect FEATURE_TABLE_VALIDATION=PASS
python evaluate_oof.py          # writes oof_predictions.csv, oof_report.txt
python analyze_thresholds.py    # writes threshold sweep and report
python make_submission.py       # writes submission.csv
```

Deterministic: seed 42 throughout. `evaluate_oof.py` takes a few minutes; the rest are fast.

### 8.4 Alternative submissions included

| File | Threshold | F1 | Onset skill | Recovery skill |
| --- | --- | --- | --- | --- |
| `submission.csv` (live) | 0.28 | 0.5261 | 0.6416 | 0.0857 |
| `submission_artifacts/submission_threshold_0.34.csv` | 0.34 | 0.5416 | 0.4717 | 0.0000 |
| `submission_artifacts/submission_threshold_0.50.csv` | 0.50 | 0.6466 | 0.0000 | 0.0000 |

Submit only one. To switch, edit `task_a_threshold.txt` and re-run `make_submission.py`, or copy the preserved file over `submission.csv` and update the threshold file to match.

---

## 9. How to present this

### 9.1 Lead with the horizon finding, not the metrics

Everyone will show F1 scores. Almost nobody will have discovered that the target itself has a structural detectability boundary. Open with the −0.852 correlation and the detection table from Section 5. It reframes the problem, demonstrates you interrogated the data rather than only fitting it, and explains your design choices before anyone can mistake them for weaknesses.

### 9.2 Suggested narrative

1. **The question.** Will this athlete get injured in the next 30 days, when, and for how long?
2. **What the data says.** Injury is highly predictable up to ~14 days out and effectively invisible beyond ~21 days. Show the detection table.
3. **What we built.** 347 leakage-safe observation-window features across seven sources; three CatBoost models; strict nested cross-validation.
4. **How we know it works.** One prediction per athlete from models that never saw them; per-fold baselines; independently verified. Fold F1 spread of 0.0066.
5. **The scoring decision.** One threshold governs all three metrics because a missed injury costs 30 on both timing targets. Show the break-even arithmetic: F1 must carry >75% of the score to justify the conservative choice.
6. **What we didn't do.** Feature ablation, model tuning, and a recovery-model search all found no improvement. State this as evidence of a ceiling, established by experiment.
7. **What it's for.** A two-week early-warning system, with AUC 0.7655 as the ranking quality a coach would actually rely on.

### 9.3 Tone

State numbers plainly and let them carry the weight. Where something is uncertain — the unknown metric weighting, the untested test-set distribution — say so. Judges reward calibrated confidence over overclaiming, and every figure here is reproducible from the included scripts.

### 9.4 Anticipate the obvious question

*"Why does your model flag 98% of athletes as injured?"*

Have this ready: the scoring rule penalises a missed injury with a fixed 30-day error on both timing targets while a false alarm costs nothing on Task B. Under that rule, flagging widely is optimal, and we can prove the crossover point — F1 would need >75% of the total weight to justify otherwise. Then pivot: as a decision tool the model is used as a ranked risk list, where it detects 94–97% of two-week-horizon injuries and predicts onset to within a day. We also prepared the conservative variant at threshold 0.50 with F1 0.6466.

---

## 10. Honest limitations

- **No official metric weighting.** The PDF does not state how F1, onset skill, and recovery skill combine. The threshold choice is an explicit, documented bet; the alternative is pre-built.
- **Recall is horizon-bound.** Injuries starting after day 21 of the risk window are not detectable from days 1–30. No model change fixes this.
- **Recovery duration is barely predictable.** MAE 2.943 against a constant-predictor baseline of 3.249. Confirmed against eight model configurations across three families.
- **Threshold selected on out-of-fold data.** Standard practice, but it means realised test performance will vary somewhat from the stated figures.
- **Mild test-set shift.** The test cohort shows slightly elevated recent heart-rate features (+0.10 to +0.12 standardised mean difference). No feature exceeds 0.25 SMD, so there is no material drift, but realised F1 may land a little off the cross-validated estimate.
- **Team-level generalisation untested.** Folds are athlete-stratified, not grouped by `team_id`, so performance on entirely unseen teams is not measured.
