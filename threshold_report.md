# Leakage-Free OOF Threshold and Pareto Report

## Scope

This report uses only `oof_predictions.csv`. Every athlete was predicted by models that did not train on that athlete. Task B skill uses the baseline mean learned from that athlete's outer-training fold.

The PDF defines **F1**, **onset skill**, and **recovery skill**, but does not define how they are weighted into one leaderboard number. Therefore this report presents trade-offs rather than claiming one official winner. The current threshold remains `0.28` and was not changed.

## Operating-point candidates

| Candidate | Threshold | F1 | Precision | Recall | Onset skill | Recovery skill | Predicted injured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current Reference | 0.28 | 0.5261 | 0.3570 | 0.9990 | 0.6416 | 0.0857 | 0.9793 |
| F1 Focused | 0.50 | 0.6466 | 0.9086 | 0.5019 | 0.0000 | 0.0000 | 0.1933 |
| Balanced | 0.28 | 0.5261 | 0.3570 | 0.9990 | 0.6416 | 0.0857 | 0.9793 |
| Timing Focused | 0.27 | 0.5255 | 0.3564 | 1.0000 | 0.6449 | 0.0942 | 0.9820 |
| Recall Focused | 0.38 | 0.5873 | 0.5084 | 0.6952 | 0.0000 | 0.0000 | 0.4787 |
| Maximum Recall Endpoint | 0.27 | 0.5255 | 0.3564 | 1.0000 | 0.6449 | 0.0942 | 0.9820 |

## Candidate definitions

- **Current Reference:** Current temporary threshold; included for comparison only.
- **F1 Focused:** Highest Task A F1; ties prefer better timing skill, then recall.
- **Balanced:** Pareto point maximizing the worst relative attainment across the three PDF-defined metrics. Relative attainment divides each metric by its best observed OOF value; this is a decision aid, not an official formula.
- **Timing Focused:** Highest mean of onset and recovery skill; ties prefer higher F1. This protects Task B but is not an official combined score.
- **Recall Focused:** Highest recall among thresholds retaining at least 90% of the best observed F1; ties prefer higher F1 and timing skill.
- **Maximum Recall Endpoint:** Maximum possible recall on the sweep, shown as an extreme endpoint rather than a recommended operating point.

## Pareto frontier

A threshold is on the Pareto frontier when no other tested threshold is at least as good on all three PDF-defined metrics and strictly better on at least one.

| Candidate | Threshold | F1 | Precision | Recall | Onset skill | Recovery skill |
| --- | --- | --- | --- | --- | --- | --- |
| Pareto | 0.27 | 0.5255 | 0.3564 | 1.0000 | 0.6449 | 0.0942 |
| Pareto | 0.28 | 0.5261 | 0.3570 | 0.9990 | 0.6416 | 0.0857 |
| Pareto | 0.31 | 0.5274 | 0.3590 | 0.9933 | 0.6239 | 0.0391 |
| Pareto | 0.32 | 0.5303 | 0.3626 | 0.9867 | 0.6005 | 0.0000 |
| Pareto | 0.33 | 0.5359 | 0.3694 | 0.9762 | 0.5651 | 0.0000 |
| Pareto | 0.34 | 0.5416 | 0.3790 | 0.9486 | 0.4717 | 0.0000 |
| Pareto | 0.35 | 0.5487 | 0.3948 | 0.8990 | 0.3015 | 0.0000 |
| Pareto | 0.36 | 0.5614 | 0.4230 | 0.8343 | 0.0847 | 0.0000 |
| Pareto | 0.50 | 0.6466 | 0.9086 | 0.5019 | 0.0000 | 0.0000 |

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
