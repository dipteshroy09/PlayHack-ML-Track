# Threshold Choice Under Unknown Aggregation

The challenge PDF (page 6) defines Task A F1, onset skill, and recovery skill
separately and never states how they combine. Organizer clarification is not
available, so the operating point is chosen under uncertainty.

Metrics come from the validated leakage-free OOF sweep (`threshold_sweep.csv`).
Regret is the shortfall against the best threshold *within each scenario*.

## Optimal threshold per scenario

| Scenario | Weights (F1 / onset / recovery) | Best threshold |
| --- | --- | --- |
| f1_only | 1.00 / 0.00 / 0.00 | 0.50 |
| f1_heavy | 0.70 / 0.15 / 0.15 | 0.27 |
| equal_thirds | 0.33 / 0.33 / 0.33 | 0.27 |
| timing_heavy | 0.20 / 0.40 / 0.40 | 0.27 |

## Worst-case regret shortlist

| Threshold | F1 | Onset | Recovery | Recall | Flagged | Max regret |
| --- | --- | --- | --- | --- | --- | --- |
| 0.34 | 0.5416 | 0.4717 | 0.0000 | 0.9486 | 0.8760 | 0.1050 |
| 0.33 | 0.5359 | 0.5651 | 0.0000 | 0.9762 | 0.9250 | 0.1107 |
| 0.32 | 0.5303 | 0.6005 | 0.0000 | 0.9867 | 0.9523 | 0.1163 |
| 0.31 | 0.5274 | 0.6239 | 0.0391 | 0.9933 | 0.9683 | 0.1192 |
| 0.28 | 0.5261 | 0.6416 | 0.0857 | 0.9990 | 0.9793 | 0.1205 |
| 0.30 | 0.5259 | 0.6322 | 0.0619 | 0.9962 | 0.9760 | 0.1207 |
| 0.29 | 0.5258 | 0.6382 | 0.0769 | 0.9981 | 0.9787 | 0.1208 |
| 0.27 | 0.5255 | 0.6449 | 0.0942 | 1.0000 | 0.9820 | 0.1211 |
| 0.26 | 0.5251 | 0.6449 | 0.0942 | 1.0000 | 0.9830 | 0.1215 |
| 0.25 | 0.5240 | 0.6449 | 0.0942 | 1.0000 | 0.9860 | 0.1227 |

- Minimax-regret threshold: **0.34** (worst-case regret 0.1050).
- Current saved threshold 0.28 has worst-case regret 0.1205.

This is a decision aid, not an official score. `task_a_threshold.txt` is unchanged.
