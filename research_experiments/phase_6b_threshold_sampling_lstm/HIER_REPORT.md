# Phase 6B exploratory report: HIER, two-stage header-free four-class model

Declared in `AMENDMENT_HIER.md` (pre-registered grid; Extension 1 declared after the first run). Code `hier.py`.
Results `results/hier_exploratory.json` (pre-registered) and `results/hier_ext_exploratory.json` (Extension 1).
All 837 development ECGs (Normal 384, Mild 185, Moderate 177, Severe 91), original labels, patient-grouped 5-fold CV ×
3 repeats. All tuning is nested inside the training folds. The locked test set was not used. Status: **EXPLORATORY**.

**Model:** stage 1 decides Normal / Middle (Mild + Moderate) / Severe; stage 2 decides Mild vs Moderate inside Middle.
Both stages are the mean of the header-free ECGFounder vote and a logistic regression on [ECGFounder f6 ‖ waveform
features ‖ 150 ECGFounder diagnosis outputs]. There is no header, age or sex input. The pipeline is evaluated as one
four-class model.

| Model | Accuracy [95 % CI] | Macro-F1 | Recall N / Mi / Mo / S | Precision N / Mi / Mo / S | Stage-1 3-class acc. | Stage-2 acc. (true and predicted Middle) |
|---|---|---|---|---|---|---|
| HIER (pre-registered grid) | 0.558 [0.522, 0.594] | 0.439 | 0.862 / 0.173 / 0.411 / 0.344 | 0.719 / 0.270 / 0.447 / 0.326 | 0.660 | 0.550 |
| HIER-vote (pre-registered grid) | 0.554 [0.504, 0.603] | 0.445 | 0.809 / 0.204 / 0.505 / 0.282 | 0.732 / 0.306 / 0.418 / 0.347 | 0.662 | 0.582 |
| **HIER-ext (Extension 1 grid)** | **0.570 [0.527, 0.612]** | 0.448 | 0.884 / 0.162 / 0.433 / 0.341 | 0.702 / 0.281 / 0.465 / 0.390 | 0.671 | 0.554 |
| HIER-vote-ext | 0.559 [0.519, 0.598] | 0.430 | 0.888 / 0.175 / 0.386 / 0.286 | – | 0.664 | 0.531 |
| IMG-C3 (header-free reference) | 0.559 [0.506, 0.613] | 0.444 | 0.847 / 0.191 / 0.452 / 0.300 | 0.713 / 0.292 / 0.441 / 0.353 | – | – |
| C3 (with header, Phase 5 selection) | 0.561 [0.520, 0.603] | 0.448 | 0.845 / 0.177 / 0.460 / 0.344 | 0.710 / 0.297 / 0.455 / 0.344 | – | – |

- **Per-repeat accuracy, HIER-ext:** 0.575 / 0.576 / 0.559.
- **HIER-ext confusion matrix** (mean over repeats; rows = label, columns = prediction N / Mi / Mo / S):
  - Normal 339 / 26 / 12 / 6;
  - Mild 86 / 30 / 49 / 20;
  - Moderate 40 / 36 / 77 / 25;
  - Severe 18 / 15 / 27 / 31.
- **Extension 1 tuning:** the logistic C is now interior (stage 1: 1.0 in all 15 folds; stage 2: 0.3 in 12 folds).

**Comparisons (accuracy, corrected t-test over 15 cells, Holm over 4):**
- HIER-ext vs IMG-C3: +0.011 (p 0.47, Holm 1.00).
- HIER-ext vs C3: +0.009 (p 0.38, Holm 1.00).
- Both differences have the same sign in all 3 repeats. Not a genuine improvement.

## Interpretation
- **Best header-free point estimate:** HIER-ext gives the highest header-free four-class accuracy on all 837 ECGs so
  far (0.570). It is admissible: Mild recall 0.162 ≥ 0.15. The gain over IMG-C3 and C3 (+0.01) is within noise, and it
  comes from Normal (recall 0.88), not from Mild (0.16).
- **Optimism:** Extension 1 was declared after seeing the first run, so this figure is optimistic. It is reported
  separately.
- **Where the two-stage design fails:** stage 1 works (three-class accuracy 0.66–0.67). Stage 2 does not: on ECGs that
  are truly Mild or Moderate and reach stage 2, it is right 53–58 % of the time, close to a coin flip. This matches the
  factor study (`HC30_REPORT.md`): Mild vs Moderate AUROC ≤ 0.64 for every factor family. Splitting the decision does
  not add information. The bottleneck is the ECG signal for Mild vs Moderate.
