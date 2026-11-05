# Phase 6B exploratory report: HC30 high-confidence-label cohort and extra Mild factors

Declared before running in `AMENDMENT_HC30.md`; code `hc30.py`; results `results/hc30_exploratory.json`,
`results/mild_factor_study.json`, member aggregates `results/experiments/P6B-*-HC30-v1__*.json`. Development set only,
patient-grouped 5-fold CV × 3 repeats, original labels. The locked test set was not used. Status: **EXPLORATORY**.

## Cohort
HC30 = exact name match (score 100), ECG–echo interval ≤ 30 days (true dates), no conflicting class within 30 days:
**395 ECGs** (Normal 188, Mild 87, Moderate 80, Severe 40). Training and every nested choice (inner validation, prior
rule, α) use only HC30 rows of the training folds; outer test folds are unchanged.

## Results (means over 3 repeats)
| Model | Trained on | Scored on | Accuracy | Macro-F1 | Recall N / Mi / Mo / S |
|---|---|---|---|---|---|
| C3 (reference) | all 837 | HC30 (395) | 0.598 | 0.491 | – / 0.245 / – / – |
| HC30 vote | HC30 | HC30 (395) | 0.590 | 0.489 | 0.828 / 0.241 / 0.467 / 0.475 |
| HC30 C3 rule | HC30 | HC30 (395) | 0.594 | 0.486 | 0.848 / 0.245 / 0.475 / 0.400 |
| HC30F vote (+ all-factors LR) | HC30 | HC30 (395) | 0.599 | **0.504** | 0.835 / 0.249 / 0.471 / 0.508 |
| **HC30F C3 rule** | HC30 | HC30 (395) | **0.603** | 0.492 | 0.865 / 0.253 / 0.471 / 0.392 |
| HC30 C3 rule | HC30 | all 837 | 0.564 | 0.452 | 0.855 / 0.196 / 0.427 / 0.348 |
| HC30F C3 rule | HC30 | all 837 | 0.569 | 0.452 | 0.873 / 0.196 / 0.418 / 0.333 |

Per-repeat accuracy, HC30F C3 on HC30: 0.610 / 0.610 / 0.587.

## Interpretation
- Training only on the 395 reliable-label ECGs does not beat training on all 837: 0.594 vs 0.598 on the same ECGs.
- Adding the all-factors member (QRS, ST, HRV, frequency, wavelet, lead statistics, signal quality, 150 ECGFounder
  diagnosis outputs) gives +0.005 accuracy and +0.013 macro-F1. That is far inside the run-to-run and fold uncertainty
  (about ±0.05 at n = 395); it is not a genuine improvement.
- Mild recall stays at 0.24–0.25 in every variant. The factor study explains why: Mild separates from Normal
  moderately (best single factor QRS width, AUROC 0.73) but from Moderate only weakly (≤ 0.64).
- Every model was among 4 variants on the same development CV, so the best value is slightly optimistic.
