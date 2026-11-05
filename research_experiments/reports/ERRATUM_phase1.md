# ERRATUM to `FINAL_REPORT.md` (phase 1). Read this before citing any phase-1 result.

`FINAL_REPORT.md` is kept unchanged as the historical record, so that earlier results are never
silently overwritten. **The statements below supersede it.**

## E1. Retracted: "header + ECGFounder vs header in 5-fold CV, p < 0.001"

| | Phase-1 statement (retracted) | Corrected (phase-2 audit) |
|---|---|---|
| Δ accuracy, pooled OOF, 1,047 ECGs | +6.6 points, "p < 0.001" | +6.6 points, **p = 0.12** |
| Δ balanced accuracy | +4.8 points, "p = 0.017" | +4.8 points, **p = 0.33** |
| Δ macro-F1 | +4.9 points, "p = 0.009" | +5.1 points, **p = 0.22** |
| Method | patient-cluster bootstrap of pooled out-of-fold predictions | Nadeau–Bengio corrected resampled t-test on the 5 per-fold differences |

The bootstrap resampled only the test predictions and ignored the variability that comes from
re-training on different folds, so it understated uncertainty.

**Correct statement:** "In phase-1 cross-validation, header + ECGFounder had higher accuracy than the
header model (+6.6 points), but the difference was not statistically significant under a corrected
resampled t-test (p = 0.12)."

Source: `phase_2/results/audit_previous.json`.

## E2. Other phase-1 statements that need qualification

| Phase-1 text | Qualification |
|---|---|
| "p = 0.0" in bootstrap tables | means p < 0.0005 (1/2000 resamples), not zero |
| "the one convincing result is in cross-validation" (§0) | **retracted:** see E1; no phase-1 comparison is statistically convincing |
| CV over 1,047 ECGs as independent evidence | that CV contained the 210 fixed-split test ECGs, so it is **not independent** of the test results |
| fine-tuned ECGFounder vs image CNN | the ECGFounder row is a 3-seed ensemble vs a single-seed CNN, so the comparison favours ECGFounder |
| test-set results of all phase-1 models | the test set was inspected for ~40 models, so it is **exposed**; the best test score across models is optimistically biased |

## E3. Dataset count wording

"669" is the **original training split**, never the total. The total is **1,047 ECGs / 993
patients**: train 669 + validation 168 + test 210. See `phase_3/results/dataset_accounting.json`.

Future reports must use the corrected values above. `phase_3/reports/CLAIMS_REGISTER.md` tracks the
status of every claim.
