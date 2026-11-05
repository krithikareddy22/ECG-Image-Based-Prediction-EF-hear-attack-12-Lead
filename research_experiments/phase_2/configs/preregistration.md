# Phase-2 pre-registration (written before any phase-2 candidate model was evaluated)

Date: 2026-11-04, before the first phase-2 CV result. The only job running at that point was the
fixed-recipe baseline image CNN. SHA-256 of this file is recorded in `preregistration.sha256`.

## Data and protocol
- Development set **D**: original train + validation, 837 ECGs / 794 patients.
- Locked test set **T**: original test split, 210 ECGs / 199 patients.
- T was inspected in phase 1, so it is **exposed, not untouched**. No phase-2 selection uses T.
- 5-fold patient-grouped stratified CV on D (`protocol.py`, repeat 0 = primary):
  - identical folds for every model;
  - every base model is trained on inner-train and selected / early-stopped on inner-val of its
    fold;
  - post-hoc parameters (fusion weights, stackers, decision rules) are cross-fitted on the OOF
    predictions of the other four folds.
- Cheap models are also run on CV repeats 1–2; deep models on extra seeds.

## Metrics
- Pooled OOF over D (n = 837): accuracy, balanced accuracy, macro-F1, per-class F1 (Mild F1),
  QWK, macro OvR AUROC.
- Per-fold mean ± SD.
- 95 % patient-cluster bootstrap CIs.

## Selection score and targets
- **Within-family hyper-parameter / variant selection:** pooled-OOF CV score
  **S3 = mean(accuracy, balanced accuracy, macro-F1)**.
- **Aspirational targets** (not guaranteed): accuracy ≥ 0.60, balanced accuracy ≥ 0.50,
  macro-F1 ≥ 0.48, Mild F1 ≥ 0.35, QWK ≥ 0.50.

## Reference model R
The current default architecture, re-fitted in the same CV:
- header multinomial logistic regression (original recipe);
- ResNet-18 image CNN (original recipe);
- late probability fusion with cross-fitted weight.

## Final selection (applied once, on CV results only)
1. **Best overall** = highest CV S3 among the candidates. If ΔS3 < 0.005, the model with fewer
   inputs and parameters wins.
2. **Accuracy-priority model** = highest CV accuracy among candidates with
   - balanced accuracy ≥ R − 0.01,
   - macro-F1 ≥ R − 0.01,
   - Mild F1 ≥ R − 0.02.
3. **Shortlist** for the locked test: R + up to 5 models (top 5 by CV S3, always including
   models 1 and 2).

## Inference
- **Primary test for a CV comparison of candidate B vs reference A:** Nadeau–Bengio corrected
  resampled t-test on the 5 per-fold differences.
- **Secondary:** patient-cluster paired bootstrap on pooled OOF.
- **Holm correction** within each metric over this family:
  - F1: best frozen-ECGFounder model vs image CNN (waveform vs image representation)
  - F2: header numeric measurements vs image CNN
  - F3: header + ECGFounder vs R
  - F4: header + image + ECGFounder vs R
  - F5: best fusion/stacking model vs R
  - F6: best ordinal head vs multinomial (same input)
  - F7: best imbalance method vs class-weighted CE (same input)
  - F8: fine-tuned vs frozen ECGFounder (same head)
- All other comparisons are descriptive.

## Locked test
- Scored at most once, for R + the shortlist, **only after the user authorises it**.
- Reported as "previously exposed hold-out".
- Locked-test probabilities of every base model (mean of its 5 fold models) are written to
  `locked_test/` during CV but not scored before then.

## Forbidden
- Using T for any choice.
- Using the EF spreadsheets.
- Synthetic samples in T.
- Duplicating patients across splits.
