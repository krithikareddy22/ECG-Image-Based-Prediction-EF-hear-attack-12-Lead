# Phase 6B exploratory amendment: MILDFIX2 (round 2: attack the EF error itself)

**Date:** 2026-11-05. Written before any MILDFIX2 result. Owner request: "let's rectify these even more, the 5 problems
... try your best". Round 1 (`MILDFIX_REPORT.md`) showed two things:
- four-class accuracy is capped by an EF error of about 13 points SD (A1 13.2; ridge 13.4; 11.7 on clean echo links);
- the best usable result was EF grading with abstention (0.694 answering 52 %).

Round 2 therefore attacks the EF estimate directly. Code: `mildfix2.py`.

## EF regressors
All are nested: outer out-of-fold predictions, plus 4-fold inner OOF on each fold's training portion with the
`existing6b.inner_oof` splits. The target is the nearest-echo EF.

| Id | Model | Inputs |
|---|---|---|
| R1 | ridge (RidgeCV α 10^-1…10^4); round-1 baseline | [header ‖ ECGFounder f6 ‖ clinical], block-standardised |
| R2 | ridge | R1 inputs ‖ the 121 image-only clinician features (`mildfeat.py`; in-fold median imputation), block-standardised |
| R3 | LightGBM regression | [header ‖ clinical ‖ image clinician features ‖ ECGFounder PCA-32 fitted on the fit rows] |
| R4 | equal-weight mean of R2 and R3 | — |

R3 settings are fixed:
- learning rate 0.03, at most 600 rounds;
- 7 leaves, min child 15, subsample 0.8, colsample 0.5;
- early stopping (50 rounds) on a grouped 20 % split of the fit rows.

**Noise-aware training (reason 5).** Each regressor is trained three ways:
- **W1:** unweighted;
- **W.3:** rows with a noisy echo link (outside the NF set of `AMENDMENT_MMCLEAN.md`) get weight 0.3;
- **W0:** clean-link rows only.

That gives 4 × 3 = 12 variants.

**Nested selection (SEL).** In each outer fold, the variant with the lowest inner-OOF RMSE on the clean-link rows of
the training portion is selected. Clean-link rows are used because their targets are the most trustworthy. The
selected variant gives the outer predictions; no outer label is used.

## Classification and abstention (ORIG-trained members; reference C3, 0.562)
- **G2, EF grading:** as F3b, with the SEL regressor. Cut points d0 / t1 / t2 are tuned on the inner OOF by the floor
  objective (accuracy subject to every recall ≥ 0.15).
- **G2s, EF stacking:** as F3a, with the SEL regressor. A balanced multinomial LR on [log vote probabilities ‖
  EF̂ / 10], plus the nested prior rule.
- **AG, agreement abstention (reason 4):** answer only when C3 and G2 predict the same class. No tuning.
- **AGT:** AG, plus a nested threshold on the G2 confidence (|EF̂ − nearest cut|) for target 0.70.
- **G2-T:** G2 with the nested confidence threshold for target 0.70, as round 1.

## Reporting
- **EF error:** outer out-of-fold residual SD, MAE and r on all 837 ECGs and on clean-link ECGs, for every variant
  and for SEL. The A1 + SEL average is also reported, as an outer-only EF predictor with no tuning.
- **Four-class:** G2 and G2s on all 837 (primary), NF_clear and HC. Paired corrected t-tests vs C3 (ORIG), with Holm
  correction.
- **Abstention:** coverage, accuracy (corrected CI), class coverage, recall among answered, and the predicted mix.
- **Final rules (fixed now):**
  - final four-class = the admissible candidate with the highest all-837 accuracy;
  - final abstention = the option with the highest coverage among those whose mean accuracy on answered ECGs is
    ≥ 0.70. If none qualifies, the highest accuracy among options answering ≥ 20 % is used.

  All are maxima over candidates on the development CV, so they are optimistic. The locked test set is not used.
- **Status:** EXPLORATORY, reported in `MILDFIX2_REPORT.md`. No model-dependent relabelling or removal.
