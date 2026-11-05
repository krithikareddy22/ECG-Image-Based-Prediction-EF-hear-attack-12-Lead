# Phase 6B exploratory report: MILDFIX2 (round 2: better, noise-aware EF regressors)

Development set (837 ECGs), patient-grouped 5-fold CV × 3 repeats, original labels; the locked test set was not used.
Declared before running in `AMENDMENT_MILDFIX2.md` (commit ccfd3ef). Code: `mildfix2.py`. Results:
`results/mildfix2_results.json`, `mildfix2_ef_error.csv`, `mildfix2_results.csv`, `mildfix2_selective.csv`,
`mildfix2_comparisons.csv`. Plot: `plots/mildfix2_summary.png`. All figures are aggregates (means over 3 repeats;
corrected repeated-CV 95 % CIs). Status: **EXPLORATORY**.

## Answer
- **The EF error did not come down.**
  - The best EF predictor, the mean of A1 (fine-tuned ECGFounder) and the nested-selected regressor, has a residual SD
    of **13.2** points on all 837 ECGs (MAE 10.3, r 0.66) and 11.8 on clean echo links.
  - A single model is at 13.3–13.9.
  - About 9 is needed for 70 % four-class accuracy.
- **Four-class accuracy did not improve.**
  - EF grading reaches 0.558 [0.512, 0.604]; EF stacking reaches 0.551 [0.511, 0.591].
  - Both fall below the 0.15 Mild-recall floor, so neither is admissible. Against C3: Holm p 1.00.
  - C3 stays best at 0.562.
- **Abstention got better coverage.**
  - **AGT** answers only when C3 and EF grading agree *and* the predicted EF is far enough from a cut-off. It scores
    **0.698 [0.652, 0.744] while answering 57 %** of ECGs, which is **39.5 correct answers per 100 ECGs**.
  - Before: round-1 EF grading gave 36.4 per 100 (0.694 at 52 %); C3 with noise-free training gave 24.5 (0.702 at
    35 %).
  - By the rule fixed in advance (largest coverage with mean accuracy ≥ 0.70), the final abstention model stays C3
    with noise-free training, at 0.702 [0.643, 0.790] answering 35 %. AGT misses that cut by 0.002, well inside its CI.

## EF error by regressor (out-of-fold, mean of 3 repeats)
| Regressor | Noise treatment | SD, all 837 | MAE | r | SD, clean links |
|---|---|---|---|---|---|
| R1 ridge (round-1 baseline) | none / 0.3 / clean-only | 13.50 / 13.49 / 13.89 | 10.69 / 10.58 / 10.88 | 0.640 / 0.643 / 0.631 | 11.97 / 11.89 / 12.25 |
| R2 ridge + 121 image clinician features | none / 0.3 / clean-only | 13.51 / 13.46 / 13.79 | 10.75 / 10.56 / 10.82 | 0.640 / 0.645 / 0.632 | 12.27 / 11.99 / 12.11 |
| R3 LightGBM | none / 0.3 / clean-only | 13.62 / 13.58 / 13.78 | 10.79 / 10.67 / 10.70 | 0.632 / 0.635 / 0.624 | 12.51 / 12.32 / 12.43 |
| R4 mean of R2 and R3 | none / 0.3 / clean-only | 13.36 / **13.30** / 13.48 | 10.64 / 10.43 / 10.49 | 0.650 / 0.654 / 0.644 | 12.16 / 11.90 / 11.93 |
| SEL (nested choice per fold) | — | 13.53 | 10.56 | 0.642 | 12.14 |
| A1 (fine-tuned ECGFounder, image only) | — | 13.36 | 10.46 | 0.650 | 11.93 |
| **A1 + SEL mean** | — | **13.22** | **10.31** | **0.659** | **11.77** |

Nested selection picked R1 / W0.3 in 5 folds, R4 / W0.3 in 5, R4 / clean-only in 3, R1 / clean-only in 1 and
R2 / clean-only in 1.

**What moved:**
- down-weighting noisy links (0.3) helps a little: −0.01 to −0.06 SD;
- averaging different models helps a little: R4 13.30 and A1 + SEL 13.22, against 13.50.

**What did not:**
- training on clean links only, which loses data (+0.2 to +0.4 SD);
- adding the 121 image clinician features (±0.05);
- non-linear LightGBM (+0.1).

The earlier "13.2" for A1 was the mean of the 3 repeats' predictions. A single A1 run is 13.36.

## Four-class (G2 EF grading, G2s EF stacking)
| Model | All 837 | Macro-F1 | Recall N / Mi / Mo / S | NF_clear (264) | HC (333) |
|---|---|---|---|---|---|
| C3 (reference) | **0.562** [0.519, 0.605] | 0.449 | 0.85 / 0.18 / 0.46 / 0.34 | 0.707 | 0.619 |
| G2 EF grading (SEL regressor) | 0.558 [0.512, 0.604] | 0.422 | 0.88 / 0.14 / 0.47 / 0.23 | 0.698 | 0.600 |
| G2s EF stacking | 0.551 [0.511, 0.591] | 0.436 | 0.84 / 0.13 / 0.41 / 0.48 | **0.710** | 0.650 |

Paired against C3, all 837 ECGs:
- G2: accuracy −0.004 [−0.052, +0.044]; Mild recall −0.04;
- G2s: accuracy −0.011 [−0.047, +0.024];
- every Holm p is 1.00.

## Abstention (answer only when confident; the rest referred to echo)
| Option | Answered | Accuracy on answered | Correct per 100 ECGs | Answered by true class N / Mi / Mo / S | Recall among answered N / Mi / Mo / S | Predicted N / Mi / Mo / S |
|---|---|---|---|---|---|---|
| AG: C3 and EF grading agree (no tuning) | 73 % | 0.641 [0.588, 0.696] | **46.5** | 87 / 66 / 61 / 50 % | 0.95 / 0.07 / 0.51 / 0.24 | 435 / 32 / 113 / 27 |
| **AGT: agree + nested confidence threshold** | **57 %** | **0.698 [0.652, 0.744]** | **39.5** | 79 / 48 / 35 / 23 % | 0.99 / 0.00 / 0.46 / 0.19 | 404 / 1 / 58 / 11 |
| G2T: EF grading + nested threshold | 56 % | 0.694 [0.640, 0.751] | 38.8 | 78 / 47 / 31 / 30 % | 0.99 / 0.00 / 0.41 / 0.23 | 404 / 1 / 51 / 12 |
| Round 1: F3b EF grading + threshold | 52 % | 0.694 [0.648, 0.750] | 36.4 | 73 / 41 / 29 / 34 % | 0.99 / 0.00 / 0.34 / 0.32 | 375 / 1 / 43 / 20 |
| Round 1: C3, noise-free training (declared final) | 35 % | 0.702 [0.643, 0.790] | 24.5 | 48 / 26 / 22 / 21 % | 0.98 / 0.09 / 0.23 / 0.49 | 236 / 7 / 16 / 33 |
| Round 1: C3 + threshold | 25 % | 0.670 [0.564, 0.839] | 16.4 | 32 / 19 / 17 / 18 % | 0.96 / 0.06 / 0.28 / 0.57 | 153 / 6 / 19 / 27 |

**Clinical caveat.** In AGT, G2T and F3b, about half of Mild patients are *answered*, and they are answered wrongly:
- Mild recall among answered is 0.00 in all three;
- the answer is mostly "Normal".

A triage built on these would confidently miss mild LV dysfunction. C3 with noise-free training answers fewer patients
but at least occasionally flags Mild (recall 0.09 among answered) and more often Severe (0.49).

## Conclusion after two rounds
- Every modelling route tried tops out at the same place:
  - label cleaning, noise weighting, ordinal models, EF regression (linear, non-linear, ensembles), image-only
    clinician features, decision rules and agreement;
  - the ECG–echo EF error stays at about 13 points;
  - four-class accuracy on all ECGs stays at about 0.56.
- **The bottleneck is the information in this dataset:** printed-PDF ECGs, Teichholz EF linked by name, 837 ECGs.
  The modelling is not what limits it.
- **Abstention is the usable gain:** answering about 57 % of ECGs at about 70 % (mostly Normal, some Moderate),
  referring the rest. It must not be presented as Mild detection.
- **Phase 2 needs a ~4-point smaller EF error.** It needs more patients, the raw digital ECG, patient-ID echo linkage
  within days, and Simpson biplane EF.

**Optimism:** the best rows are maxima over many candidates on the same development CV. Confirmation needs the locked
test set, which is the owner's decision and is not used here.
