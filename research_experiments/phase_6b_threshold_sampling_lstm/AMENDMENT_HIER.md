# Phase 6B exploratory amendment: HIER (two-stage, header-free four-class model)

**Date:** 2026-11-05. Written before any run, at the owner's request: "first do Normal / middle / Severe, then
classify the middle into Mild and Moderate; give the combined accuracy as a single model; without the header".

## Inputs (header-free, as in `AMENDMENT_IMGONLY.md`)
No printed machine measurements, no age and no sex. The Phase-3 clinical family `clin3` is excluded as well, because
part of it was derived with header values. Allowed:
- **IMG-vote probabilities:** the two header-free ECGFounder heads of IMG-ONLY (MLP on f6, 3 seeds; linear
  distance-CE head). Outer OOF and the cached nested inner OOF are reused.
- **ECGFounder f6 embedding** (1024) from the digitised waveform.
- **Waveform features:** `qrs`, `st`, `hrv`, `freq`, `wavelet`, `leadstat`, `sq` and `logits150`, the 150 ECGFounder
  diagnosis outputs. Block-standardised with f6, missing values imputed with the training-fold median.

## Model (one pipeline = one model)
- **Stage 1, Normal / Middle (Mild + Moderate) / Severe.** The mean of:
  - the IMG-vote probabilities collapsed to three classes;
  - a class-balanced multinomial logistic regression on [f6 ‖ features].
- **Stage 2, Mild vs Moderate.** Trained only on the Mild and Moderate ECGs of the training folds. The mean of:
  - the IMG-vote ratio P(Moderate) / (P(Mild) + P(Moderate));
  - a class-balanced binary logistic regression on [f6 ‖ features].
- **Combined:** if stage 1 says Middle, stage 2 decides Mild or Moderate. Four-class probabilities are
  [P_N, P_mid (1 − q), P_mid q, P_S].
- **Tuned parameters,** all chosen inside the outer training folds:
  - logistic C ∈ {0.001, 0.003, 0.01, 0.03, 0.1}, by log-loss on inner OOF;
  - stage-1 class weights w_mid, w_sev ∈ {0.6, 0.7, …, 2.0};
  - stage-2 threshold t ∈ {0.30, 0.32, …, 0.70}.

  Inner OOF uses the Phase-5 nested protocol: patient-grouped 4-fold inner CV with the same seeds as the cached IMG
  inner OOF.
- **Decision objective (inner OOF):** maximise four-class accuracy subject to every class recall ≥ 0.15 and macro-F1 ≥
  the untuned rule's (w = 1, t = 0.5) minus 0.01. If no setting qualifies, the untuned rule is used.
- **Primary:** HIER (vote + LR in both stages). **Secondary:** HIER-vote, the same two-stage decision on the IMG-vote
  probabilities only, without the LR.

## Evaluation
- All 837 development ECGs, original labels, Phase-2 folds (5 × 3).
- Reported:
  - four-class accuracy, balanced accuracy, macro-F1 and per-class recall / precision;
  - stage-1 three-class accuracy;
  - stage-2 accuracy on the true Mild / Moderate ECGs.
- **Paired corrected t-tests (15 cells, Holm)** against IMG-C3 (header-free) and C3 (with header).
- **Status:** EXPLORATORY, reported separately. The locked test set is not used.

## Extension 1 (2026-11-05, declared after the pre-registered run, before running the extension)
In the pre-registered run the logistic C was the top of its grid (0.1) in all 15 outer folds, and the stage-1 weights
were often the bottom of theirs (0.6). The grid, not the data, limited the model. Extension 1 (`hier.py --ext`) changes
only the grids: C ∈ {0.1, 0.3, 1, 3, 10}; w_mid, w_sev ∈ {0.3, 0.4, …, 2.0}. Everything else, including the nested
selection inside the training folds, is unchanged. Because it was declared after seeing the first run's outer results,
it is reported as a second, separate model (HIER-ext), not as a replacement.
