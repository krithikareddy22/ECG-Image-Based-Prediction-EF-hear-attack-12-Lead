# Phase 6B exploratory amendment: MILDFIX (one fix per reason Mild is not predicted)

**Date:** 2026-11-05. Written before any MILDFIX result. Owner request: "not only noise, rectify and find solutions and
implement them for all 5 reasons, and give final accuracies". Code: `mildfix.py`. Builds on `mmclean.py`
(MMCLEAN / NF) and on the existing nested members (MLP fusion ×3 seeds, ECGFounder linear, header + clinical LR).

## The five reasons (laptop diagnostic, aggregates; `MMCLEAN_REPORT.md`, `MILDFEAT_REPORT.md`)
1. True Mild ECGs are spread over all four predicted classes (C3: Normal 43 %, Mild 18 %, Moderate 28 %,
   Severe 11 %); many look Normal.
2. Neighbouring grades separate poorly (AUROC Mild-vs-Moderate 0.63, Moderate-vs-Severe 0.64) while distant ones
   separate well (Normal-vs-Severe 0.90).
3. The ECG's EF estimate (A1 regression, r 0.66) has an error SD of 13 points, wider than the 12-point Mild band.
   Only 29 % of true Mild ECGs get a predicted EF inside the band.
4. A ceiling follows: even a calibrated EF estimator with 13-point error gives at most about 0.63 accuracy and 0.34
   Mild recall.
5. Label noise: weak echo links (echo > 30 days, name score < 100) lower the ECG–echo EF correlation to 0.42–0.58.

## Fixes (all nested; the outer test fold never informs any choice)
Every tuned parameter is chosen per outer fold on the fold's training portion only: the nested inner out-of-fold
predictions (4-fold, `existing6b.inner_oof` splits) of the members.

- **F1, Mild-aware decision rule (reason 1).**
  - Rule: per-class logit offsets on the C1 vote probabilities (Phase 2 / 5 `decide(..., "bias")` grid, 729 offset
    vectors).
  - Tuning: the offsets that maximise macro-F1 on the inner OOF.
  - This moves borderline Normal predictions into Mild when that improves class balance.
- **F2a, ordinal decision (reason 2).**
  - Grade: the expected grade E = Σ k·p_k (Normal 0 … Severe 3) of the vote probabilities.
  - Cut points: c1 < c2 < c3 on a 0.05 grid, chosen on the inner OOF to maximise accuracy subject to every class
    recall ≥ 0.15 (the admissibility floor).
  - This uses the order of the classes, which a flat four-way argmax ignores.
- **F2b, ordinal model (reason 2).** Frank & Hall:
  - three class-balanced binary logistic regressions P(y > k) on [header ‖ ECGFounder f6 ‖ clinical], block
    standardised on the fitting rows;
  - C from {0.001, 0.01, 0.1, 1} chosen on inner-val (accuracy);
  - class probabilities = successive differences (clipped, renormalised); prediction = argmax. Standalone.
- **F3a, EF stacking (reason 3).**
  - EF regression: ridge (RidgeCV, α 10^-1…10^4) predicting the nearest-echo EF from [header ‖ ECGFounder f6 ‖
    clinical].
  - Outer: fitted on the fold's training rows.
  - Inner: 4-fold OOF on the training portion.
  - Stacker: a class-balanced multinomial logistic regression on [log vote probabilities ‖ EF̂/10], fitted on the
    inner OOF.
  - Decision: the nested class-prior rule, with α chosen on cross-fitted inner stacker predictions by the same
    floor objective as F2a.
- **F3b, EF grading (reason 3).** Classify by the nested ridge EF̂ alone:
  - Normal if EF̂ > sex boundary + d0;
  - Mild if EF̂ ≥ t1;
  - Moderate if EF̂ ≥ t2;
  - Severe otherwise.

  d0, t1 and t2 are chosen on the inner OOF EF̂ (0.5-point grid) by the floor objective. This is the "doctor grades by
  EF" route.
- **F4, abstention (reason 4).** Every final model above, and C3, is also reported with selective prediction:
  - a nested confidence threshold for target accuracy 0.70, as in `AMENDMENT_MMCLEAN.md`;
  - confidence = maximum final probability; for F3b, |EF̂ − nearest cut point|;
  - thresholds set on inner predictions (cross-fitted for F3a).
- **F5, noise-free training (reason 5).** NFS: the 412 Normal / Mild / Moderate ECGs with a noisy echo link are removed
  from every fit; Severe is kept. F1–F4 are run on the ORIG-trained members and on the NFS-trained members (including
  the ridge and the ordinal model, trained on NFS-kept rows only).

## Evaluation
- **Primary:** all 837 ECGs, original labels: accuracy (corrected CI), balanced accuracy, macro-F1, per-class recall,
  admissibility.
- **Descriptive:** NF_clear (264: noise-free link, not near a boundary), HC (333) and the NFS set (425).
- **Statistics:** paired corrected repeated-CV t-tests vs C3 trained on ORIG, on all 837 ECGs (accuracy, macro-F1,
  Mild recall), Holm-corrected over all MILDFIX comparisons. A genuine improvement needs Holm p < 0.05, the same sign
  in all repeats, and admissibility.
- **Final-model rule (fixed now):**
  - final four-class model = the admissible MILDFIX candidate with the highest mean accuracy on all 837 ECGs;
  - final abstention model = the one with the highest accuracy at the 0.70 target among those answering ≥ 20 %;
  - the best-Mild model (highest Mild F1) is also reported.

  Being the maximum over many candidates on the same development CV, each is optimistic. Confirmation would need the
  locked test set, which is the owner's decision and is not used here.
- **Status:** EXPLORATORY, reported separately (`MILDFIX_REPORT.md`). No model-dependent relabelling or removal.
