# Phase 6B exploratory amendment: BANDS (sweep of the four-class EF band limits)

**Date:** 2026-11-05. Written before any run, at the owner's request: "change the band sizes, sweep from the minimum to
the optimum, and say where I get the maximum accuracy with Mild and Moderate detected and a reasonable data size".

**Status: sensitivity analysis.** Changing the band limits changes the task, so accuracies under different limits are
not the same quantity. The original labels remain the primary result (C3 0.561; header-free 0.569 / 0.570). This sweep
is reported separately, and every definition is shown with its chance level.

## Definitions
- Labels are re-derived from the stored nearest-echo EF (`label_table().ef`, 837 development ECGs):
  - **Normal:** EF > c_N for men, EF ≥ c_N + o for women;
  - **Mild:** EF ≥ c_Mi;
  - **Moderate:** EF ≥ c_S;
  - **Severe:** otherwise.
- **Reference ORIG-EF:** the original limits on the same EF (c_N 52, o 2, c_Mi 41, c_S 31). It agrees with the folder
  labels for 93.4 % of ECGs.
- **Grid:** c_N ∈ {45, …, 56}; c_Mi ∈ {35, …, 46}; c_S ∈ {25, …, 36}; o ∈ {0, 2}.
- **Named clinical point ESC-2021:** Normal ≥ 50, Mild 41–49, Moderate 30–40, Severe < 30 (c_N 49.9, o 0, c_Mi 41,
  c_S 30).
- **Feasible** ("reasonable size"):
  - every class ≥ 60 ECGs;
  - Mild band (c_N − c_Mi) ≥ 5 EF points;
  - Moderate band (c_Mi − c_S) ≥ 5.

## Sweep model (the same for every definition; header-free)
- **EF prediction:** ŷ, from a ridge regression of EF on [aligned ECGFounder f6 ‖ synth ECGFounder f6 ‖ waveform
  features], the M6 inputs of `AMENDMENT_ENH.md`.
  - α ∈ {1, 10, 100, 10³, 10⁴}, chosen by inner-OOF RMSE.
  - Nested inner OOF and outer OOF use the same splits as ENH.
  - ŷ does not depend on the definition.
- **EF grading:** three cut points on ŷ (grid 20–70, step 1), chosen per outer fold on the inner OOF to maximise
  accuracy, subject to every class recall ≥ 0.15. They are applied to the outer predictions.
- **Per definition** (mean over 3 repeats):
  - accuracy, per-class recall ("% detected"), macro-F1;
  - Cohen's κ (chance-corrected), majority-class accuracy and chance accuracy Σ p_c².

## Reported points
1. **One-dimensional sweeps:** each limit varied with the other two at their original values.
2. **MAX-ACC-DET:** the feasible definition with the highest accuracy among those with Mild recall ≥ 0.25 and Moderate
   recall ≥ 0.25.
3. **MAX-KAPPA:** the feasible definition with the highest κ, i.e. the best agreement beyond chance.
4. ORIG-EF and ESC-2021.

## Confirmation with retrained classifiers
For points 2–4 the classifier is retrained on the relabelled classes:
- **Members:** M2 (synth ECGFounder f6 ‖ logits) and M5 (aligned f6 ‖ waveform features) LR, plus EF-band probabilities
  from ŷ, using N(ŷ, σ̂²) integrated over the definition's bands with σ̂ the inner-OOF residual SD.
- **Combination:** an equal-weight vote with nested class weights, as in ENH.
- The original folder-label models (C3 0.561, ENH-vote 0.569) are shown alongside.

## Caveat
Points 2 and 3 are selected on the same development data over about 2,600 definitions, so their values are
optimistic. They describe which definition the ECG separates best. They are not confirmed performance. The locked test
set is not used.
