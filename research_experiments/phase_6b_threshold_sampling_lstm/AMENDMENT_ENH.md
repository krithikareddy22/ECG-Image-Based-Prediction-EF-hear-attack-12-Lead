# Phase 6B exploratory amendment: ENH (new image-only features, better pre-processing, learned combination)

**Date:** 2026-11-05. Written before any run, at the owner's request:
- improve four-class accuracy without the header;
- separate Normal, Mild and Moderate better, so they are mistaken for each other less often;
- extract more from the ECG image and pre-process it better;
- try several approaches, report per-class detection, and explain the errors.

**Not repeated,** because they were already tried and failed (Phase 6, 6A, 6B and the laptop's Phase 7):
- multi-scale ECGFounder layers (stage-4 / stage-5 / max / std), per-printed-column embeddings, and the 150 ECGFounder
  outputs used alone;
- multitask EF, soft labels and contrastive heads;
- 1-D ResNet, LSTM and CNN-LSTM;
- cascades, HIER and selective Mild;
- the 121 clinician features.

## Inputs
Image only. No header, printed measurements, age or sex, and no `clin3`. Development ECGs only (837). The locked test
set is not used.

### P1. Pre-processing: beat-synchronous 10-s 12-lead reconstruction ("synth")
- **The problem:** the printout gives each lead only 2.5 s, in four successive columns. Only lead II covers 10 s (the
  rhythm strip, simultaneous with all columns). ECGFounder was pre-trained on simultaneous 10-s 12-lead ECGs. The
  current "aligned" input zero-fills 7.5 s of every lead except II, a strong domain shift.
- **The fix:**
  - **R peaks:** detected on the band-passed rhythm strip (derivative-energy envelope, 0.3-s refractory period, peak
    refined to the largest |amplitude| within ±50 ms).
  - **Templates:** for every lead, a median R-to-R cycle template is built from the complete cycles inside its own
    2.5-s column. The window starts min(0.25 s, 0.35·RR) before R and has the median RR length; it is band-passed in
    the ECGFounder way.
  - **Filling the 10 s:** each lead is filled over the full 10 s by placing its template at every rhythm-strip R peak.
    The P-QRS-T part (up to min(R + 0.45 s, 0.85 · interval)) keeps its shape; only the remainder (TP segment) is
    resampled to each RR interval, so heart-rate variation changes diastole, not the QRS.
  - **Recorded samples:** inside the lead's own column the recorded signal replaces the template, with a 20-ms linear
    cross-fade at both edges. Lead II is the recorded rhythm strip.
  - **Scaling:** one z-score over all 12 leads (as for "aligned"), which keeps the inter-lead amplitude ratios.
- **Fidelity check (aggregate only):** median correlation between the template-synthesised signal and the recorded
  signal inside each lead's own column, and the fraction of ECGs with ≥ 1 complete template cycle per lead.

### P2. New representations
| Code | Representation |
|---|---|
| F-syn | ECGFounder (frozen) on the synth input: mean-pooled stage-6 embedding f6 (1024) ‖ the 150 output logits |
| F-beat | Synthesised median beat: the 12 lead templates resampled to 100 samples per cycle (1,200 values; morphology itself) |
| F-rocket | MiniRocket (multivariate, aeon 1.6.0, about 10k features, random_state 0) on the synth input at 100 Hz (12 × 1000), fitted without labels on each outer-training fold |
| F-wave | Existing waveform families `qrs`, `st`, `hrv`, `freq`, `wavelet`, `leadstat`, `sq` and `logits150` ("aligned"), as in HIER |

MiniRocket runs in a separate virtual environment (aeon needs an older numpy). Only the features cross over, through a
git-ignored file.

## Members
Each member gives four-class probabilities: outer OOF plus nested inner OOF on the outer-training portion, using the
same patient-grouped 4-fold inner splits as the cached IMG members (random_state 1000 + 10 r + k).

| Member | Model |
|---|---|
| M1 | IMG-vote: ECGFounder MLP + linear distance-CE on the aligned f6 (existing, cached) |
| M2 | Class-balanced multinomial LR on F-syn |
| M3 | LR on F-beat |
| M4 | LR on F-rocket |
| M5 | LR on [aligned f6 ‖ F-wave] |
| M6 | EF-ordinal: ridge regression of EF on [aligned f6 ‖ F-syn f6 ‖ F-wave]. Class probabilities come from N(μ̂, σ̂²) integrated over the bands < 31 / 31–41 / 41–53 / ≥ 53, with σ̂ the inner-OOF residual SD. EF is a training target only, and 53 replaces the sex-specific 52 / 54 cut because sex is not an input. |

- **LR penalty:** C ∈ {10⁻⁴, 3·10⁻⁴, …, 10} (11 values), chosen by inner-OOF log-loss.
- **Ridge penalty:** α ∈ {1, 10, 100, 10³, 10⁴}, chosen by inner-OOF RMSE.
- **Block scaling:** multi-block inputs are block-standardised; MiniRocket and F-beat are standardised on the training
  rows.

## Combiners (one complete four-class model each)
- **ENH-vote:** equal-weight mean of M1–M6.
- **ENH-stack (primary):** multinomial LR (C = 1, L2) on the concatenated log-probabilities of M1–M6.
  - It is trained on the members' inner OOF.
  - Its own OOF on the training portion (patient-grouped 4-fold) is used for the decision rule.
- **Decision rule (both):** class weights w_Mild, w_Moderate, w_Severe ∈ {0.4, 0.5, …, 2.5} (w_Normal = 1). They are
  chosen on training-portion OOF to maximise accuracy, subject to every class recall ≥ 0.15 and macro-F1 ≥ the untuned
  rule's minus 0.01.
- **Diagnostics (not candidates):** each member alone with argmax.

## Evaluation
- **Data:** all 837 ECGs, original labels, Phase-2 folds 5 × 3.
- **Reported:**
  - accuracy (corrected 95 % CI), balanced accuracy, macro-F1;
  - per-class recall ("% detected") and precision;
  - the confusion matrix;
  - **neighbour confusion rates** (Normal→Mild, Mild→Normal, Mild→Moderate, Moderate→Mild, Moderate→Severe,
    Severe→Moderate) and the share of errors ≥ 2 classes away.
- **Tests:** paired corrected t-tests (15 cells) of ENH-stack and ENH-vote against IMG-C3, C3 (with header) and
  HIER-ext; Holm over the 6 tests.
- **Error analysis, "why" (descriptive):** error rate of ENH-stack and C3 by
  - distance of the echo EF from the nearest class boundary (< 3, 3–6, 6–10, ≥ 10 points);
  - echo-link quality (HC30 vs the rest);
  - digitisation quality (`quality_score` value bins);
  - position within the Mild band.
- **Status:** EXPLORATORY. Reported separately; the best value is a maximum over a few combiners and is optimistic.

## Diagnostic D1 (2026-11-05, declared after seeing the member diagnostics, before computing D1)
To isolate the effect of the P1 pre-processing, `M2ref` fits exactly the M2 head on the "aligned" ECGFounder input:
block-standardised [aligned f6 ‖ aligned 150 logits], the same LR, the same C grid and the same inner splits. M2 vs
M2ref is a diagnostic of the pre-processing only. M2ref is not a combiner member and not a candidate.
