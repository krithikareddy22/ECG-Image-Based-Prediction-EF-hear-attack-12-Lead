# Public-EF Amendment 1: image-only, all-feature EchoNext members (declared before running)

**Date:** 2026-11-06. Written before any of these configurations was run. The interim plan runs (stack-BASE / PLUS)
had not reported four-class stack results when this was written.

**Owner's request:** "make sure you extract all features possible and train it so well so it get the best accuracy of 4
class with images only not header or anything ... features should be extracted from images ... keep updating me
updated accuracy after each batch gets trained".

**Code:** `en_image.py` (members, stacks) and `en_stream.py` (interim updates per 50-MB download batch).

## 0. Inputs: image only
**Hospital ECGs:** every input comes from the digitised printout traces (12 leads × 2.5 s + 10-s lead II).

**EchoNext ECGs:** every input comes from the waveforms cut into the same printout format (`en_prep.printout`).

**Never used as inputs:**
- header text;
- printed machine measurements;
- age;
- sex. Sex only sets the label's Normal / Mild cut-off, as in the plan.

The plan's stack-BASE and stack-PLUS already satisfy this. C3 uses the header and stays a reference only.

## 1. Feature sets
All are computed with the same code for both sources, from (leads 12 × 1250, rhythm 5000) at 500 Hz.

| Set | Content | Size |
|---|---|---|
| G1 | ECGFounder f6, aligned + synth (as the plan) | 2 × 1024 |
| G2 | ECGFounder 150 logits, aligned + synth | 2 × 150 |
| G3 | CLIN: MILDFEAT clinician features. Settings as validated: QRS threshold 0.1, T-end median, P threshold 0.5. Signal-quality counts excluded, as in MILDFEAT. | 118 |
| G4 | Waveform families `qrs`, `st`, `hrv`, `freq`, `wavelet`, `leadstat` (Phase 5 extractor `one`) | ~ 400 |

**Signal-quality family `sq` excluded** from EchoNext-trained members. It measures digitisation and clipping artefacts
that differ between the sources. In a label-free check on 300 EchoNext rows, the median flat-sample share was 0.018 on
EchoNext vs 0.628 on the hospital ECGs.

**Label-free domain check before declaring (medians, EchoNext vs hospital):**
- QRS duration 82 vs 82 ms;
- QTc 415 vs 416 ms;
- R amplitude in V5 0.94 vs 0.93 mV.

The recovery of millivolts (published normalisation inverted) is therefore correct.

## 2. Harmonised hospital inputs (variant H)
EchoNext-trained members see hospital inputs prepared like EchoNext's own pipeline (IntroECG `parse_xml` / `preprocess`).
Each hospital trace (every lead segment and the rhythm strip) goes through:
1. 500 → 250 Hz by keeping every other sample;
2. baseline-wander removal (median filter 0.2 s, then 0.6 s, subtracted);
3. clipping at the published per-lead bounds × 4.88 µV;
4. back to 500 Hz (polyphase, as `en_prep`).

G1–G4 are then recomputed. The aligned input is built as for EchoNext (no PDF gap masking). Plain hospital inputs (no
harmonisation) are reported as a sensitivity analysis.

## 3. New EchoNext-trained members
These are fixed functions. Hyper-parameters are chosen on the plan's 80 / 20 patient-grouped EchoNext split
(`GroupShuffleSplit`, random_state 0); the member is then refitted on all EchoNext rows.
- **EN-cls-all:** class-balanced multinomial LR on block-standardised [G1 ‖ G2 ‖ G3 ‖ G4].
  - G3 / G4 missing values are imputed with the EchoNext training-row medians.
  - C ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}, chosen by held-out log-loss.
- **EN-mu-all:** ridge regression of EF on the same matrix, α from the plan's grid, chosen by held-out MSE.
  - **EN-band-all:** N(μ, σ̂²) integrated over the class bands, exactly as the plan's EN-band.
- **EN-gbm:** LightGBM, four classes, class-balanced, on [G2 ‖ G3 ‖ G4].
  - MILDFEAT's fixed settings: 7 leaves, learning rate 0.03, at most 400 rounds, subsample 0.8, colsample 0.5,
    min child 15.
  - Early stopping (50 rounds) on the held-out 20 %, then refitted on all rows with that number of rounds.

Internal held-out accuracy, per-class recall and EF r are reported for each.

## 4. Stacks and comparisons
Hospital nested protocol, meta-learner and ULT decision rule exactly as the plan, §2.
- **stack-BASE, stack-PLUS:** the plan, unchanged.
- **stack-ALL:** BASE members plus:
  - EN-cls and EN-band (the plan);
  - EN-cls-all, EN-band-all and EN-gbm (variant H);
  - μ_plan / 50 and μ_all / 50 as extra meta inputs.
- **stack-ALL-plain:** stack-ALL with plain hospital inputs for the new members (sensitivity).
- **EN-all zero-shot:** mean of EN-cls-all and EN-gbm probabilities, with the decision rule tuned on each
  outer-training fold. No hospital ECG trains a model.

**Comparisons:** corrected repeated-CV t-test over the 15 cells, Holm over the plan's and these comparisons.
- stack-ALL vs stack-BASE (primary for this amendment);
- stack-ALL vs stack-PLUS;
- stack-ALL vs C3.

**Reported for every configuration:** accuracy with corrected 95 % CI, per-class recall and precision, admissibility
(every recall ≥ 0.15).

**Selection:** the "best" model is the highest-accuracy admissible configuration declared here or in the plan, and is
labelled as selected after the fact. Nothing else is tuned on hospital outcomes.

## 5. Interim updates while downloading
After each 50-MB download batch, everything above is recomputed on the EchoNext rows so far (`en_stream.py`).
- **Interim results:** reported to the owner and kept in the git-ignored `data/private/interim/`.
- **Final results:** all downloaded EchoNext rows (val + test, plus train if downloaded); committed as aggregates only.

## 6. Stage 2: GPU fine-tuning (declared now; runs only when val and test are completely downloaded)
- **Model:** ECGFounder fine-tuned on the EchoNext printout-format inputs (aligned and synth, both as training
  samples).
  - Two heads: four-class class-balanced CE, plus EF regression (MSE on EF / 100, weight 1).
  - Trainable: the last two stages plus the heads.
- **Training:**
  - AdamW, learning rate 10⁻⁴, weight decay 10⁻², batch 32, at most 10 epochs, seed 0;
  - early stopping on the plan's 80 / 20 held-out loss (patience 2).
- **Application:** the fine-tuned network embeds the hospital ECGs (variant H) and gives FT-cls probabilities and FT-μ.
- **stack-FT:** stack-ALL + FT-cls + FT-band, compared with stack-ALL (corrected t, Holm).

**Status:** exploratory, reported separately from the plan's primary comparison.
