# Image-only route: an image CNN pre-trained on EchoNext ECGs drawn as hospital printouts (pre-registration)

**Date:** 2026-11-05. Written before any EchoNext image is drawn or any model is trained.
- **Branch:** `research/public-ef`. The data-use rules of `PUBLIC_EF_PLAN.md` apply unchanged.
- **Owner's request:** "classify using an image only".

**Question.** Does pre-training an **image** CNN on thousands of EchoNext ECGs, drawn exactly like the hospital
printouts, improve four-class LVEF classification of the hospital ECG **images**?

## Images
- **Hospital, 837 development ECGs only; the locked test set is not touched.**
  1. Page 1 of each PDF.
  2. **All text removed** (name, header, measurements, lead labels).
  3. The waveform area `ecg_ef.config.WAVEFORM_CLIP_PT` rasterised at 0.6 px per pt.
  4. The original project's grid filter (`ecg_ef.preprocess.filter_trace`) applied.
  5. Result: a 320 × 640 uint8 image (white trace on black).

  The model sees pixels only.
- **EchoNext:** each ECG (12 leads × 10 s, 250 Hz, recovered to mV as in `en_prep.py`) is cut into the printout
  layout:
  - columns I/II/III 0–2.5 s, aVR/aVL/aVF 2.5–5 s, V1–V3 5–7.5 s, V4–V6 7.5–10 s;
  - the 10-s lead II rhythm strip.

  It is drawn at the hospital coordinates (rows 707 / 970 / 1233 / 1496 pt, columns 75 / 588 / 1102 / 1615 pt,
  205.2 pt/s, 82 pt/mV, black 2-pt line). The black page furniture (calibration pulses, column ticks) is copied
  from a hospital page. It is then rasterised and filtered exactly as above (`img_render.py`).
- **Fidelity (measured before this plan; development ECGs, aggregates only):** a hospital ECG redrawn from its
  digitised traces matches its real image.
  - pixel correlation: median 0.993, minimum 0.986;
  - trace-mask IoU: median 0.967, minimum 0.946;
  - 30 ECGs.
- **Labels:** EchoNext `lvef_value` and sex, with the project's four-class rule. Rows without LVEF or with other sex
  are excluded.

## Models (pixels only; no header, age or sex input)
- **Backbone:** ResNet-18, ImageNet-pretrained (torchvision). The grey image is repeated to 3 channels.
- **Heads:** four-class (class-balanced cross-entropy) plus an EF-regression head ((EF − 50) / 15, MSE, weight 0.5).
- **Augmentation:** small random translation (±2 %), scale (0.95–1.05) and brightness / contrast jitter.
- **EN (pre-training):** trained on all EchoNext images.
  - Early stopping on a patient-grouped 10 % EchoNext hold-out (macro-F1; patience 3; at most 15 epochs).
  - AdamW, learning rate 3·10⁻⁴, weight decay 10⁻⁴, batch 32, mixed precision.
  - Seed 42.
- **Hospital evaluation:** Phase-2 folds, 5 × 3, all 837 ECGs.
  - **ZS (zero-shot):** EN applied to every hospital image. Class weights are tuned on each outer-training fold with
    the ULT decision rule (maximise accuracy; every recall ≥ 0.18, fallback 0.15).
  - **FT-EN:** EN fine-tuned on each outer-training fold.
    - Inner-train / inner-val = the Phase-2 inner split.
    - Early stopping on inner-val macro-F1; patience 4; at most 25 epochs; learning rate 10⁻⁴.
    - Class weights tuned on inner-val with the same rule.
  - **FT-IN (reference):** identical to FT-EN but starting from ImageNet only, learning rate 3·10⁻⁴. This is the
    pixel-CNN baseline; the original project's ResNet-18 scored 0.471 in CV.
  - **Seed:** 42 only (compute). Results are exploratory and not "eligible" in the Phase 6 sense.

## Reported (aggregates only)
- **EchoNext hold-out:** accuracy, per-class recall, EF r.
- **Hospital, ZS / FT-EN / FT-IN:** accuracy (corrected 95 % CI), per-class recall and precision, macro-F1,
  admissibility (every recall ≥ 0.15).
- **Primary test:** FT-EN vs FT-IN, a paired corrected t-test over 15 cells, which isolates EchoNext image
  pre-training.
- **Secondary tests:** FT-EN vs C3 (with header); Holm over the 2.
- **Context:** the best image-derived models so far are ULT-stack 0.571 (traces read from the image) and the triage
  model (0.694 on 56 % answered).

## Not allowed
- Training or selecting on the hospital locked test set.
- Any per-ECG or EchoNext-derived file in git. Rendered EchoNext images are EchoNext-derived data: they stay in
  `research_experiments/public_ef/data/private/`.
- Relabelling to model outputs.

## Amendment 1 (2026-11-06): EchoNext test split only
Added before any EchoNext image was drawn and before any model was trained on EchoNext.

**Owner:** "for my phase 1 results lets only use test file....val and train file lets do some other day".

**Changed:**
- **EchoNext images:** the `test` split only (`img_prep --echonext --splits test`). That is 5,442 ECGs, of which those
  with LVEF and male / female sex are used.
- **EN pre-training:** the patient-grouped 10 % hold-out is taken from these ECGs. All other settings are unchanged.

**Unchanged:** image size, model, augmentation, epochs, learning rates, early stopping, the decision rule and the
hospital evaluation.

**Run order:** the FT-IN folds were run before EN pre-training. They use the identical code path (`img_train.Net`,
`train` and `predict`, seed 42, the same per-fold cache files that `--evaluate` reuses), and they use no EchoNext
data.

**Laptop fidelity check** (step 4; 30 development ECGs, `img_fidelity.py`; mask = pixels > 127):
- pixel correlation: median 0.993, minimum 0.988;
- trace-mask IoU: median 0.966, minimum 0.949.

## Amendment 2 (2026-11-06): train / test split inside the EchoNext test file
Added before any EchoNext image was drawn or any model was trained on EchoNext.

**Owner:** "save few in test for test also...split accordingly as u will only be given test file".

**Split:** the patients with sha256(`patient_key`) mod 5 = 0 (20 %, `en_split.locked`; the same patients as in
PUBLIC_EF_PLAN Amendment 3) form the EchoNext locked test.

**Use of the locked test:**
- Excluded from EN pre-training, including its 10 % early-stopping hold-out, which is now taken from the other 80 %.
- Scored once after pre-training (`results/img_pretrain.json`, `echonext_locked_test`): accuracy, per-class recall
  and precision, macro-F1, EF r.

**Unchanged:** everything else.

## Amendment 3 (2026-11-06): IMG70, towards 70 % four-class accuracy on the hospital images
Declared before any of these experiments ran. **Code:** `img70.py`.

**Owner:** "make sure you are aiming for 70 plus accuracy..you can try all you want if it doesnt work analyse why and
rectify or fix and try again".

**Validity rules:**
- original hospital labels;
- all 837 development ECGs, Phase-2 folds 5 × 3;
- every choice made inside the training folds;
- no relabelling, no dropping of ECGs;
- the hospital locked test set is closed;
- the classifier input is the page image only (text removed, waveform area).

**EchoNext data:** the test file only, using its 80 % train part. The locked 20 % is never used to choose anything.

**E1 (all EchoNext data, train split): deferred by the owner.** Reported as not run.

**Images for E2–E4:** 384 × 768. The same rendering, text removal and grid filter as IMG_PLAN (`img_render`,
0.6 px per pt), for the 837 hospital images and the EchoNext train / locked images.

**Pre-training (all variants):** as IMG_PLAN.
- AdamW, learning rate 3·10⁻⁴, weight decay 10⁻⁴, at most 15 epochs, patience 3.
- Effective batch 32: micro-batch 8 × 4 gradient-accumulation steps. Mixed precision. Seed 42.
- Early stopping on a patient-grouped 10 % hold-out of the train part (macro-F1).

**Hospital fine-tuning (all variants unless stated):** as IMG_PLAN FT-EN.
- Learning rate 10⁻⁴, at most 25 epochs, patience 4.
- Phase-2 inner-train / inner-val split. Class weights chosen on inner-val by the ULT rule (maximise accuracy, every
  recall ≥ 0.18, fallback 0.15).

**E2 (stronger backbone, higher resolution):**
- Two candidates: ConvNeXt-Tiny and EfficientNet-B2, both ImageNet-initialised. Loss: CE + 0.5 · MSE(EF).
- Both are pre-trained. The backbone with the higher EchoNext hold-out macro-F1 is chosen.
- Only the chosen backbone is fine-tuned and evaluated on the hospital images.

**E3 (EF as the main pre-training target):**
- The E2-chosen backbone. Pre-training loss: 0.5 · CE + 1.0 · MSE(EF).
- On the hospital images, the class head is combined with EF grading:
  P = ½ · (P_class + P_EF), where P_EF = N(μ̂, σ²) integrated over the class bands.
  - Bands: Normal limit 52 (men) / 54 (women), 41 and 31, all shifted by δ.
  - σ is the SD of the inner-val EF residuals.
  - δ ∈ {−4, −2, 0, 2, 4}, chosen on inner-val accuracy (after the decision rule).
  - The class weights are then chosen on inner-val.

**E4 (fine-tuning recipe):**
- Starting point: from E3 if E3's mean inner-val accuracy exceeds E2's, else from E2.
- Learning rates: backbone 10⁻⁴, heads 10⁻³ (AdamW, cosine).
- Mixup with α = 0.2 on images, class targets and EF targets.
- Ensemble: the mean of seeds 42, 43 and 44.
- Test-time augmentation: identity plus ±1 % shifts in x and y, averaged. It is switched on or off in each fold on
  inner-val accuracy.
- E4 inherits E3's decision (EF grading) if it starts from E3.

**Final model (headline):**
- **Choice:** among IMG FT-EN (the IMG_PLAN baseline), E2, E3 and E4, the highest mean inner-val accuracy after the
  decision rule, over the 15 folds. FT-IN is a reference only.
- **Seeds:**
  - If E2 or E3 is chosen, seeds 43 and 44 are also run, and the mean over the 3 seeds is reported.
  - If E4 is chosen, its ensemble and the mean of its 3 single seeds are reported.
- **Tests:** the final model vs C3, and vs IMG FT-EN. Corrected t-test over 15 cells (seed-mean cells), Holm.

**E5 (reported separately as "image-derived: pixels + traces read from the image"):**
- Amendment 1's stack-ALL plus one fixed-function member: the final backbone's EchoNext-pre-trained network (no
  hospital training) applied to the hospital images. It contributes class probabilities and an EF band
  (N(μ, σ̂²); σ̂ from outer-training residuals).
- Being a fixed function, its inner values are its predictions on the training rows.
- ULT stacking and decision rule as in Amendment 1.

**E6 (reported separately): triage with the final model.**
- In each outer fold, the threshold on the maximum probability is the lowest whose inner-val answered accuracy is
  ≥ 0.70 with ≥ 20 answered. It is applied unchanged to the outer fold.
- **Reported:** share answered and accuracy on the answered ECGs, by class.
- **Labelled descriptive, after the fact:** the largest outer share at which the answered accuracy is ≥ 0.70.

**Reported:** every variant run, with accuracy [corrected 95 % CI], per-class recall and precision, macro-F1 and
admissibility. Plus the plain sentence "70 % four-class accuracy on all patients was / was not reached", with the best
honest number.

A step that cannot finish in the available time is reported as not run. It is never replaced by a cheaper version
without a new amendment.
