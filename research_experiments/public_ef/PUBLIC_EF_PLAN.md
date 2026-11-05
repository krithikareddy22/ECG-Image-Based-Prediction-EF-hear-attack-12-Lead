# Public-EF study: learning ECG → EF from EchoNext, formatted like the hospital printouts (pre-registration)

**Date:** 2026-11-05. Written before any run. Branch `research/public-ef`, kept separate from `research/phase6` at the
owner's request ("do this separately in a branch").

**Owner's goal:** improve four-class LVEF classification of the hospital ECG images (837 development ECGs, unchanged)
by learning from a larger public ECG–echo dataset. The dataset under study stays the same; the public data only teach
the model.

## Data-use rules (PhysioNet credentialed licence; non-negotiable)
- EchoNext files stay on the owner's laptop in `C:\krithika_major\public_data\echonext\`, outside the repository
  and outside OneDrive.
- No EchoNext row, waveform, identifier or derived per-ECG array is committed, pushed, uploaded or printed. Caches go
  to the git-ignored `research_experiments/public_ef/data/private/`.
- Committed: code, plus aggregate metrics (counts, accuracies, AUROCs).
- Models trained on EchoNext stay private and are not committed.
- The scripts refuse an `ECHONEXT_DIR` inside the repository.

## EchoNext inputs (metadata columns verified from the header only)
- **Metadata:** `echonext_metadata_100k.csv`, with `lvef_value` (continuous EF), `sex`, `patient_key` and `split`.
- **Waveforms:** `EchoNext_<split>_waveforms.npy`, shape N × 1 × 2500 × 12 (10 s at 250 Hz). The lead order is I, II,
  III, aVR, aVL, aVF, V1–V6, in the order of the metadata rows of that split (as in the public EchoNext-Mini
  notebook).
- **Splits used first:** `val` and `test` (about 2.2 GB). `train` (16 GB) is optional, for a later run.
- **Labels:** the project's own rule applied to `lvef_value` and sex:
  - **Normal:** EF > 52 (men) / ≥ 54 (women);
  - **Mild:** ≥ 41;
  - **Moderate:** 31 to < 41;
  - **Severe:** < 31.
- **Excluded:** rows without `lvef_value` or with sex other than male / female.

## Making EchoNext look like the hospital printouts
Per ECG:
1. **Recover millivolts.** If the arrays are the normalised model inputs (per-lead SD ≈ 1), invert the published
   EchoNext-Mini normalisation (z-score → MUSE units → × 4.88 µV). Values the publishers clipped stay clipped.
2. **Resample** 250 → 500 Hz (polyphase).
3. **Cut into the printout layout:** I / II / III from 0–2.5 s, aVR / aVL / aVF from 2.5–5 s, V1–V3 from 5–7.5 s,
   V4–V6 from 7.5–10 s; lead II for the full 10 s as the rhythm strip.
4. **Build the two inputs** used for the hospital ECGs:
   - "aligned" (`ecgfounder.inputs.build`; the PDF gap masking is not applicable);
   - "synth" (`enh_build.build_one`, the beat-synchronous 10-s reconstruction).
5. **Embed** each with the frozen ECGFounder: f6 and the 150 logits.

The hospital ECGs go through the same steps, from their digitised printout traces. Both sources therefore enter the
model in the same format.

## Models trained on EchoNext only
- **EN-cls:** class-balanced multinomial LR on [aligned f6 ‖ synth f6] → four-class probabilities.
- **EN-mu:** ridge regression of EF on the same inputs → predicted EF μ.
- **Hyper-parameters:** C ∈ 10^{−4…1} (step 0.5) and α ∈ {1, 10, 100, 10³, 10⁴}, chosen on an 80 / 20
  patient-grouped split of the EchoNext rows. The models are then refitted on all EchoNext rows.
- **Internal check (aggregate):** accuracy, per-class recall and EF r on the held-out 20 %.

## Evaluation on the 837 hospital development ECGs (Phase-2 folds, 5 × 3; locked test not used)
1. **Zero-shot transfer:** EN-cls argmax on all 837. Also EN-cls with class weights tuned on each outer-training
   fold, using the ULT decision rule: maximise accuracy with every recall ≥ 0.18, fallback 0.15. Also the AUROC
   of μ for echo EF ≤ 45, Normal vs Mild, and Mild vs Moderate.
2. **Added value (primary):** stack-PLUS vs stack-BASE.
   - **BASE members:** M1 IMG-vote, M5 and M6 (ENH definitions), with nested inner OOF.
   - **PLUS members:** BASE plus EN-cls and EN-band. EN-band = N(μ, σ̂²) integrated over the class bands, with σ̂
     the residual SD on the outer-training rows. The EchoNext members are fixed functions, so their inner values
     are simply their predictions on the training rows.
   - **Meta-learner:** multinomial LR on the members' log-probabilities and μ / 50; C chosen by patient-grouped
     4-fold meta-CV.
   - **Decision rule:** as in ULT.
   - **Test:** paired corrected t-test over 15 cells (PLUS vs BASE, PLUS vs C3).
3. **Reported:** accuracy (corrected 95 % CI), per-class recall and precision, admissibility (every recall ≥ 0.15).

**Status:** exploratory. Results are compared with C3 (0.561), ULT-stack (0.571) and ULT2-stack (0.569).

## Amendment 2 (2026-11-06): EchoNext test split only
**Owner:** "for my phase 1 results lets only use test file....val and train file lets do some other day".

**Changed:** the final run of this plan and of Amendment 1 (`AMENDMENT_1_IMAGE_ONLY.md`) uses the EchoNext `test`
split only (5,442 ECGs). `val` and `train` are deferred to a later run.

**Unchanged:** everything else. Interim updates (`en_stream.py`) already use the test split only.

## Amendment 3 (2026-11-06): train / test split inside the EchoNext test file
**Owner:** "save few in test for test also...split accordingly"; "i mean split train test accordingly as u will only be
given test file".

**Split:** the EchoNext `test` file is divided by patient (`en_split.locked`):
- **EchoNext locked test:** 20 % of patients, those with sha256(`patient_key`) mod 5 = 0. The assignment depends only
  on the patient, so it is identical in interim and final runs.
- **EchoNext train:** the other 80 %.

**Use of the locked test:**
- Never used for training, standardisation, early stopping, hyper-parameter choice or class weights.
- It is only scored: accuracy, per-class recall, EF r.
- It is shown at every interim update; nothing is chosen with it.

**Use of the train part:**
- All EchoNext models (the plan's EN-cls / EN-mu and Amendment 1's members) are fitted only on the train part.
- The plan's patient-grouped 80 / 20 selection split is now taken inside the train part.

**Code:** `en_split.py`; `en_train.train_en` and `en_image.fit_en_all` exclude locked patients.

**Unchanged:** the hospital evaluation (837 development ECGs, Phase-2 folds).
