# Prompt for the laptop Claude Code session (EchoNext public-EF study)

Paste everything below the line into Claude Code on the laptop (repository clone `C:\krithika_major\ecg-lvef`, venv
`C:\krithika_major\venv`).

---

Run the public-EF study on branch `research/public-ef`. The pre-registration is
`research_experiments/public_ef/PUBLIC_EF_PLAN.md`; read it first and follow it exactly.

## Hard rules (PhysioNet credentialed data-use agreement + project privacy rules)
- EchoNext data stay in `C:\krithika_major\public_data\echonext\`, outside the repository and outside OneDrive.
- Never commit, push, upload, paste or print any EchoNext row, waveform, identifier, or derived per-ECG array or model.
  Commit only code and aggregate JSON (counts, accuracies, AUROCs).
- Never upload EchoNext files to any online service, including other chats.
- The hospital's locked 210-ECG test set must not be loaded or scored.
- Before every push:
  - run the privacy scan (`research_experiments.phase_5.p5common.privacy_scan`) on the staged files;
  - check that the staged diff contains only `research_experiments/public_ef/` code, docs and `results/*.json`;
  - push only to `research/public-ef`.

## Steps
1. **Get the code:**
   ```
   git fetch origin research/public-ef
   git checkout research/public-ef
   ```
   Activate the venv.
2. **Get the EchoNext files.** In `C:\krithika_major\public_data\echonext\` you need:
   - `echonext_metadata_100k.csv`
   - `EchoNext_val_waveforms.npy`
   - `EchoNext_test_waveforms.npy`
   - `README.md`
   - `LICENSE.txt`

   Read `README.md` and confirm three things:
   - the waveform arrays are N × 1 × 2500 × 12 at 250 Hz, lead order I, II, III, aVR, aVL, aVF, V1–V6;
   - each split's waveform rows follow that split's metadata rows in file order;
   - whether the arrays are already normalised.

   If anything differs from the plan, stop and tell the owner.
3. **Get the published EchoNext-Mini normalisation parameters** (code repository, not data):
   ```
   git clone --depth 1 https://github.com/PierreElias/IntroECG.git C:\krithika_major\public_data\IntroECG
   ```
4. **Make sure the hospital-side inputs exist:**
   - `research_experiments/phase_6b_threshold_sampling_lstm/data/private/enh/build.npz` — if missing, run
     `python -m research_experiments.phase_6b_threshold_sampling_lstm.enh_build`;
   - the Phase 6 inner caches and the Phase 6A labels — restore them per `research_experiments/phase_6/PHASE6_RESUME.md`
     if missing.
5. **Prepare EchoNext in the printout format:**
   ```
   set ECHONEXT_DIR=C:\krithika_major\public_data\echonext
   set ECHONEXT_MINI_DIR=C:\krithika_major\public_data\IntroECG\7-EchoNext Minimodel
   python -m research_experiments.public_ef.en_prep --splits val,test
   ```
   Check `results/en_prep_summary.json` (aggregate counts only):
   - the input was detected as normalised or raw as the README says;
   - the class counts are plausible.
6. **Train and evaluate:**
   ```
   python -m research_experiments.public_ef.en_train
   ```
7. **Report to the owner:**
   - the internal EchoNext held-out accuracy and EF r;
   - the transfer AUROCs on the 837 hospital ECGs;
   - zero-shot accuracy;
   - stack-PLUS vs stack-BASE vs C3: accuracy with 95 % CI and per-class recall;
   - admissibility (every recall ≥ 0.15).
8. **Commit and push** only `research_experiments/public_ef/results/*.json` plus a short `PUBLIC_EF_REPORT.md`
   (aggregates only), after the privacy checks above. Push to `research/public-ef`.

## Optional (only if the owner asks)
- **Bigger run:** add `train` (`EchoNext_train_waveforms.npy`, 16 GB) with `--splits train,val,test`.
- **GPU fine-tuning** of ECGFounder on the EchoNext printout-format inputs. This needs a new pre-registered amendment
  before running.
