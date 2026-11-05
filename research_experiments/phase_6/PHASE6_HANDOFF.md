# Phase 6 handoff (living document; updated at every milestone)

- **Branch:** `research/phase6`, created from `research/phase5` @ `27cbdbd`, pushed to `origin` (the `chaithr45`
  fork). No force-push; no other branch modified; never push to `claude/funny-gauss-udlmv0`.
- **Read first:** `PHASE6_RESUME.md` (commands), `PHASE6_PLAN.md` (pre-registration), `PHASE6_STATUS.md` (live
  status), `PHASE6_CLAIMS_REGISTER.md`, `REPOSITORY_AUDIT_P6.md`.

## State
**Handover (2026-11-05):** the remaining Phase 6A / 6B computation moves to the owner's laptop (NVIDIA GPU) in a local Claude Code session. Per-ECG predictions are private and machine-local, so that session regenerates every input and baseline deterministically (PHASE6_RESUME.md; baselines must be 24/24 byte-identical) and then runs everything still open. The cloud session stopped its LSTM runs (L4 29/45 folds, L1 1/45; nothing committed from them) to avoid computing anything twice.

**GPU laptop session (2026-11-05): Phase 6A and Phase 6B complete.**
- *Environment:* clone at `C:\krithika_major\ecg-lvef` (outside OneDrive), venv `C:\krithika_major\venv` (Python 3.13,
  torch 2.15.0+cu132, RTX 3060 6 GB). Windows helpers are outside the repository (see `PHASE6_RESUME.md`).
- *Restore:*
  - data cache, manifest and header features match their recorded SHA-256; `restore_inputs` identical;
  - baselines 24/24 byte-identical; ECGFounder cache f6 max diff 1.5e-05;
  - A1 fine-tuned regenerated (max diff 0.0000); 84 Phase 6A existing-model runs regenerated (max 0.0047);
    `existing --check` True;
  - integrity snapshot `phase6_before` re-taken locally.
- *New code:* MISSING_PRIVATE regeneration under a 0.005 tolerance (`p6common` / `p6acommon` `save_run`, records in
  `results/regeneration/`). Phase 6B adds `lstm6b.py` (L5, L6, F1-F4, LSTM-fixed under the 6B configurations),
  `audit6b.py`, `compare6b.py` and `diagram6b.py`; Phase 6A adds `inventory.py`.
- *Results:* see `phase_6a_normal_mild_lstm/PHASE6A_FINAL_REPORT.md` and
  `phase_6b_threshold_sampling_lstm/PHASE6B_FINAL_REPORT.md`. The best model on the original labels is still C3
  (0.561 [0.520, 0.603]). No LSTM, CNN-LSTM, fusion, threshold or Normal-size configuration gives a genuine improvement.
  60 % is not reached.
- *Concurrent exploratory work (another session, same branch):* BRES / VER / VERB label-quality configurations
  (`phase_6b_threshold_sampling_lstm/AMENDMENT_BRES.md`, `AMENDMENT_VERIFIED.md`, `bres.py`, `verified.py`). They are
  EXPLORATORY and are **not** part of the pre-registered comparison in `PHASE6B_FINAL_REPORT.md`. Their per-ECG
  predictions are not on the laptop.
- *MMCLEAN (laptop, exploratory, owner / supervisor request "70 %"):* `phase_6b_threshold_sampling_lstm/MMCLEAN_REPORT.md`.
  - Training without suspect labels, with fewer Normal ECGs or with echo relabelling does not raise accuracy on all 837
    ECGs (best C3 unchanged, 0.562).
  - On cleaner subsets the maximum is 0.677.
  - Only selective prediction reaches 0.71: 22 % of ECGs answered, almost all Normal / Severe.
  - Model-dependent removal is excluded as circular.
- *NF / NFS and MILDFIX (laptop, exploratory):* `phase_6b_threshold_sampling_lstm/MILDFIX_REPORT.md`.
  - Removing echo-link noise and seven nested Mild fixes do not beat C3 (0.562) on all 837 ECGs.
  - About 70 % appears only with abstention (0.708, answering 34 %, Normal / Severe only) or on the 264 clean,
    clear-cut ECGs (0.71-0.72).
- *MILDFIX2 (laptop, exploratory):* `phase_6b_threshold_sampling_lstm/MILDFIX2_REPORT.md`.
  - Richer and noise-aware EF regressors leave the EF error at about 13 points SD (best 13.22) and four-class at 0.56.
  - Agreement abstention answers 57 % at 0.698; Mild is never answered correctly.
- *MILDFEAT (laptop, exploratory, owner request "classify Mild from the image only"):*
  `phase_6b_threshold_sampling_lstm/MILDFEAT_REPORT.md`.
  - 121 clinician ECG features come from a synthesised simultaneous 12-lead beat (columns aligned to the rhythm strip).
  - They do not improve on ECGFounder (best image-only 0.558).
  - Mild separates from Normal (AUROC up to 0.757) but barely from Moderate (up to 0.63).
- *Laptop session logs* (aggregate metrics only) are in the git-ignored
  `phase_6b_threshold_sampling_lstm/data/private/logs/laptop_session/`.

| Milestone | State |
|---|---|
| M0 setup + pre-registration | DONE |
| M1 timing (experiment A) | DONE (plan Amendment 1) |
| A stage 1 | DONE: R0 reproduces Phase 4; A1 and A2 fail the screen (r 0.650 vs 0.652) |
| B stage 1 | DONE for ECG-FM (fails; also combined with ECGFounder); HuBERT-ECG / ST-MEM need an owner licence decision; ECG-JEPA / MERL blocked (Google Drive) |
| D, Stage 2, E, N, final report | not started |
| Phase 6A (Normal/Mild boundary + LSTM) | DONE (GPU laptop): `PHASE6A_FINAL_REPORT.md` |
| Phase 6B (thresholds, Normal size, LSTM family, fusions) | DONE (GPU laptop): `PHASE6B_FINAL_REPORT.md` |
| C (extra ECGs) | not run: blocked (no extra PDFs / waveforms; owner-confirmed) |
| Phase 7 (no recurrent models; ECGFounder representations, EF supervision, 1D CNN, data needs) | DONE (GPU laptop): `phase_7/PHASE7_FINAL_REPORT.md` - no candidate passed its gate; C3 0.562 remains best; 70 % not reached |

## Known issues (owner decisions; recorded, not acted on)
1. **Public exposure:** the public repository tracks the patient-named PDFs, its history contains the deleted
   `ECG_ECHO` sheets, and Phase 1–3 per-ECG CSVs (incl. `phase_2/locked_test/`) are tracked. The owner handles this;
   Phase 6 modifies none of it and adds nothing similar.
2. **Name pairs:** 29 near-identical name pairs under different keys: 15 within D (sensitivity analysis in Phase 6),
   **11 straddling D and the locked test**, 3 within the locked test. The 11 + 3 must be resolved before any
   locked-test evaluation.
3. **Locked test:** untouched in Phase 6; no test PDF is run through any model.

## Final state (2026-11-05)
All Phase 6 / 6A / 6B work is complete and pushed. Final reports: `reports/PHASE6_FINAL_REPORT.md`, `../phase_6a_normal_mild_lstm/PHASE6A_FINAL_REPORT.md`, `../phase_6b_threshold_sampling_lstm/PHASE6B_FINAL_REPORT.md` (+ `ADDENDUM_LABEL_EXPERIMENTS.md`). Best model unchanged: C3, 0.561 [0.520, 0.603]; no genuine improvement. Open owner decisions: 11 development/test name pairs, HuBERT-ECG / ST-MEM licences, optional blinded manual label review, any locked-test evaluation.
