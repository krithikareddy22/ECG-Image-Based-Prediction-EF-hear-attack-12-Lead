# Phase 6B report: FIXES (what the existing data allow; `AMENDMENT_FIXES.md`)

- **Data audit:** neither the matching sheets nor the ECG PDFs carry a patient identifier (checked: sheet columns; ID
  labels in 60 sampled PDFs). Sex agreement was already enforced by the original matching. Echo-side age is empty for
  matched rows. EF is Teichholz only. Patient-ID re-linking and Simpson re-reads therefore need the hospital.
- **Re-check worklist** (private, git-ignored; counts in `results/fixes_worklist_summary.json`):
  - priority high 197 (Normal / Mild / Moderate / Severe 25 / 70 / 78 / 24), medium 266, low 220, none 154;
  - 184 of 185 Mild and all 177 Moderate ECGs carry at least one flag.
- **Teichholz-unreliability flag:** ECGs with an ECGFounder infarct or LBBB pattern do **not** disagree more with the
  echo. The SD of predicted minus echo EF is 11.9 (MI pattern, n 163) and 13.0 (LBBB, n 70), vs 13.6 for the rest
  (n 615). The flag keeps weight 1.
- **More patients from the sheets:** 4,175 unique matched ECGs. The 3,128 not in the project are 99 % Normal (Normal
  3,104, Mild 22, Moderate 1, Severe 1), and their PDFs are not available. The dataset already contains essentially
  every abnormal-EF ECG.
- **Averaging repeated echoes:** 35 ECGs have ≥ 2 echoes within 30 days, and averaging would change 9 labels. That is
  too few to matter and was not run.
- **Ingest self-test** (`results/fixes_selftest.json`): the pipeline reproduces the current-label result exactly
  (0.594 [0.560, 0.627], κ 0.353, recall 0.91 / 0.20 / 0.32 / 0.38; Δ 0.000). It is ready for verified data.
- **Data request:** `DATA_REQUEST.md`.
