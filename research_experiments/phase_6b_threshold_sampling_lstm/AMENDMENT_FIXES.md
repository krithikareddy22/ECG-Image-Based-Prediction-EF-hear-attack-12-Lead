# Phase 6B amendment: FIXES (operationalising the root-cause fixes of `PROBLEM_ANALYSIS.md`)

**Date:** 2026-11-05. Written before any run, at the owner's request ("do those fixes").

## What the existing data allow
| Fix | What the data have | What is done here |
|---|---|---|
| 1. Patient-ID linkage | Neither the matching sheets nor the ECG PDFs carry a patient identifier. Sex agreement was already enforced by the original matching. Echo-side age is empty for matched records. | A re-linking worklist (below) for the hospital records team |
| 2. Biplane Simpson EF | The sheets hold Teichholz EF only (no Simpson, no LV volumes, no wall-motion data) | The same worklist, prioritised for a Simpson re-read |
| 3. Digital 10-s ECGs | Only PDFs exist (owner-confirmed) | Data request (`DATA_REQUEST.md`) |
| 4. More patients | None | Data request |
| 5. Clinical covariates | None | Data request |

## Worklist (`fixes.py --worklist`)
- **Output:** one row per development ECG, written to the git-ignored `data/private/fixes/recheck_worklist.csv` (it
  contains file paths and dates). Only aggregate counts are committed (`results/fixes_worklist_summary.json`).
- **Priority score:** label and link evidence only. No model prediction is used, so a later re-read cannot be steered
  by the model.

  | Evidence | Points |
  |---|---|
  | Folder label ≠ nearest-echo class | 3 |
  | EF within 3 points of a class limit | 3 |
  | EF within 3–6 points of a class limit | 1 |
  | Name-match score < 100 | 2 |
  | Nearest echo > 30 days from the ECG | 2 |
  | Nearest echo 8–30 days from the ECG | 1 |
  | Another-class echo within 30 days | 2 |
  | EF spread > 10 points across linked echoes | 1 |
  | ECG pattern in which Teichholz is unreliable (ECGFounder output for myocardial infarction, or for LBBB, with sigmoid ≥ 0.5) | 1 |

- **Re-read protocol** (written into the CSV header notes):
  - the reader is blinded to the folder label and to any model output;
  - the link is confirmed by MRN / UHID;
  - EF is re-measured by biplane Simpson;
  - every re-read value is used, whatever it shows.

## Ingest pipeline (`fixes.py --template`, `fixes.py --ingest`)
- **Template:** `--template` writes the fill-in file `data/private/fixes/verified_echo_template.csv` (git-ignored), with
  these columns:
  - `ecg_id`;
  - `mrn_verified` (yes / no);
  - `echo_date`;
  - `ef_simpson`, `ef_teichholz`;
  - `regional_wma` (yes / no);
  - `notes`.
- **New labels** (`--ingest`):
  - **EF:** `ef_simpson` if present, otherwise `ef_teichholz`, otherwise the current EF.
  - **Class:** the original sex-specific rule.
  - **Exclusion:** rows with `mrn_verified = no` are excluded from training and evaluation, and the count is reported.
- **Model:** the header-free retrained vote of `AMENDMENT_BANDS.md` (M2 and M5 logistic regressions plus EF-band
  probabilities from a nested EF ridge, nested class weights), on the Phase-2 folds.
- **Outputs:** accuracy, κ and per-class detection, with a paired comparison against the same pipeline on the current
  labels over the same ECGs.
- **Self-test:** running `--ingest` on the template filled with the current values must reproduce the current-label
  result. This is run now and gives the baseline that new data will be compared with.
