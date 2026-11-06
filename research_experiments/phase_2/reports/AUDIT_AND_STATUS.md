# Phase 2: audit findings and status (interim, committed 2026-11-04)

Phase 2 was **still running** when this commit was made. It contains:
- the protocol and the pre-registration;
- the audits;
- all completed development-set CV runs.

The robust repeat × seed validation, the image-CNN CV (folds 2–4) and the phase-2 report are not
included yet. The locked test set (210 ECGs) has **not** been scored in phase 2.

## Corrections to the phase-1 report (`research_experiments/reports/FINAL_REPORT.md`, left unchanged)

1. **Overstated CV significance.** Phase 1 reported header + ECGFounder vs header in 5-fold CV as
   +6.6 accuracy points, "p < 0.001", from a patient-cluster bootstrap on pooled out-of-fold
   predictions.
   - That bootstrap ignores training-set variability.
   - The corrected resampled t-test (Nadeau & Bengio 2003) on the 5 per-fold differences gives
     **p = 0.12** for accuracy and 0.33 for balanced accuracy.
   - The improvement is therefore **not statistically established** (`results/audit_previous.json`).
2. **Bootstrap p-value notation.** "p = 0.0" in the phase-1 tables means p < 1/2000, not zero.
3. **Population overlap.** The phase-1 CV used all 1,047 ECGs, including the 210 test ECGs, so it was
   not independent of the fixed-split test set. Phase 2 cross-validates the 837 development ECGs only.
4. **Unequal comparison conditions.** The phase-1 "fine-tuned ECGFounder" row is a 3-seed ensemble,
   compared with single-seed baselines.
5. **Exposed test set.** The test set was inspected for ~40 phase-1 models, so it is exposed. Phase 2
   uses it at most once, for a pre-registered shortlist, with authorisation.

## Dataset (unchanged; verified by SHA-256)

- 1,047 ECGs from 993 patients.
- Original split: train 669 / validation 168 / test 210 ECGs (635 / 159 / 199 patients).
- **669 is the training split only, never the total.**
- Phase 2:
  - development set = train + validation = 837 ECGs (794 patients);
  - locked test = 210 ECGs (199 patients);
  - **0 patients in both**.
- See `results/dataset_reconciliation.json`.

## Waveform-input audit (ECGFounder)

**Calibration and lead identity** (all 1,047 PDFs, `results/audit_waveform.json`):
- calibration exact: 1 mV pulse = 82 pt, 5 mm = 41 pt;
- native vector density about 280 points/s per lead, resampled to 500 Hz;
- lead polarity and R-wave progression physiological;
- heart rate vs the printed rate: median difference 0.35 bpm;
- QRS axis vs the printed axis: median difference 8°;
- derived limb leads exact up to rendering, r ≈ 0.99.

**Input composition:** the "aligned" input places each printed 2.5-s segment at its true time within
the single simultaneous 10-s acquisition:
- 37.5 of 120 lead-seconds measured;
- 12.5 derived exactly;
- 70 masked as zero, not fabricated.

**Alternatives** (development CV, S3; label-free agreement with printed statements in brackets):

| Input | S3 | Label-free agreement |
|---|---|---|
| aligned | **0.470** | 0.887 |
| measured-only | 0.468 | 0.885 |
| tiled (fabricates repeats) | 0.432 | 0.765 |
| 1-lead model on 10-s lead II | 0.429 | 0.765 |
| lead II only | 0.407 | 0.858 |

**Fixes:**
- 2 PDFs with lead coverage gaps of up to 0.5 s are now masked instead of extrapolated flat.

## Leakage fixes made during phase 2 (before any calibrated result was produced)

- The class prior for prior-correction is estimated from training-portion labels only.
- Decision thresholds are selected on genuinely out-of-fold predictions from an inner CV inside each
  outer training portion (`nested_calibration.py`).
