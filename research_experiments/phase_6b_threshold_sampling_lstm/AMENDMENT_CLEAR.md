# Phase 6B exploratory amendment: CLEAR (train on clear-cut cases only)

**Date:** 2026-11-05. Written before any CLEAR run, at the owner's request ("do the clear-cut cases one for 66 % and
try improving more").

- **Border ECG:** nearest-echo EF within 3.0 points of any class boundary (sex-specific upper Mild boundary 52 M / 54 F,
  41 %, 31 %), or folder label ≠ class of the nearest echo. **255 of 837** development ECGs; the remaining 582 are
  "clear-cut". Echo EF defines the training subset only; it is never a model input.
- **CLEAR:** border ECGs removed from every fit (outer members, nested inner models, prior / rule fitting); original
  labels; inner-val, folds and evaluation unchanged.
- **Evaluation:** (1) all 837 ECGs, original labels (primary, comparable with every other result); (2) the clear-cut
  subgroup (descriptive only; not deployable, because the true EF is unknown at prediction time); (3) the border
  subgroup. Compared with the unchanged original (ORIG) models on the same subgroups.
- **Status:** EXPLORATORY; reported separately; not eligible as an improvement of the original task unless it meets the
  usual criterion on all 837 ECGs.
