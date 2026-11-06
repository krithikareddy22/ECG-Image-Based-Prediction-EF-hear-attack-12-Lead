# Phase 6B exploratory amendment: BRES (borderline Mild resolved to the nearest class)

**Date:** 2026-11-05. Written before any BRES run, at the owner's request ("change the borderline Mild and try").
Kept in its own file so it doesn't edit PHASE6B_PLAN.md, which the laptop session is also amending.

- **Definition BRES:** consistently linked folder-Mild ECGs (nearest echo also Mild under the original rule) with
  nearest-echo EF within 2.0 points of the sex-specific upper Mild boundary (52 M / 54 F) -> **Normal** (16 ECGs);
  within 2.0 points of 41 % -> **Moderate** (29 ECGs). All other labels unchanged (Mild 185 -> 140).
- **Status:** EXPLORATORY. No clinical guideline defines this; it removes the measurement-uncertainty band around
  Mild. It was proposed after earlier results were known and is **not** eligible as a "clinically justified
  alternative definition" or as an improvement of the original task.
- **Evaluation:** existing best model recipe (MLP fusion, C1 vote, C3 rule; nested inner models re-fitted with BRES
  labels), scored against the ORIGINAL labels (primary) and against the BRES labels, with the definition-effect /
  training-effect decomposition. Folds, validation and evaluation populations unchanged; no ECG removed.
- Rejected variant (not run): moving all 28 % borderline Mild to Normal, because EF 41-43 % is not normal.
