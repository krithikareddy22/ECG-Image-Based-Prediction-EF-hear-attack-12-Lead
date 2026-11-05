# Phase 6B exploratory amendment: EF-verified labels (VER, VERB)

**Date:** 2026-11-05. Written before any run, at the owner's request ("classify them properly first and use better EF
labels").

- **High-confidence link:** the nearest echo by the true ECG acquisition date (file-name timestamp, which corrects the
  2021 day/month swap), exact name match (score 100), |ECG–echo| ≤ 30 days, and no other linked echo of a different
  class within 30 days. This is the best available evidence; no EF is verified by a patient identifier.
- **VER:** the folder label is replaced by the class of the high-confidence echo where the two disagree. The change is
  model-independent: model predictions are never used.
- **VERB:** VER, plus borderline Mild ECGs with a high-confidence link resolved to the nearest class (within 2 points of
  the sex-specific upper boundary → Normal; within 2 points of 41 % → Moderate). Borderline Mild with uncertain links
  keep their label.
- **Status:** EXPLORATORY label-quality sensitivity analyses, separate from the primary result (original labels).
  Models: the existing best recipe (MLP fusion, C1 vote, C3 rule; nested inner models re-fitted with the
  configuration's labels). Scored on the original, VER and VERB labels; the unchanged original (B1) models are scored on
  the same labels for the definition-effect / training-effect split. No ECG is removed; folds unchanged.
