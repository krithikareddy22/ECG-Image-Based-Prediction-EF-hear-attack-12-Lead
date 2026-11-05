# Phase 6B exploratory amendment: HC30 cohort study and Mild-factor study (owner request)

**Date:** 2026-11-05. Written before any run.

1. **HC30 cohort:** development ECGs whose nearest echo (true acquisition date) has an exact name match (score 100), is
   ≤ 30 days from the ECG, and has no linked echo of another class within 30 days (395 ECGs). Defined by link quality
   only, never by EF difficulty or model output.
   - **HC30 (existing recipe):** MLP fusion (3 seeds), ECGFounder linear distance-CE, header + clinical LR, C1 vote, C3
     nested prior rule. All fitting (outer, nested inner, rule / prior) uses HC30 ECGs only; Phase 2 folds; evaluation
     on the HC30 ECGs of each outer fold (and, descriptively, on all 837).
2. **Mild-factor study (descriptive):** univariate AUROC of every ECG-derived input for Mild vs Normal and Mild vs
   Moderate on the development set: header numbers (16), clinical features (28), Phase 5 families (qrs, st, hrv, freq,
   wavelet, leadstat, sq) and ECGFounder's 150 pretrained diagnostic-statement logits. Aggregates only; never used to
   choose model inputs.
3. **HC30+F (pre-declared model):** the HC30 recipe plus a fourth vote member, a multinomial logistic regression
   (class-balanced; C on inner-val S3) on ALL the above factors at once, so that any selection happens inside the
   training folds through regularisation. Variants: vote of 4 members, and + nested prior rule.
- **Inputs:** all are ECG-derived. Echo EF defines only the cohort and labels. Exploratory; reported separately from
  the primary all-837 result.
