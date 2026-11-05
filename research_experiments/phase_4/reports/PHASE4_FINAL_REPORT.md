# Phase 4 final report: can continuous LVEF supervision and better label quality unlock more predictive information?

**Short answer: no, not with these data.**
- Continuous EF from name-matched echoes is learnable from the ECG: r = 0.66, R² 0.43 on all development ECGs;
  r = 0.75, R² 0.55 on the high-confidence cohort.
- But neither EF regression, nor multitask EF supervision, nor cleaner training labels produce a 4-class
  classifier that is better than the Phase 3 vote ensemble.
- All differences fall within ±0.01 S3, with no Holm-significant gain on any metric. The 60 % accuracy target was
  not reached.

**Companion reports:**
- `PHASE4_DATA_AUDIT.md` (4A–4C);
- `PHASE4_LABEL_QUALITY_REPORT.md` (4B / 4F, G cohort);
- `PHASE4_REGRESSION_REPORT.md` (4D);
- `PHASE4_MULTITASK_REPORT.md` (4E);
- `PHASE4_MILD_CLASS_ANALYSIS.md` (4G);
- `PHASE4_MODEL_BENCHMARK.md` (4H);
- `PHASE4_CLAIMS_REGISTER.md`.

**Pre-registration:** `configs/PREREG_PHASE4.md` (committed `25e1d1f` before any Phase 4 run).

## Scope and safeguards
- **Data:** development set D, 837 ECGs / 794 patients.
- **Validation:** patient-grouped 5-fold CV × 3 repeats, identical folds to Phases 2–3; every choice is made on
  inner-val.
- **Locked test:** the 210-ECG locked test was **not predicted and not scored**. It was exposed in Phase 1, so an
  independent external or temporal cohort is required for any performance claim.
- **Integrity:** 4,648 protected original files were verified unchanged (SHA-256) before and after Phase 4.
- **Privacy:** per-ECG outputs stay in the git-ignored `phase_4/data/private/`. Committed files are aggregates,
  scanned for every ECG id, acquisition timestamp and patient hash.
- **Run volume:** 162 trained runs and 360 evaluated runs, summarised as 68 models.

## Answers to the 13 research-decision questions

**1. How many reliable continuous EF measurements were found?**
- **None can be verified.** ECG and echo are linked by fuzzy **patient-name** matching only: no patient ID and no
  date window in the original matching.
- **By grade:**
  - every one of the 1,047 ECGs has at least one linked Teichholz EF;
  - **448 ECGs** (357 development / 345 patients + 91 test) meet the high-confidence criteria: exact name,
    ≤ 30 days, sex agrees, no conflicting link;
  - 688 have the nearest echo within 14 days.
- **Measurement uncertainty** cannot be estimated: only 18 repeat-echo pairs qualify.

**2. How many ECGs could be matched within 7, 14 and 30 days?**
- Nearest-in-time link, label-agnostic: **672 / 688 / 735** of 1,047. In development: 541 / 550 / 587; test:
  131 / 138 / 148.
- 543 ECGs have a same-day echo (label-consistent link).

**3. How many existing categorical labels disagree with continuous-EF-derived labels?**
- **68 of 1,047 (6.5 %)** disagree with the class of the nearest-in-time echo: agreement 93.5 %, 93.4 % in
  development.
- A further **91** ECGs have links implying different classes, and **22** have no link that reproduces their
  label.
- Agreement is 98.0 % for single-link ECGs vs 67.7 % for multi-link ECGs, so disagreement reflects **link
  ambiguity**.
- Borderline EF alone is not treated as evidence of a wrong label.

**4. Is label uncertainty a plausible performance limitation?**
- **Only a minor one; it is not the main limitation.**
  - Labels agree with the best EF evidence 93 % of the time, against 55 % model accuracy.
  - Training only on the high-confidence cohort, dropping the disagreeing ECGs, or relabelling from the EF changes
    S3 by −0.007 to +0.008 vs control, all n.s.
  - Every model is about 7 points more accurate on G than non-G. Class mix explains 1–2 points and label agreement
    about 3; this is an **evaluation-set effect**, not something a model can exploit.
- **The dominant limitation is information:** the ECG explains R² ≈ 0.43 of EF variance. By the error-budget
  analysis, the targets need r ≈ 0.80 between predicted and echo EF; observed r is 0.66.

**5. Did EF regression improve classification?**
- **No.**
  - All 12 primary-decoder regression models have lower S3 than the vote ensemble (Δ −0.022 to −0.074) and than
    the direct classifier on the same features (Δ −0.016 to −0.068).
  - With the clinical thresholds, regression keeps accuracy (0.54–0.56) but loses Severe detection: recall
    0.10–0.12, because predictions shrink toward the mean.
  - As an extra ensemble member, regression raises accuracy to 0.557–0.560 (n.s., p ≥ 0.42) and macro AUROC by
    +0.013 (Holm p 0.07), but lowers balanced accuracy.

**6. Did multitask learning improve classification?**
- **No demonstrable improvement.**
  - All 8 EF-supervised variants are ≥ the classification-only control (ΔS3 +0.001 to +0.007; Mild F1 +0.004 to
    +0.029), but none is significant (Holm p = 1.00). The control reproduces the Phase 2 MLP fusion exactly.
  - Loss weight, learned weighting, inner-val λ selection and cleaner EF targets make no difference.
  - The joint model predicts EF as well as dedicated regressors (MAE 10.3–10.5, R² 0.53 on G).

**7. Did Mild F1 and recall improve?**
- **Not usefully.**
  - EF regression with clinical thresholds gives the highest Mild F1 (0.309) and recall (0.34), vs 0.255 / 0.22
    for the vote ensemble, but at the cost of Severe recall (0.10). This is not acceptable clinically.
  - Among models that keep overall performance, Mild F1 is 0.19–0.25, below the vote ensemble.
  - EF regression ranks Mild better (one-vs-rest AUROC 0.61 vs 0.55, p = 0.043, uncorrected): a **hypothesis**
    for future work.
  - The Mild F1 target (0.35) was not reached.

**8. Did we reach 60 % accuracy?**
- **No.**
  - The best development-CV accuracy in Phase 4 is **0.560** (vote + MLP-regression member). The Phase 3
    calibrated MLP fusion remains at 0.568.
  - For the candidates with the best point estimates, the corrected 95 % CI of the accuracy gain over the vote
    ensemble has an upper bound of +0.03 to +0.05, i.e. accuracy ≤ 0.58–0.59. Only high-variance models whose point
    estimates are at or below the vote ensemble have upper bounds reaching 0.60.

**9. Which model offers the best accuracy?**
- The **Phase 3 MLP fusion with nested protected prior calibration**: 0.568, with Mild F1 0.159 (reference;
  predictions not stored).
- Within Phase 4: **vote ensemble + MLP-regression member**, 0.560 (balanced 0.458, macro-F1 0.446), not
  significantly different from the vote ensemble.

**10. Which model offers the best balanced performance?**
- **Still the Phase 3 vote ensemble:** accuracy 0.547, balanced accuracy 0.467, macro-F1 **0.455**, Mild F1 0.255,
  S3 0.490.
- `lq_dropdisagree` (S3 0.491, balanced 0.471) and `mt_lam1_G` (S3 0.490) tie it.

**11. Which results are statistically convincing?**
- **(a) The ECG carries substantial continuous EF information:** MAE 10.4 vs 15.6 for the mean predictor; R² 0.43
  (all) and 0.55 (G), consistent across 3 repeats and 3 seeds. The ECGFounder waveform embedding carries most of it
  (header alone R² 0.26).
- **(b) The G evaluation-set effect** (+5–8 points for every model) and its partial explanation by label agreement.
- **(c) Clinical-threshold conversion of regressed EF suppresses Severe detection:** recall 0.43 → 0.10–0.12;
  0.06–0.14 in every individual run, against 0.42–0.44 for the vote ensemble.
- **(d) Errors concentrate near class boundaries:** accuracy 0.37 within 5 EF points vs 0.74 at ≥ 10.
- **(e) The negative result:** no Phase 4 model improves 4-class S3 by more than about 0.03 (corrected 95 % CI
  upper bounds) over the vote ensemble.
- **Not convincing:** any 4-class improvement; the Mild AUROC gain; the +0.013 macro-AUROC gain of the regression
  member (Holm p 0.07).

**12. What is the strongest defensible research contribution?**
- **A rigorous, pre-registered negative result with a quantified explanation.**
  - On a 1,047-ECG printout dataset whose labels were built by fuzzy name matching, continuous EF supervision and
    label cleaning do not improve 4-class LVEF classification.
  - The reason is quantified: an ECG-derived EF estimate with r ≈ 0.66 (R² 0.43) is far below the r ≈ 0.80 that
    60 % 4-class accuracy would require.
- **Positive deliverables:**
  - a label-provenance audit showing that the 93.5 % agreement reflects matching consistency, not independent
    validation;
  - an ECG-to-EF regressor (r 0.75 on high-confidence links);
  - clinically standard binary performance: AUROC 0.82 for EF < 41 and 0.79–0.80 for EF < 31 against folder labels
    (0.86–0.87 against EF-derived labels on G).
- **Recommendation:** binary reduced-EF detection, or EF regression, is the more defensible clinical task than a
  narrow-band 4-class split with a 11–13-point Mild band.

**13. What additional data would provide the greatest benefit?**
1. **A verified ECG–echo link** (medical record number or encounter ID) with dates. This turns 1,047 fuzzy links
   into verified EF targets and makes label noise measurable.
2. **More abnormal-EF ECGs.**
   - The matching sheets contain 4,177 matched ECGs, but the 3,130 not in the dataset are almost all Normal
     (3,106; only 22 Mild, 1 Moderate, 1 Severe; `results/matched_pool_summary.json`).
   - Every abnormal-EF ECG in the sheets is already used, so gains need **new years or sites**. The error budget
     suggests several thousand ECGs with abnormal EF.
3. **Digital waveforms** (XML / DICOM-ECG) instead of PDF printouts: full 10-s, 12-lead, higher fidelity.
4. **Repeat or biplane (Simpson) echo measurements**, to quantify EF measurement noise. Only Teichholz is available
   and its uncertainty is unknown.
5. **An external or temporally independent validation cohort** (e.g. 2025 ECGs), since the locked test was
   exposed in Phase 1.
6. The 3,106 additional Normal ECGs could support self-supervised or regression pre-training if their PDFs still
   exist. They would not address the class imbalance.

## Recommended model and next steps
- **Model to keep:** the Phase 3 vote ensemble remains the recommended 4-class model; it is unchanged by Phase 4.
  For ranking or AUROC use, adding the EF-regression member gives a small, non-significant gain.
- **Before any claim:**
  - pre-register a shortlist: vote ensemble; vote + MLP-regression member; calibrated MLP fusion;
  - score it once on a **new independent cohort**. The exposed 210-ECG test set may be scored once, only with the
    owner's approval, and reported as exposed.
- **Further work on the same data:** further architecture work on these 837 ECGs is unlikely to pay off. The
  evidence points to data (verified links, more abnormal-EF ECGs, digital waveforms), not models.
