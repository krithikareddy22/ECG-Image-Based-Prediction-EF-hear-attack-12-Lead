# Phase 4 claims register (extends `phase_3/reports/CLAIMS_REGISTER.md`)

**Status labels:**
- **VALID**: directly measured;
- **CORRECTED**: a previously stated value revised;
- **RETRACTED**: unsupported or incorrect;
- **HYPOTHESIS**: plausible but unverified;
- **ESTIMATE**: projected, not measured.

**Previous claims:** P4-1 to P4-8 were registered at the handoff (commit `039a77b`). Their wording is kept; any
status change is recorded in the "Update" column, never silently.

**Measurement basis:** all model metrics are development CV (837 ECGs / 794 patients; folds as in Phases 2–3).
The locked test was not scored.

| # | Claim | Status | Evidence | Update (Phase 4D–4H) |
|---|---|---|---|---|
| P4-1 | The four `LV_ECG_Dataset_*/*.xlsx` sheets contain no EF values, no echo dates and no patient IDs; they are file-copy logs | VALID | `PHASE4_DATA_AUDIT.md` §1 | – |
| P4-2 | Folder labels were produced by fuzzy NAME matching (no patient ID, no date window) with Teichholz EF and sex-specific thresholds 52 / 54 / 41 / 31 | VALID | DATA_AUDIT §4 (recovered notebook code) | – |
| P4-3 | 93.5 % of folder labels agree with the class of the nearest-in-time linked echo; 22 ECGs have no linked echo consistent with their label | VALID | DATA_AUDIT §5 | Agreement in D = 93.4 %; 98.0 % for single-link vs 67.7 % for multi-link ECGs (LABEL_QUALITY §2.3) |
| P4-4 | 91 ECGs are linked to echoes of different EF classes; the median EF spread across multiple links is 17 points | VALID | DATA_AUDIT §5 | – |
| P4-5 | Within 7 / 14 / 30 days: 541 / 550 / 587 development ECGs (672 / 688 / 735 incl. test) | VALID | DATA_AUDIT §6 | – |
| P4-6 | Large EF spreads across links reflect wrong-person matches | HYPOTHESIS | not verifiable without patient IDs | unchanged |
| P4-7 | Label noise limits 4-class accuracy | HYPOTHESIS → **partly supported, not as the main limitation** | label-quality training no gain (ΔS3 −0.007 to +0.008 vs control, n.s.); G effect ≈ +7 points for every model, of which about 3 points relate to label agreement | Re-stated as P4-12 and P4-13 |
| P4-8 | EF measurement uncertainty can be quantified from these data | **NOT SUPPORTED** | only 18 qualifying repeat pairs | unchanged; no uncertainty claim made in Phase 4 |
| P4-9 | ECG features predict nearest-echo EF with MAE 10.4, RMSE 13.3, R² 0.43, r 0.66 on all 837 ECGs (mean predictor: MAE 15.6); R² 0.54–0.55, r 0.74–0.75 on G | VALID | REGRESSION §2 | – |
| P4-10 | The ECGFounder embedding carries most of the EF information (header-only R² 0.26; ECGF 0.42; adding the header to ridge does not help) | VALID | REGRESSION §2 | – |
| P4-11 | Training on cleaner EF cohorts (C ≤ 14 d, G) does not improve regression or classification at this sample size | VALID | REGRESSION §2; MULTITASK §2 | – |
| P4-12 | Every model is 5–8 points more accurate on G than on non-G. Class mix explains 1–2 points, label agreement about 3. This is an evaluation-set effect, not a model improvement | VALID | LABEL_QUALITY §2.4 | – |
| P4-13 | The 95.8 % folder-EF agreement in G is largely internal consistency of the matching, not independent label validation | VALID (logical, with the 98.0 % vs 67.7 % single- vs multi-link evidence) | LABEL_QUALITY §2.3 | – |
| P4-14 | EF regression → 4-class (any decoder) does not beat direct classification; clinical thresholds suppress Severe recall (0.10–0.12 vs 0.43) through shrinkage | VALID | REGRESSION §3; BENCHMARK §3 | – |
| P4-15 | Continuous EF supervision improves Mild ranking (one-vs-rest AUROC 0.61 vs 0.55; p = 0.043 uncorrected) | **HYPOTHESIS** | MILD §6 | not multiplicity-robust; to be tested on independent data |
| P4-16 | Multitask EF supervision improves 4-class classification | **NOT SUPPORTED** | MULTITASK §2: ΔS3 +0.001 to +0.007 vs control, Holm p = 1.00 | – |
| P4-17 | No Phase 4 model genuinely improves on the Phase 3 vote ensemble (0 of 26 confirmatory models on S3 / accuracy / balanced accuracy / macro-F1; smallest Holm p 0.25) | VALID | BENCHMARK §2 | – |
| P4-18 | Adding an EF-regression member to the vote ensemble raises macro AUROC by about 0.013 | HYPOTHESIS | BENCHMARK §3 (p 0.007, Holm 0.07) | borderline |
| P4-19 | 60 % 4-class accuracy is reached | **NOT SUPPORTED** | best Phase 4 0.560; Phase 3 calibrated MLP 0.568 | – |
| P4-20 | Reaching the owner's targets would need an EF estimate with r ≈ 0.78–0.82 against the echo EF (observed 0.66) | ESTIMATE | REGRESSION §4 (classical-error simulation) | see C-1 |
| P4-21 | Essentially all abnormal-EF ECGs in the matching sheets are already in the dataset; the 3,130 extra matched ECGs are 99 % Normal | VALID (counts; links unverified) | `results/matched_pool_summary.json` | – |
| P4-22 | Errors concentrate within 5 EF points of a class boundary (accuracy 0.37 vs 0.74 at ≥ 10 points) | VALID | MILD §5 | – |
| P4-23 | The multitask λ = 0 control reproduces the Phase 2 MLP fusion (max \|ΔP\| 5 × 10⁻⁶) | VALID | MULTITASK §1 | – |
| C-1 | Interim statement in the working session: "EF regression sits just below the error level the 60 % target would need" (it compared RMSE 13 with simulation σ 12) | **CORRECTED** | RMSE of the shrunk regression is not comparable with classical-error σ; the correct comparison by correlation gives r 0.66 vs about 0.81 needed | corrected before any report was committed; see P4-20 |
| — | All Phase 1–3 claims keep their status | – | `phase_3/reports/CLAIMS_REGISTER.md`; the Phase 1 CV p < 0.001 stays RETRACTED | – |
