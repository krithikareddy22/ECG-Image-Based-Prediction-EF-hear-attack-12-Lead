# Phase 5 final report: four-class ECG-based LVEF classification

**Answer:** the 65 % target was **not achieved**.
- **Best admissible model:** the vote ensemble with a nested class-prior decision rule, accuracy **0.561** (corrected
  95 % CI 0.520–0.603).
- **Best of any eligible model:** 0.572, but it makes Mild nearly undetectable.
- **Genuine improvements:** none of the 43 Phase 5 candidates is a genuine (Holm-significant, consistent) improvement
  over the Phase 3 vote ensemble (0.547) or the calibrated MLP (0.568).
- **65 % is outside the confidence interval of every model:** the upper bounds are ≤ 0.612.

**Companion documents:**
- `REPOSITORY_AUDIT.md`;
- `PHASE5_PLAN.md` (pre-registration, SHA-256 in `configs/PLAN.sha256`);
- `reports/PHASE5_BASELINE_REPORT.md`, `reports/PHASE5_ERROR_ANALYSIS.md`, `reports/PHASE5_FEATURE_REPORT.md`;
- `PHASE5_CLAIMS_REGISTER.md`.

**All numbers:**
- **Population:** development-set patient-grouped cross-validation (837 ECGs / 794 name-based patient keys, 5 folds ×
  3 repeats, identical folds for every model).
- **Locked test:** the 210-ECG locked test was never predicted, scored or inspected.

## 1. Research objective
Improve exact four-class accuracy (Normal / Mild / Moderate / Severe) from the 12-lead ECG PDF printout, without making
any class undetectable, under patient-grouped validation with no leakage.

## 2. Dataset
- **Total:** 1,047 ECG PDFs, labelled from echocardiography by fuzzy name matching (Phase 4 audit).
- **Development set** (all Phase 5 work): 837 ECGs; Normal 384, Mild 185, Moderate 177, Severe 91.
- **Inputs:** the digitised waveform (12 leads × 2.5 s + 10-s lead II), printed machine measurements, age and sex.
  Echo EF is never an input.

## 3. Exact four-class definitions (hospital rule, unchanged)
| Class | Male EF | Female EF |
|---|---|---|
| 0 Normal | > 52 % | ≥ 54 % |
| 1 Mild | 41–52 % | 41–53 % |
| 2 Moderate | 31–40 % | 31–40 % |
| 3 Severe | < 31 % | < 31 % |

## 4. Baseline (5A)
| Model | Accuracy | Balanced acc. | Macro-F1 | Min class recall | Admissible |
|---|---|---|---|---|---|
| **C1** vote ensemble (Phase 3) | 0.547 | 0.467 | 0.455 | 0.220 | yes |
| **C2** calibrated MLP, 56.8 %; **reproduced exactly** with stored predictions | 0.568 | 0.463 | 0.442 | 0.112 | no |
| Best admissible baseline: vote + EF-regression member (Phase 4) | 0.560 | 0.458 | 0.446 | 0.182 | yes |

- **Comparability:** the 56.8 % benchmark was verified to be four-class, on the same labels and population (not a
  test-set result).
- **Admissibility (pre-registered):** every class recall ≥ 0.15.

## 5. Methodology
- **Validation:**
  - patient-grouped StratifiedGroupKFold(5) outer folds × 3 repeats, from Phase 2;
  - inner-train / inner-val inside every outer training portion for fitting, early stopping and hyper-parameters;
  - a patient-grouped **nested 4-fold inner CV** for anything fitted on predictions (stacking, decision rules,
    calibration, quality gating). Meta-models only ever see training-portion out-of-fold predictions.
- **Leakage controls:**
  - development-only data loader (no module that loads the locked test is imported; tested);
  - patient-disjoint folds (asserted);
  - all preprocessing fitted in-fold;
  - echo EF never used as input;
  - per-ECG predictions in git-ignored storage, hash-checked.
- **Implementation checks:**
  - the Phase 5 MLP trainer reproduces the Phase 2 MLP fusion (max |ΔP| 6 × 10⁻⁷);
  - the waveform-augmentation pipeline reproduces the committed ECGFounder embedding (1.4 × 10⁻⁶);
  - the calibrated-MLP reproduction is identical.
- **Statistics:**
  - corrected repeated-CV t-test on 15 matched (repeat, fold) cells;
  - Holm within each planned family and globally;
  - same-sign-in-all-repeats requirement;
  - patient-cluster paired bootstrap (2,000 resamples) and exact McNemar on repeat 0.

## 6. Feature comparisons (5C)
- **Engineered families:** QRS morphology, ST/T, HRV, spectral, wavelet, lead statistics and signal quality, validated
  against the printed measurements (heart rate r 0.85; QRS width r 0.75).
- **LR results:**
  - only **QRS morphology** adds to the header numbers (0.489 → 0.538; p 0.043, Holm 0.38);
  - all families together dilute the signal (0.503).
- **E1:** no family improves the MLP fusion (|Δacc| ≤ 0.010, Holm 1.00). The ECGFounder embedding already carries
  the QRS information.

## 7. Model comparisons (5D)
| Experiment | Best variant | Accuracy | Min recall | Δacc vs its baseline (p) | Verdict |
|---|---|---|---|---|---|
| E1 feature-enhanced MLP | + all new families | 0.554 | 0.164 | +0.010 vs MLP (0.59) | no gain |
| E2 gradient boosting | LightGBM, engineered + ECGF PCA | 0.544 | 0.184 | +0.000 vs MLP (1.00) | no gain |
| E3 nested stacking | weighted average (accuracy) | 0.547 | 0.184 | +0.000 vs C1 (1.00) | no gain |
| | (unweighted LR) | 0.567 | 0.041 | +0.020 vs C1 (0.31) | not admissible |
| E4 hierarchical (N vs abnormal → Mi / Mo / S) | – | 0.567 | 0.067 | +0.023 vs MLP (0.08) | not admissible |
| E5 nested decision rules on the vote | **prior, protected** | **0.561** | 0.177 | +0.014 vs C1 (0.16) | admissible, n.s. |
| | bias, recall floor | 0.561 | **0.204** | +0.014 vs C1 (0.41) | admissible, n.s. |
| | bias, accuracy | 0.566 | 0.025 | +0.019 vs C1 (0.18) | not admissible |
| E5 on the MLP fusion | bias, accuracy | 0.571 | 0.016 | +0.028 vs MLP (0.04; Holm 0.12) | not admissible |
| E6 signal quality | quality-gated blend (nested) | 0.546 | 0.164 | −0.001 vs C1 (0.93) | no gain |
| | quality-weighted / quality-filtered training | 0.545 / 0.537 | 0.137 / 0.151 | +0.001 / −0.006 vs MLP | no gain |
| E7 training strategies | mixup | 0.552 | 0.126 | +0.009 vs MLP (0.27) | n.s. |
| | focal / smoothing / cosine LR / accuracy-based stopping | 0.544–0.545 | 0.115–0.205 | ≤ +0.002 | no gain |
| | waveform augmentation (2 copies) | 0.533 | 0.162 | −0.011 (0.18) | no gain |
| | **class-unweighted CE** (3 repeats) | **0.572** | 0.144 | +0.025 vs C1 | not admissible (Mild) |

Full table: `results/phase5_all_models.csv`; family tests with Holm: `results/phase5_family_comparisons.csv`.

## 8. Ablations
Each E-experiment changes one factor relative to its baseline.

| Group | Changes tested | Effect |
|---|---|---|
| E7 | class weighting, focal loss, label smoothing, mixup, LR schedule, selection metric, augmentation | −0.014 to +0.009 accuracy |
| Phase 2 strategies re-used (repeat 0 only) | dropout 0.3 / 0.7, hidden 32 / 128, balanced sampler | −0.014 to +0.009 |
| E1 | feature families added one at a time | −0.009 to +0.003 |

**The only levers that move accuracy above about 0.56 trade away Mild (or Severe) recall:**
- class weighting removed;
- accuracy-objective decision rules;
- the hierarchical split.

## 9. Confusion matrices
Mean counts over 3 repeats; rows = folder label, columns = prediction (N / Mi / Mo / S). Figure:
`results/plots/phase5_selected_confusion.png`.

| Selected (vote + prior rule) | N | Mi | Mo | S |
|---|---|---|---|---|
| Normal (384) | 324.3 | 35.3 | 15.0 | 9.3 |
| Mild (185) | 79.3 | 32.7 | 52.3 | 20.7 |
| Moderate (177) | 38.0 | 27.0 | 81.3 | 30.7 |
| Severe (91) | 15.0 | 14.3 | 30.3 | 31.3 |

C1 and C2 matrices: `results/phase5_all_models.csv` (`confusion_mean`). Relative to C1, the selected rule moves about
23 Normal ECGs from Mild / Severe predictions back to Normal and costs about 8 Mild and 7 Severe hits.

## 10. Per-class metrics (selected model vs C1 vs C2)
| Class | Precision (sel / C1 / C2) | Recall | F1 |
|---|---|---|---|
| Normal | 0.710 / 0.736 / 0.712 | 0.845 / 0.784 / 0.866 | 0.772 / 0.759 / 0.781 |
| Mild | 0.297 / 0.305 / 0.288 | 0.177 / 0.220 / 0.112 | 0.221 / 0.255 / 0.160 |
| Moderate | 0.455 / 0.453 / 0.445 | 0.460 / 0.439 / 0.501 | 0.457 / 0.446 / 0.471 |
| **Severe** | **0.344 / 0.315 / 0.343** | **0.344 / 0.425 / 0.374** | 0.343 / 0.362 / 0.356 |

- **Selected model:** macro AUROC 0.726, QWK 0.571.
- **Severe** (91 ECGs): recall 0.34–0.49 across the leading models. Missed Severe ECGs lie near EF 31, have narrower
  QRS, and are mostly called Moderate (`PHASE5_ERROR_ANALYSIS.md` §8).

## 11. Confidence intervals
| Model | Accuracy | Corrected repeated-CV 95 % CI | Patient-cluster bootstrap (r0) |
|---|---|---|---|
| Selected (vote + prior, protected) | 0.561 | 0.520–0.603 | 0.531–0.597 |
| Vote + bias rule, recall floor | 0.561 | 0.518–0.604 | 0.524–0.590 |
| Class-unweighted MLP (not admissible) | 0.572 | 0.533–0.612 | 0.544–0.610 |
| Hierarchical (not admissible) | 0.567 | 0.533–0.601 | 0.535–0.600 |
| C1 vote ensemble | 0.547 | 0.504–0.591 | – |
| C2 calibrated MLP | 0.568 | 0.534–0.603 | – |

- **Selected model, other metrics** (r0 bootstrap): balanced accuracy 0.429–0.499; macro-F1 0.420–0.491.
- **Fold-level SD of accuracy:** 0.026–0.032.

## 12. Statistical tests
- **Selected vs C1:**
  - Δacc +0.014; corrected t p = 0.16; Holm (global, 43 candidates) 1.00; same sign in all repeats;
  - cluster-bootstrap ΔCI [0.000, +0.034]; McNemar p = 0.065.
- **Selected vs C2:** Δacc −0.007 (p 0.61).
- **Genuine improvements:** 0 vs C1 and 0 vs C2.
- **Smallest family-level Holm-adjusted p:** 0.06 (E7 class-unweighted CE vs the MLP fusion; not admissible).
- **Basis:** significance statements rest on the paired tests, not on CI overlap.

## 13. Was 65 % achieved?
**No.**
- **Best admissible:** 0.561.
- **Best of any eligible model:** 0.572 (with Mild recall 0.14).
- **Repeat-0-only exception:** a repeat-0-only run reached 0.577, but under the full three-repeat protocol it becomes
  0.572 and is not admissible. Selection rules were applied as pre-registered; correction C5-1.
- **Decision gate:** < 58 %, so investigate feature and data limitations.
- **Why:** Phase 4 showed that the ECG representation explains only R² ≈ 0.43 of echo EF. 65 % four-class accuracy
  would need an EF estimate correlating r ≈ 0.87 with the echo EF (classical-error simulation, Phase 4). The Mild class (an 11–13-point EF band) is
  confused in both directions.
- **Selective prediction:** on the 50 % most confident ECGs, accuracy is 0.66–0.68. That is a different, selective
  task (5B §7).

## 14. Generalisation limitations
- **Development-CV only:** the estimates of the best of 43 candidates are optimistic. Holm addresses multiplicity
  across the family, not selection optimism.
- **Locked test:** exposed in Phase 1 and not used. **An external or temporally independent cohort is required**
  before any performance claim.
- **Patient grouping by name only:**
  - 29 near-identical name pairs fall under different keys;
  - 11 of them straddle development and the locked test, so resolve them before any locked-test evaluation;
  - there is no evidence of leakage benefit (5B §5).
- **Labels:** from fuzzy name-matched echoes (93.5 % agreement with the nearest echo); measurement uncertainty unknown.
- **Data source:** a single centre, a single printer format, and a 2.5-s-per-lead digitised printout.

## 15. Recommendations for further research
1. **Data, not models.**
   - Verified ECG–echo links (MRN / encounter ID) and more abnormal-EF ECGs from new years or sites; the existing
     matching sheets contain almost no extra abnormal ECGs (Phase 4).
   - Digital waveforms (XML / DICOM) instead of PDF printouts.
2. **Reconsider the task granularity** with the clinical owner.
   - A binary reduced-EF endpoint reaches AUROC about 0.82 (Phase 4).
   - Selective four-class prediction with an explicit abstention rate is another option.
3. **If a four-class model is needed now,** the defensible choice is the selected model (vote ensemble + nested prior
   rule): accuracy 0.561 with every class recall ≥ 0.18. Alternatively, C1 for the best macro-F1 (0.455).
   Accuracy-maximising variants (0.57) should not be used, because they nearly abandon Mild.
4. **Before any locked-test evaluation:**
   - pre-register a shortlist (selected model, C1, C2);
   - resolve the 11 development / test name pairs;
   - obtain the owner's explicit authorisation;
   - report the test as previously exposed.
