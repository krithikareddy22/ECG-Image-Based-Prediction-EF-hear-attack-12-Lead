# Phase 4B / 4F: label quality, the high-confidence cohort G and label-quality training

- **Code:** `phase_4/audit_matching.py`, `cohorts.py`, `analysis4.py`, `label_quality.py`.
- **Results:** `results/audit_matching.json`, `cohort_summary.csv`, `g_cohort_analysis.json`,
  `g_vs_nonG_performance.csv` and `phase4_tables.md`.
- **Privacy:** every number is an aggregate over development ECGs; cells < 5 ECGs are suppressed.

## 1. Label provenance (summary of `PHASE4_DATA_AUDIT.md`) [VALID]
- **How the folder labels were made:** by matching each ECG to echo reports **by patient name only**, with fuzzy
  scores ≥ 50–80, no date window and no patient ID. Teichholz EF was then classified with the recovered
  sex-specific rule.
- **Folder label vs the class of the nearest-in-time linked echo** (label-agnostic choice):

  | Set | Agreement | Disagreements |
  |---|---|---|
  | All 1,047 ECGs | 93.5 % | 68 |
  | Development set D (837) | 93.4 % | – |

- **Link quality across all 1,047 ECGs:**
  - 91 ECGs have links implying different EF classes;
  - 22 have no link that reproduces their label;
  - the median EF spread across multiple links is 17 points.
- **Measurement uncertainty:** echo measurement uncertainty **cannot be quantified**. Only 18 repeat-echo pairs
  qualify (median |ΔEF| 0.0, 90th percentile 15.3), too few and too selected for a variability estimate. No claim
  about measurement error is made from them.
- **EF method:** Teichholz only, so differences between EF methods cannot contribute to label variability here.

## 2. The high-confidence cohort G (357 ECGs / 345 patients in D) [VALID]

### 2.1 Definition (label-agnostic; uses only the links)
G requires all of the following for the nearest-in-time linked echo:
- name score 100;
- |ECG − echo| ≤ 30 days;
- echo sex agrees with the ECG where recorded;
- no other linked echo of a different EF class within 30 days.

It does **not** use the folder label, so selecting G cannot directly select "correct" labels.

### 2.2 Composition (D only; `g_cohort_analysis.json`)

| | G | non-G |
|---|---|---|
| ECGs / patients | 357 / 345 | 480 / 469 |
| Normal / Mild / Moderate / Severe | 47.6 / 22.1 / 20.2 / 10.1 % | 44.6 / 22.1 / 21.9 / 11.5 % |
| Year 2020 / 2021 / 2022 / 2024 | 11.5 / 29.1 / 27.7 / 31.7 % | 31.0 / 28.5 / 15.8 / 24.6 % |
| Male / mean age | 62.5 % / 57.0 y | 62.9 % / 56.4 y |
| Mean (SD) nearest-echo EF | 53.6 (17.6) | 53.4 (17.5) |
| Median ECG–echo interval / same-day share | 0 d / 78.7 % | 41 d / 35.4 % |
| ECGs with a single linked echo | 85.7 % | 84.2 % |
| Folder vs nearest-echo agreement | **95.8 %** | 91.7 % |

- **What G does not change:** the class mix, sex, age and EF distribution are nearly identical in G and non-G.
- **What G does change:**
  - it shifts the **interval** (same-day echoes);
  - it shifts the **year**: 2020, whose matching used the loosest threshold (score ≥ 50), is under-represented.

### 2.3 Why the 95.8 % agreement is high, and what it does *not* mean
1. **It is largely an internal-consistency measure.**
   - The folder labels were produced *from the same matching sheets* whose links define G. When an ECG has only
     one linked echo, its folder label must come from that echo.
   - Across all of D, agreement is **98.0 % for the 710 single-link ECGs vs 67.7 % for the 127 multi-link ECGs**.
   - Disagreement therefore comes almost entirely from **link ambiguity** (which echo was used), not from an
     independent judgement that a label is wrong.
2. **G's 15 disagreements:**
   - in 8, the folder label equals the matching sheet's *own* LV class for the nearest echo. That class differs
     from the recomputed rule (1.1 % of sheet rows), e.g. boundary handling or rounding;
   - in 6, the folder label equals the class of *another* linked echo (farther in time). Counts may overlap.
3. **Consequences:**
   - 95.8 % is an upper bound on how well the folder labels agree with the best available EF evidence. It is
     **not** a validated label accuracy: neither source is independent of the name matching.
   - Agreement says nothing about echo measurement error (unquantifiable here), so even G labels may be wrong by
     the measurement noise near the boundaries.

### 2.4 Is G "easier"? Decomposing the performance gap [VALID]
Every model is more accurate on G than on non-G. Accuracy is against folder labels; seed-combined, mean over 3
repeats.

| Model | acc G | acc non-G | non-G reweighted to G class mix | non-G where folder = echo class | gap total | due to class mix | remaining |
|---|---|---|---|---|---|---|---|
| Vote ensemble | 0.588 | 0.517 | 0.528 | 0.545 | +0.072 | +0.012 | +0.060 |
| MLP fusion | 0.582 | 0.515 | 0.527 | 0.546 | +0.066 | +0.011 | +0.055 |
| Multitask λ selected | 0.586 | 0.515 | 0.528 | 0.547 | +0.071 | +0.012 | +0.059 |
| Ridge EF regression (tuned) | 0.570 | 0.507 | 0.521 | 0.537 | +0.064 | +0.014 | +0.049 |
| MLP fusion trained on G only | 0.572 | 0.526 | 0.537 | 0.562 | +0.047 | +0.012 | +0.035 |
| MLP fusion trained on EF-derived labels | 0.575 | 0.522 | 0.533 | 0.552 | +0.054 | +0.012 | +0.042 |

- **Class mix:** explains only **1–2 points** of the 5–8-point gap.
- **Label agreement:** restricting non-G to ECGs whose folder label agrees with the echo class recovers about
  3 points (0.517 → 0.545 for the vote ensemble). Label disagreement therefore explains roughly half of the
  remaining gap.
- **The rest:** about 3 points stays unexplained. [HYPOTHESIS] Other selection effects of G: same-day echoes,
  fewer 2020 ECGs, and fewer ambiguous patients.
- **Conclusion:** any metric computed on G is **optimistic for reasons unrelated to the model**. Evaluating the
  benchmark on G already raises its accuracy from 0.547 to 0.588.
  - All model comparisons in Phase 4 are therefore made on **all 837 ECGs with folder labels** (primary).
  - On G, models are compared only **paired against the benchmark on the same ECGs**. There, no Phase 4 model
    differs from the vote ensemble: |ΔS3| ≤ 0.04, all p ≥ 0.26 (`phase4_tables.md`, "Evaluation-set effects").

## 3. Label-quality training (4F) [VALID]
- **Setup:** MLP fusion (Phase 2 configuration), 3 seeds × 3 repeats. Only training and inner-val selection
  change; evaluation is always on all 837 ECGs.
- **Comparisons:** against the matched control (standard MLP fusion) and against the vote ensemble.

| Variant | Training ECGs per fold (≈) | Accuracy | Balanced acc. | Macro-F1 | F1 Mild | S3 | ΔS3 vs MLP fusion (p) | ΔS3 vs vote (p) |
|---|---|---|---|---|---|---|---|---|
| MLP fusion (control) | 535 | 0.544 | 0.466 | 0.440 | 0.186 | 0.483 | – | −0.007 (0.65) |
| `lq_dropdisagree`: drop folder ≠ echo class | ≈ 500 | **0.554** | **0.471** | 0.449 | 0.197 | **0.491** | +0.008 (0.56) | +0.001 (0.94) |
| `lq_trainG`: train on G only | ≈ 228 | 0.546 | 0.467 | 0.444 | 0.206 | 0.485 | +0.002 (0.95) | −0.005 (0.88) |
| `lq_relabel`: EF-derived labels | 535 | 0.544 | 0.465 | 0.442 | 0.196 | 0.484 | +0.000 (0.99) | −0.006 (0.71) |
| `lq_trainC`: train on C only | ≈ 350 | 0.542 | 0.455 | 0.433 | 0.187 | 0.477 | −0.007 (0.74) | −0.013 (0.44) |

- **Cleaner training labels give no measurable gain.**
  - Removing the 6.6 % of training ECGs whose folder label contradicts the nearest echo gives the highest point
    estimate of all Phase 4 models (S3 0.491, accuracy 0.554). It is still indistinguishable from the control and
    from the vote ensemble (Holm p = 1.00).
- **Training only on G** keeps performance unchanged with 43 % of the data. That is mildly encouraging about label
  quality, but it does not improve anything.
- **Effect on the G gap:** models trained on G or on EF-derived labels have a **smaller** G vs non-G gap
  (+0.047 / +0.054 vs +0.072). They do slightly better on non-G and slightly worse on G. This is consistent with
  training-label choice shifting *which* ECGs are right, not *how many*.
- **Answer to "Is label uncertainty a plausible performance limitation?"**
  - **As a hard ceiling, no.** The labels agree with the best available EF evidence 93.4 % of the time, far above
    the 55 % accuracy reached. Cleaning or replacing training labels does not move performance.
  - **As a contributor to measured error, yes, modestly.** Folder/echo disagreements and borderline EF explain part
    of the errors: accuracy is 0.37 within 5 EF points of a boundary vs 0.74 at ≥ 10 points
    (`PHASE4_MILD_CLASS_ANALYSIS.md`).
  - The dominant limitation is the information about EF that this ECG representation carries:
    R² ≈ 0.43 (`PHASE4_REGRESSION_REPORT.md`).

## 4. Limitations
- **No ground truth:** without patient identifiers, no link can be verified, and G is a proxy.
- **Not independent:** the folder labels and the EF evidence come from the same name-matching process, so
  agreement cannot validate either.
- **Measurement noise:** unknown (18 repeat pairs).
- **Power:** label-quality effects smaller than about 0.02 S3 cannot be detected with 837 ECGs.
