# Phase 6B exploratory report: classifying Mild from the ECG image only (MILDFEAT)

- **Data:** development set (837 ECGs); patient-grouped Phase 2 folds × 3 repeats; original labels. The locked test
  set was not used.
- **Declaration:** before any model run (`AMENDMENT_MILDFEAT.md`, commit 42dc57f).
- **Code:** `mildfeat.py` (features), `mildmodels.py` (models).
- **Results:**
  - `results/mildfeat_models.json` / `.csv`, `mildfeat_comparisons.csv`;
  - `mildfeat_doctors_table.csv`, `mildfeat_findings_by_class.csv`, `mildfeat_validation.json`;
  - `plots/mildfeat_models.png`, `plots/mildfeat_doctors_table.png`.
- **Format:** aggregates only, means of 3 repeats, corrected repeated-CV 95 % CIs.
- **Status:** **EXPLORATORY**.

## Answer
1. **The ECG can tell Mild from Normal reasonably well.** It cannot tell Mild from Moderate.
   - The best image-only model ranks a Mild ECG above a Normal one 74–76 % of the time (AUROC Normal-vs-Mild 0.740–0.757).
   - Between Mild and Moderate it does so only 59–63 % of the time.
   - Four-class Mild F1 therefore stays at 0.16–0.26 for every model, with Mild recall ≤ 0.24.
2. **This is how Mild LV dysfunction looks on an ECG**, as a cardiologist would read it:
   - About half of Mild ECGs show the same abnormalities as Moderate ones:
     - spatial QRS-T angle > 100°: Normal 20 %, Mild 52 %, Moderate 53 %;
     - poor R-wave progression: 7 / 31 / 43 %;
     - T inversion in ≥ 2 leads: 12 / 31 / 40 %.
   - The other half look normal.
   - The ECG reflects whether the ventricle is diseased (scar, conduction, repolarisation). It does not reflect where
     the EF falls within 41–52 %. An EF-class boundary inside that band is invisible to the ECG.
3. **Hand-crafted clinician features do not beat the ECGFounder embedding.** Adding them changes nothing (16 tests,
   all Holm p = 1.00). On their own they reach 0.51 against 0.54 for ECGFounder; the foundation model already encodes
   the same information.
4. **The most useful new, interpretable marker is the spatial QRS-T angle.** It comes from the synthesised
   simultaneous 12-lead beat (Kors VCG):
   - median 56° in Normal vs 105° in Mild;
   - AUROC Normal-vs-Mild 0.70 [0.66, 0.75], as good as the whole 118-feature logistic model (0.70).
5. **Best way to use the ECG for Mild with this data:**
   - Do not treat Mild as a separate class the ECG can recognise.
   - Treat LV function as a severity scale: first "reduced vs normal" (AUROC 0.83–0.84), then grading.
   - Among the four-class models, the ordinal model (CLIN-ORD) gives the best Mild recall of the feature models (0.240,
     Mild F1 0.251). The equal-weight IMG-vote gives the best Mild F1 overall (0.264).
   - Neither is a genuine improvement.
6. **70 % is not reached by any image-only model.** The best four-class accuracy is IMG-C3 / IMG-C3+CLIN at 0.558.

## 1. Features: what was measured (121 features, all from the drawing)
- **New in this phase:** the 10-s rhythm strip is recorded at the same time as all four printed 2.5-s columns.
  - Its R peaks time-stamp the beats in every column.
  - Each lead's median beat is aligned to the same beat, which gives a synthesised simultaneous median 12-lead beat.
  - That makes global fiducials (QRS onset / offset from the 12-lead spatial velocity, T end, P wave) and vector
    measures possible. Phase 5 had concluded that inter-lead timing was impossible.
- **Groups:**
  - rhythm;
  - intervals (QRS, PR, P, QT, QTc, JT, Tpeak–Tend);
  - conduction (R-peak time, LBBB / RBBB / IVCD-like);
  - scar (pathological Q and fragmented QRS per territory, R progression, transition);
  - voltage (Sokolow-Lyon, Cornell, Peguero, QRS sum, low voltage);
  - repolarisation (ST J+60, T amplitude / area, T inversion, flat T, strain, T/R ratio);
  - atrial (P amplitude, P-terminal force V1, P axis);
  - vector (spatial / frontal QRS-T angle, ventricular gradient, QRS / T vector magnitudes, axes).
- **Validation against the printed machine measurements** (label-free; the machine values are never a model input):

  | Measure | Agreement |
  |---|---|
  | Heart rate | r 0.91 |
  | QRS duration | r 0.82, −10 ms |
  | PR | r 0.74, −19 ms |
  | QT | r 0.61, −20 ms |
  | QRS axis | median difference 3.8°, 97 % within 30° |
  | P axis | median difference 6.8° |
  | T axis | 73 % within 30° |

  - QRS duration agrees much better than the Phase 5 estimator (r 0.75, −29 ms bias).
  - Thresholds were chosen by Pearson r with the machine value (QRS 0.10 of peak spatial velocity; T end as the median
    of lead tangents; P onset at 0.5 of P amplitude).
  - There were 0 extraction errors; missingness is ≤ 7 % per feature, except R-peak time V1 at 25 % (QS complexes).

## 2. Models (four-class, original labels, all 837 ECGs)
| Model | Inputs | Accuracy [95 % CI] | Macro-F1 | Mild recall | Mild precision | Mild F1 | AUROC N-vs-Mild | AUROC Mild-vs-Mod | AUROC reduced-vs-N | Admissible |
|---|---|---|---|---|---|---|---|---|---|---|
| ECGF-MLP (baseline) | ECGFounder | 0.542 [0.505, 0.578] | 0.446 | 0.207 | 0.283 | 0.239 | 0.745 | 0.614 | **0.838** | yes |
| IMG-vote (baseline) | ECGFounder ×2 | 0.541 [0.502, 0.580] | **0.450** | 0.238 | **0.297** | **0.264** | 0.740 | 0.614 | 0.833 | yes |
| IMG-C3 (baseline; cloud 0.559) | ECGFounder ×2 + rule | **0.558** [0.504, 0.611] | 0.441 | 0.187 | 0.290 | 0.227 | 0.740 | 0.614 | 0.833 | yes |
| CLIN-LR | clinician features | 0.509 [0.458, 0.561] | 0.417 | 0.177 | 0.244 | 0.205 | 0.701 | 0.598 | 0.802 | yes |
| CLIN-ORD (ordinal) | clinician features | 0.508 [0.464, 0.553] | 0.428 | **0.240** | 0.264 | 0.251 | 0.702 | 0.590 | 0.801 | yes |
| CLIN-GBM | clinician features | 0.512 [0.471, 0.553] | 0.418 | 0.211 | 0.259 | 0.232 | 0.717 | 0.594 | 0.807 | yes |
| IMG-MLP+CLIN | features ‖ ECGFounder | 0.539 [0.502, 0.576] | 0.426 | 0.115 | 0.259 | 0.159 | **0.757** | **0.630** | 0.833 | no |
| IMG-vote+CLIN | ECGFounder ×2 + CLIN-LR | 0.536 [0.501, 0.570] | 0.435 | 0.204 | 0.282 | 0.236 | 0.745 | 0.614 | 0.837 | yes |
| IMG-C3+CLIN | the same + rule | **0.558** [0.515, 0.600] | 0.437 | 0.168 | 0.284 | 0.210 | 0.745 | 0.614 | 0.837 | yes |
| *Reference with header: C3* | header + ECGFounder | *0.561 [0.520, 0.603]* | *0.448* | *0.177* | | | | | | *yes* |

- **Training cost** (CPU, per 5-fold run): CLIN-LR / ORD / GBM 0.03–0.04 min; IMG-MLP+CLIN 0.5 min.
- **Paired comparisons** (corrected repeated-CV t, Holm over 16 tests: 4 pairs × accuracy, macro-F1, Mild F1, Mild
  recall): **no genuine improvement; every Holm p = 1.00.**

  | Pair | Δ accuracy | Δ macro-F1 | Δ Mild F1 | Δ Mild recall |
  |---|---|---|---|---|
  | IMG-vote+CLIN vs IMG-vote | −0.005 | −0.014 | −0.028 | −0.034 |
  | IMG-C3+CLIN vs IMG-C3 | +0.000 | −0.004 | −0.017 | −0.020 |
  | IMG-MLP+CLIN vs ECGF-MLP | −0.003 | −0.021 | −0.079 | −0.092 |
  | CLIN-ORD vs CLIN-LR | −0.001 | +0.011 | +0.046 | +0.063 |

- **Reproducibility:** the image-only C3 recomputed on this machine gives 0.558, against the cloud session's 0.559.

## 3. A cardiologist's reading: which ECG signs separate the classes (descriptive; no model uses this selection)
**Normal vs Mild:** univariate AUROC with patient-bootstrap 95 % CI; all have BH q < 1e-7.

| Sign | Normal (median) | Mild (median) | AUROC [95 % CI] |
|---|---|---|---|
| Spatial QRS-T angle | 56° | 105° | 0.70 [0.66, 0.75] |
| Spatial peak QRS-T angle | 35° | 74° | 0.69 [0.64, 0.73] |
| R amplitude V4 (lower in Mild) | 1.02 mV | 0.65 mV | 0.31 [0.26, 0.36] (= 0.69 inverted) |
| R amplitude V3 (lower in Mild) | 0.95 mV | 0.56 mV | 0.31 [0.26, 0.37] |
| QRS duration | 78 ms | 82 ms | 0.69 [0.64, 0.73] |
| T amplitude lead I / lateral (flatter in Mild) | 0.15 mV | 0.08 mV | 0.34 [0.29, 0.39] |
| Frontal QRS-T angle | – | – | 0.65 [0.60, 0.70] |

**Mild vs Moderate:** every sign is weak. The best are T amplitude in III (0.63 [0.57, 0.69]), R-peak time V1 (0.62),
anterior pathological Q (0.60) and QRS duration (0.59, 82 vs 88 ms).

**Share of each class with standard abnormal findings** (`results/mildfeat_findings_by_class.csv`):

| Finding | Normal | Mild | Moderate | Severe |
|---|---|---|---|---|
| Spatial QRS-T angle > 100° | 20 % | 52 % | 53 % | 69 % |
| QRS ≥ 110 ms | 2 % | 14 % | 24 % | 36 % |
| Poor R progression (R V3 ≤ 0.3 mV) | 7 % | 31 % | 43 % | 36 % |
| T inversion in ≥ 2 leads | 12 % | 31 % | 40 % | 39 % |
| Lateral strain | 6 % | 13 % | 18 % | 20 % |
| QTc (Bazett) > 450 ms | 11 % | 16 % | 24 % | 43 % |
| P-terminal force V1 > 4 mV·ms | 7 % | 14 % | 12 % | 23 % |
| Heart rate > 100 | 12 % | 15 % | 22 % | 32 % |

**Reading:**
- Every marker rises from Normal to Severe.
- The step from Normal to Mild is large, and the step from Mild to Moderate is small.
- The ECG grades "how much LV disease" coarsely. A Mild ECG is either normal-looking or looks like Moderate. That
  ceiling cannot be removed by better features from the same ECGs.

## 4. Limitations
- **Input quality:**
  - PDF digitisation gives 2.5 s per lead, so the median beats rest on 1–3 beats per column.
  - The synthesised 12-lead beat assumes stable morphology across the 10 s; non-dominant beats were removed.
- **Measurement:**
  - The pathological-Q criterion over-calls: 41 % of Normal ECGs meet it, mostly from the 25 %-of-R rule with small
    R waves.
  - QT agreement is moderate (r 0.61).
- **Design:** no header, age or sex, by request. EF is Teichholz, linked by name matching (see `PHASE6B_AUDIT.md`).
- **Evidence level:** development CV only, exploratory; the locked test set was not used.

## 5. For phase 2 (about 100,000 patients)
- **Inputs:** use the raw digital 12-lead ECG (simultaneous 10 s, 500 Hz) rather than PDF printouts. The vector
  measures (spatial QRS-T angle, ventricular gradient) then become exact instead of synthesised.
- **Targets:**
  - predict EF as a continuous or ordinal target, and derive classes from it;
  - report "reduced vs normal" screening and severity grading separately, rather than a hard four-class Mild label;
  - prefer biplane (Simpson) EF over Teichholz, with a verified patient ID link.
- **Expected gain:** with that many patients, a CNN on the raw signal can learn the subtle Mild pattern. These
  development results suggest the four-class Mild boundary will remain the hardest part.
