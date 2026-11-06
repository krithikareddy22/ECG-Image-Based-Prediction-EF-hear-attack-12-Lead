# Phase 5B: four-class error analysis

- **Script:** `error_analysis/analyse.py`.
- **Output:** `results/error_analysis.json` (aggregates; cells < 5 ECGs suppressed).
- **Figure:** `results/plots/confidence_calibration.png`.
- **Models:** C1 vote ensemble, C2 calibrated MLP (56.8 %) and MLP fusion; outer-fold predictions on 837 development
  ECGs, averaged over 3 repeats.
- **Echo EF:** nearest echo, from the Phase 4 private table. It is used for analysis only, never as a model input.

## 1–4. Confusion pairs and boundary proximity [VALID]
Mean error counts over repeats; "within 5" = echo EF within 5 points of the boundary the error crossed.

| Pair (true → predicted) | C1: errors (share of true class) | C1: within 5, errors vs correct | C2: errors (share) |
|---|---|---|---|
| Normal → Mild | 53.0 (13.8 %) | 12 % vs 5 % | 24.3 (6.3 %) |
| **Mild → Normal** | **65.3 (35.3 %)** | 35 % vs 30 % | **82.3 (44.5 %)** |
| **Mild → Moderate** | **50.3 (27.2 %)** | 60 % vs 53 % | 59.7 (32.3 %) |
| Moderate → Mild | 26.7 (15.1 %) | 43 % vs 47 % | 16.3 (9.2 %) |
| Moderate → Severe | 41.0 (23.2 %) | 50 % vs 49 % | 35.0 (19.8 %) |
| **Severe → Moderate** | 28.3 (31.1 %) | **58 % vs 33 %** | 32.7 (35.9 %) |
| Normal → Severe | 14.7 (3.8 %) | 0 % | 8.3 (2.2 %) |
| Severe → Normal | 11.0 (12.1 %) | 9 % vs 0 % | 15.3 (16.8 %) |

- **Mild:** errors go both ways (35 % → Normal, 27 % → Moderate). They sit near the boundaries, but so do correct
  Mild predictions, because the Mild band is only 11–13 EF points wide. The error concentration near the boundary is
  modest (Mild → Normal 35 % vs 30 %).
- **Calibration** (C2) moves Mild errors toward Normal (44.5 %). This is how it gains accuracy.
- **Severe:** missed Severe ECGs lie much closer to EF 31 (58 % within 5 points, vs 33 % of detected Severe).
- **Adjacent classes:** most errors are between neighbouring classes. Normal ↔ Severe confusions are rare (2–4 % of
  Normal) and far from any boundary.

## 3. ECG–echo interval [VALID, descriptive]
| Interval | ECGs | C1 accuracy | C2 accuracy |
|---|---|---|---|
| same day | 451 | 0.517 | 0.544 |
| 1–14 d | 99 | 0.700 | 0.724 |
| 15–90 d | 138 | 0.548 | 0.563 |
| > 90 d | 149 | 0.535 | 0.544 |

There is no monotone degradation with interval. The 1–14 d group is unexpectedly easy (n = 99). [HYPOTHESIS] This
reflects case mix (e.g. scheduled follow-up), not timing.

## 4. Signal quality (Phase 5 `sq` v2 features) [VALID, descriptive]
| Group | ECGs | C1 accuracy | C2 accuracy |
|---|---|---|---|
| 0 poor leads | 440 | 0.550 | 0.570 |
| 1 poor lead | 198 | 0.574 | 0.596 |
| ≥ 2 poor leads | 199 | 0.514 | 0.538 |
| HF-power tertile low / mid / high | 279 each | 0.496 / 0.546 / 0.600 | 0.528 / 0.568 / 0.609 |

- **Poor leads:** ECGs with ≥ 2 flagged leads are 3–4 points less accurate.
- **"High-frequency power"** *correlates with higher accuracy*. In vector-digitised printouts it measures sharp QRS
  content rather than noise, so it is **confounded** and must not be read as a noise level.

## 5. Repeated ECGs and possible split identities [VALID]
| Group | ECGs | C1 | C2 | MLP fusion |
|---|---|---|---|---|
| single-ECG patient keys | 756 | 0.560 | 0.583 | 0.557 |
| keys with ≥ 2 ECGs | 81 | **0.428** | **0.428** | **0.420** |
| ECGs in possible split identities (near-identical names under different keys) | 29 | 0.494 | 0.540 | 0.529 |
| all other ECGs | 808 | 0.549 | 0.569 | 0.544 |

- **Multi-ECG patients are much harder** (about 15 points lower accuracy). The same prediction is given to all ECGs of
  a key in 54–58 % of keys. [HYPOTHESIS] Repeat recordings belong to sicker or changing patients, whose echo label may
  differ between visits.
- **No sign of leakage through split identities:** these ECGs are *less* accurately classified than the rest, not more.

## 6. Lead-wise informativeness [VALID, descriptive]
Univariate AUROC, EF < 41 (Moderate + Severe) vs Normal, of per-lead median-beat features:
- **QRS width:** most informative in V3–V6 and aVR (AUROC 0.74–0.78).
- **T amplitude:** in I and aVL (0.74).
- **R amplitude:** in V1–V3 (0.71–0.73; poor R progression).
- **QRS area:** in aVL, III and V1 (0.72–0.73).
- **Inferior leads** (II, aVF) are least informative.

## 7. Confidence and calibration [VALID]
| Model | ECE | Accuracy, top 25 % confidence | top 50 % | top 75 % | all |
|---|---|---|---|---|---|
| C1 vote | 0.087 | 0.684 | 0.658 | 0.602 | 0.547 |
| C2 calibrated MLP | 0.104 | 0.716 | 0.675 | 0.607 | 0.568 |
| MLP fusion | 0.086 | 0.716 | 0.673 | 0.601 | 0.544 |

- **Calibration:** probabilities are moderately calibrated (ECE 0.09–0.10).
- **Selective prediction:** on the most confident half of ECGs, accuracy reaches 0.66–0.68. That is above 65 %, but
  for a **selective** model that abstains on half the ECGs; it is not the four-class overall-accuracy target.

## 8. Severe (91 ECGs) [VALID]
| Model | Recall | Precision | Missed Severe predicted as N / Mi / Mo | Median echo EF: detected vs missed | Median printed QRSD: detected vs missed | False Severe from N / Mi / Mo |
|---|---|---|---|---|---|---|
| C1 vote | 0.425 | 0.315 | 11.0 / 13.0 / 28.3 | 21.8 vs 27.7 | 111 vs 101.5 ms | 14.7 / 28.7 / 41.0 |
| C2 cal. MLP | 0.374 | 0.343 | 15.3 / 9.0 / 32.7 | 23.8 vs 27.3 | 112.5 vs 102 ms | 8.3 / 22.3 / 35.0 |
| MLP fusion | 0.487 | 0.303 | 11.0 / 6.7 / 29.0 | 24.4 vs 27.4 | 111 vs 100 ms | 17.0 / 35.0 / 49.7 |

- **Detection depends on EF depth and QRS width:** Severe is detected when EF is very low and QRS is wide. Missed
  Severe ECGs are near EF 31 with narrower QRS and are mostly called Moderate.
- **False Severe calls** come mainly from Moderate.

## Conclusions for Phase 5C / 5D
1. **The dominant error source is the Mild band.** It is confused in both directions, and its errors are only slightly
   more boundary-concentrated than its correct predictions. This is an information limit, not a decision-rule
   problem.
2. **Severe errors are boundary-driven** (EF near 31). QRS width carries Severe information, which motivates the QRS
   morphology family (5C).
3. **Signal quality:** low quality explains only a small loss (3–4 points on 24 % of ECGs), so E6 can gain at most
   about 1 point overall.
4. **Repeat-ECG patients** are a hard subgroup (81 ECGs; 15 points below the rest).
5. **Selective prediction** is the only route to ≥ 65 % on these data, and only on a subset. This is reported, not
   pursued as the primary objective.
