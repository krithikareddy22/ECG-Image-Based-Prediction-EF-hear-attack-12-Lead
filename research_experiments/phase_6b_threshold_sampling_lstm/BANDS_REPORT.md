# Phase 6B exploratory report: BANDS (sensitivity of four-class accuracy to the EF band limits)

Declared in `AMENDMENT_BANDS.md` (commit e00191e). Code `bands.py`. Results:
- `results/bands_sweep.csv`: all 3,313 definitions;
- `results/bands_exploratory.json`: sweep summary;
- `results/bands_confirm.json`: retrained classifiers.

Header-free; 837 development ECGs; patient-grouped 5-fold CV × 3 repeats; all tuning nested. The locked test set was not
used. **Sensitivity analysis:** different band limits define different tasks, so each row is shown with its chance
level and Cohen's κ. The original folder-label result (C3 0.561; header-free ENH-vote 0.569) remains primary.

## How the sweep was done
- Labels are re-derived from the nearest-echo EF with limits c_N (Normal; women + offset), c_Mi (Mild / Moderate) and
  c_S (Moderate / Severe).
- **Grid:** c_N 45–56, c_Mi 35–46, c_S 25–36, offset 0 or 2: 3,313 definitions with every class ≥ 10 ECGs.
- **Feasible** ("reasonable size"): every class ≥ 60 ECGs and both middle bands ≥ 5 EF points wide. 1,457 definitions.
- **Sweep model:** one header-free EF regression, graded with nested cut points.
- **Confirmation:** the selected points were confirmed by retraining the classifiers on the new labels.

## Confirmed results (retrained classifiers; means over 3 repeats)
| Definition | Limits (Normal / Mild-Moderate / Moderate-Severe) | Size N / Mi / Mo / S | Accuracy [95 % CI] | κ | Chance accuracy | Detected N / Mi / Mo / S |
|---|---|---|---|---|---|---|
| Original limits on echo EF (ORIG-EF) | > 52 M, ≥ 54 F / 41 / 31 | 415 / 170 / 165 / 87 | 0.594 [0.560, 0.627] | 0.353 | 0.337 | 91 / 20 / 32 / 38 % |
| **ESC 2021 (clinical)** | ≥ 50 / 41 / 30 | 446 / 139 / 176 / 76 | **0.626 [0.588, 0.664]** | 0.384 | 0.364 | 90 / 17 / **42** / 35 % |
| MAX-ACC-DET (best accuracy with Mild and Moderate ≥ 25 % in the sweep) | > 51 / 38 / 31 | 439 / 197 / 114 / 87 | 0.619 [0.593, 0.645] | 0.380 | 0.360 | 90 / **28** / 25 / 43 % |
| MAX-KAPPA (best agreement beyond chance) | > 52 / 46 / 29 | 428 / 70 / 272 / 67 | 0.634 [0.591, 0.678] | 0.399 | 0.380 | 87 / 12 / 49 / 27 % (not admissible) |
| Reference: folder labels, ENH-vote | original | 384 / 185 / 177 / 91 | 0.569 [0.527, 0.610] | – | 0.336 | 88 / 17 / 46 / 29 % |

## One-dimensional sweeps (EF grading; the other two limits at their original values)
| Normal limit c_N | 45* | 46 | 47 | 48 | 49 | 50 | 51 | **52** | 54 | 56 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.646 | 0.632 | 0.621 | 0.617 | 0.611 | 0.607 | 0.595 | 0.587 | 0.587 | 0.579 |
| Mild detected | 0.09 | 0.07 | 0.09 | 0.09 | 0.11 | 0.12 | 0.12 | 0.12 | 0.12 | 0.14 |

| Mild / Moderate limit c_Mi | 35* | 36 | 37 | 38 | 39 | 40 | **41** | 43 | 44 | 46 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.623 | 0.597 | 0.593 | 0.591 | 0.583 | 0.587 | 0.587 | 0.601 | 0.612 | 0.631 |
| Mild / Moderate detected | 0.44 / 0.10 | 0.33 / 0.18 | 0.31 / 0.14 | 0.24 / 0.25 | 0.15 / 0.38 | 0.12 / 0.44 | 0.12 / 0.43 | 0.14 / 0.55 | 0.13 / 0.57 | 0.14 / 0.56 |

| Moderate / Severe limit c_S | 25* | 27* | 29 | 30 | **31** | 33 | 35 | 36 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.620 | 0.617 | 0.604 | 0.595 | 0.587 | 0.585 | 0.592 | 0.591 |
| Moderate detected | 0.60 | 0.59 | 0.56 | 0.47 | 0.43 | 0.31 | 0.17 | 0.14 |

\* not feasible: a class has fewer than 60 ECGs. **Bold** = original limit.

## Interpretation
- **Lowering the Normal limit raises accuracy** (52 → 45: +6 points) mainly by moving ECGs into the class the ECG
  recognises best (Normal). κ hardly moves (0.34–0.36).
- **The Mild / Moderate limit trades Mild against Moderate.** At 38, both are detected about 25 % of the time. At 41
  (original), Mild ≈ 12 %; at 35, Moderate ≈ 10 %. No limit detects both well, because the ECG cannot locate EF
  precisely inside 31–52.
- **Recommended sensitivity definition:** ESC 2021, which is clinically justified (heart-failure guideline classes:
  preserved ≥ 50, mildly reduced 41–49, reduced ≤ 40, with < 30 as severe).
  - Accuracy 0.626, all four classes ≥ 15 % detected, Moderate 42 %, κ 0.384 vs 0.353 on the original limits.
  - It should be reported as a sensitivity analysis next to the original-label result, not instead of it.
- **Optimism:** MAX-ACC-DET and MAX-KAPPA were selected over 1,457 feasible definitions on the same data, so they are
  optimistic. ESC 2021 was fixed in advance, so it is the least biased of the alternatives.
- **Label set:** ORIG-EF (0.594) is not the folder labels. It relabels the 55 ECGs whose folder label disagrees with the
  nearest echo, so it includes a definition effect, as ECHO did in MMCLEAN.
