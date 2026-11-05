# Phase 5A: four-class baseline

- **Scripts:** `baseline/import_existing.py`, `baseline/calibrated_mlp.py`, `experiments/nested.py`, `baseline/report.py`.
- **Results:** `results/phase5_baseline_results.{csv,json}`, `results/phase5_baseline_tables.md`.
- **Per-ECG predictions:** `data/private/runs/` (git-ignored).

## Population and protocol [VERIFIED]
- **Data:** 837 development ECGs / 794 name-based patient keys (Normal 384, Mild 185, Moderate 177, Severe 91). The
  locked test is not used.
- **Validation:** patient-grouped 5-fold CV × 3 repeats, the identical folds for every model. Seeds are combined
  within a repeat; metrics are means over repeats.
- **CIs:**
  - corrected repeated-CV 95 % CI (15 cells);
  - patient-cluster bootstrap on repeat 0 (in the JSON).
- **Paired tests:** corrected t-test against C1 (vote) and C2 (calibrated MLP).
- **Admissible:** every class recall ≥ 0.15, the pre-registered floor.

## Verified reproduction of the 56.8 % benchmark
- **Method:** the calibrated MLP was re-run with the phase-2 nested design:
  - outer predictions = the stored Phase 2 MLP-fusion outer-fold predictions, seed mean;
  - α chosen on a patient-grouped 4-fold inner CV inside each training portion.
- **Accuracy per repeat:** **0.5711 / 0.5687 / 0.5651**, identical to Phase 2, with identical per-fold α.
- **Per-ECG predictions:** now stored, so paired comparisons against 56.8 % are possible for the first time.

## Baseline table (development CV)
| Model | Accuracy [corrected 95 % CI] | Balanced acc. | Macro-F1 | Severe recall / precision | Min class recall | Admissible | Δacc vs C1 (p) | Δacc vs C2 (p) |
|---|---|---|---|---|---|---|---|---|
| Calibrated MLP, accuracy objective | **0.572** [0.537, 0.608] | 0.443 | 0.423 | 0.256 / 0.362 | 0.092 (Mild) | no | +0.025 (0.06) | +0.004 (0.67) |
| **C2: calibrated MLP, protected (56.8 %)** | **0.568** [0.534, 0.603] | 0.463 | 0.442 | 0.374 / 0.343 | 0.112 (Mild) | no | +0.021 (0.18) | – |
| Vote + EF-regression member (Phase 4) | **0.560** [0.516, 0.604] | 0.458 | 0.446 | 0.407 / 0.323 | 0.182 | **yes** | +0.013 (0.42) | −0.008 (0.60) |
| EF regression, clinical thresholds (Phase 4) | 0.558 | 0.416 | 0.418 | 0.132 / 0.485 | 0.132 (Severe) | no | +0.011 (0.51) | −0.010 (0.56) |
| Calibrated MLP, recall-floor objective (new, pre-registered) | 0.549 | 0.466 | 0.442 | 0.458 / 0.316 | 0.137 (Mild) | no | +0.002 (0.90) | −0.020 (0.10) |
| Stacking (Phase 3, cross-fitted) | 0.547 | **0.473** | 0.442 | **0.542** / 0.307 | 0.153 | yes | +0.000 (1.00) | −0.021 (0.31) |
| **C1: vote ensemble** | 0.547 [0.504, 0.591] | 0.467 | **0.455** | 0.425 / 0.315 | 0.220 | **yes** | – | −0.021 (0.18) |
| Multitask λ selected (Phase 4) | 0.546 | 0.463 | 0.444 | 0.443 / 0.296 | 0.166 | yes | −0.002 (0.90) | −0.023 (0.10) |
| Weighted-average ensemble (Phase 3) | 0.544 | 0.462 | 0.447 | 0.436 / 0.311 | 0.193 | yes | −0.003 (0.77) | −0.024 (0.07) |
| MLP fusion (Phase 2) | 0.544 [0.505, 0.583] | 0.466 | 0.440 | 0.487 / 0.303 | 0.141 (Mild) | no | −0.004 (0.78) | −0.025 (0.04) |
| 3-way MLP fusion (+ clinical) | 0.542 | 0.466 | 0.445 | 0.484 / 0.309 | 0.166 | yes | −0.006 (0.72) | −0.027 (0.11) |
| ECGF linear + distance penalty | 0.530 | 0.449 | 0.444 | 0.348 / 0.316 | 0.247 | yes | −0.018 (0.28) | −0.039 (0.06) |
| ECGF linear | 0.528 | 0.442 | 0.434 | 0.363 / 0.291 | 0.214 | yes | −0.019 (0.09) | −0.040 (0.02) |
| Header + clinical LR | 0.501 | 0.440 | 0.424 | 0.465 / 0.286 | 0.243 | yes | −0.046 (0.08) | −0.067 (0.01) |
| Header + ResNet-18 image (original default; r0 only) | 0.495 | 0.444 | 0.389 | 0.670 / 0.241 | 0.141 | no | −0.054 (0.01) | −0.076 (0.01) |
| Header numeric LR | 0.489 | 0.414 | 0.393 | 0.440 / 0.254 | 0.162 | yes | −0.058 (0.00) | −0.079 (0.00) |
| ResNet-18 image (r0 only) | 0.471 | 0.425 | 0.363 | 0.681 / 0.231 | 0.135 | no | −0.078 (0.03) | −0.100 (0.02) |

Confusion matrices (mean counts over repeats) are in `phase5_baseline_results.csv` (`confusion_mean`).

## Interpretation
- **Frozen baselines for Phase 5:**
  - **C1 (primary): vote ensemble, accuracy 0.547**, the best balanced and admissible model;
  - **C2: calibrated MLP, accuracy 0.568**, the best accuracy but not admissible;
  - the best admissible accuracy is **0.560** (vote + EF-regression member).
- **The accuracy gains of calibration come from Mild.** Calibration raises accuracy by +0.021 to +0.025 over C1 by
  predicting Normal more often. Mild recall falls to 0.09–0.11, and Severe recall also drops (accuracy objective
  0.256).
  - Under the pre-registered recall floor the gain disappears (0.549).
  - Even the uncalibrated MLP fusion is just below the floor (Mild recall 0.141).
- **No baseline is significantly more accurate than C1:** the smallest p is 0.06. The distance to the **65 % target is
  8–10 points**. Decision gate: the best admissible model is **below 58 %**, so the plan calls for investigating
  feature and data limitations (5B, 5C).

## Limitations
- **Population:** development-CV only. The locked test is exposed (Phase 1) and not used. Grouping is by name key
  (`REPOSITORY_AUDIT.md` §6).
- **Phase 1–3 imports:** their hyper-parameters were chosen in those phases on the same development folds, so their
  numbers carry the usual optimism of re-using a CV design.
- **Comparisons:** the image models have only repeat 0, so their comparisons use 5 cells.
