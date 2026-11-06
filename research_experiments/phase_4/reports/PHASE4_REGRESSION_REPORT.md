# Phase 4D: continuous EF regression

Pre-registered in `phase_4/configs/PREREG_PHASE4.md`, committed (`25e1d1f`) before the first run.
- **Code:** `phase_4/regression.py`.
- **Results:** `results/runs/reg_*.json`, `results/phase4_benchmark.csv` and `results/phase4_tables.md`.
- **Figures:** `plots/regression_pred_vs_echo_hexbin.png`, `plots/regression_binned_residuals.png`.

## 1. Design
- **Data:** development set D, 837 ECGs / 794 patients. The locked test set (210 ECGs) was **not predicted and not
  scored**.
- **Target:** EF (Teichholz) of the **nearest-in-time linked echo**, chosen without looking at the folder label
  (ties → mean).
  - It is *not* a verified measurement: the link is a fuzzy name match (`PHASE4_DATA_AUDIT.md` §4).
  - No midpoint EF values were assigned to categorical labels.
- **Training cohorts** (training and inner-val selection only):

  | Cohort | Definition | ECGs / patients |
  |---|---|---|
  | A | all | 837 / 794 |
  | C | nearest echo ≤ 14 d | 550 / 530 |
  | G | high-confidence: exact name, ≤ 30 d, sex agrees, no conflicting link ≤ 30 d | 357 / 345 |

- **Evaluation:** predictions are made for **every** outer-fold ECG, so all models are scored on the same 837 ECGs.
- **Validation:**
  - patient-grouped 5-fold outer CV, repeats 0–2, identical folds to Phases 2–3 (`phase_2/configs/folds_D.csv`);
  - model fitting on inner-train;
  - α, L2, epoch, recalibration, σ and cut-points are all chosen on inner-val;
  - the outer fold is never used for any choice.
- **Models:**
  - ridge on header numeric (16), ECGFounder f6 (1,024), and header ‖ ECGF;
  - MLP regression (Phase 2 fusion body, Huber loss) on header ‖ ECGF, 3 seeds.
  - In total, 54 runs × 4 EF → class decoders.
- **Decoders:**
  - `tuned` (primary): cut-points chosen on inner-val S3;
  - `clin`: the recovered clinical rule (M > 52 / F ≥ 54 Normal, ≥ 41 Mild, ≥ 31 Moderate) applied to the
    predicted EF;
  - `recal`: inner-val linear recalibration, then the clinical rule;
  - `gauss`: argmax of Gaussian class probabilities.
- **Leakage checks:**
  - the EF target is label-agnostic;
  - sex for the thresholds comes from the PDF header, which agrees with the echo sheet in 1,047 / 1,047 ECGs;
  - block standardisation and header medians are fitted on inner-train rows only;
  - folds are patient-disjoint (asserted in `protocol.py`).

## 2. Regression quality [VALID]
Means over 3 CV repeats (seeds combined). The reference no-information model predicts the training-fold mean:
MAE 15.6, RMSE 17.6 (EF SD 17.6 in A, C and G).

| Features / model | Train cohort | MAE (A) | RMSE (A) | R² (A) | MAE (C) | R² (C) | MAE (G) | R² (G) | Pearson / Spearman (G) |
|---|---|---|---|---|---|---|---|---|---|
| Header only, ridge | A | 12.50 | 15.17 | 0.255 | 12.26 | 0.292 | 12.35 | 0.290 | 0.54 / 0.53 |
| ECGF only, ridge | A | 10.65 | 13.43 | 0.416 | 9.91 | 0.507 | 9.91 | 0.514 | 0.73 / 0.71 |
| Header ‖ ECGF, ridge | A | 10.70 | 13.50 | 0.410 | 9.97 | 0.502 | 9.87 | 0.514 | 0.72 / 0.71 |
| Header ‖ ECGF, ridge | C | 10.49 | 13.57 | 0.403 | 9.72 | 0.512 | 9.56 | 0.536 | 0.73 / 0.72 |
| Header ‖ ECGF, ridge | G | 10.66 | 13.84 | 0.380 | 9.88 | 0.491 | 9.83 | 0.504 | 0.71 / 0.71 |
| **Header ‖ ECGF, MLP** | A | **10.37** | **13.31** | **0.426** | 9.60 | 0.527 | 9.51 | 0.541 | 0.74 / 0.72 |
| Header ‖ ECGF, MLP | C | 10.32 | 13.50 | 0.410 | **9.47** | **0.529** | **9.37** | **0.553** | 0.74 / 0.73 |
| Header ‖ ECGF, MLP | G | 10.44 | 13.70 | 0.392 | 9.60 | 0.511 | 9.69 | 0.516 | 0.72 / 0.71 |

- **What carries the signal:** the ECG waveform (ECGFounder) carries most of the EF information. Header numbers
  alone explain R² 0.26; ECGFounder explains 0.42; adding the header does not help ridge.
- **Evaluation-set effect:** every model scores better against the cleaner targets of C and G (R² 0.50–0.55)
  than against all of A (0.38–0.43). This is the same model evaluated on a more reliable target, **not** a better
  model.
- **Training cohort:** training on the cleaner cohorts (C, G) does **not** improve regression on any target. C is
  equivalent to A; G is slightly worse, having 40 % of the training data.
  - Interpretation [HYPOTHESIS]: at this sample size, target noise from fuzzy or distant links costs less than
    losing training examples.
- **Calibration:**
  - in the regression sense the predictions are well calibrated: slope of echo EF on predicted EF ≈ 0.94–1.01
    for cohort A;
  - they are strongly **shrunk toward the mean**: slope of predicted on echo EF ≈ 0.42–0.46;
  - low EF is over-predicted and high EF under-predicted (`regression_binned_residuals.png`). This is expected
    from an MSE-optimal predictor with R² ≈ 0.4.
- **Error near the class boundaries:** MAE is 9–10 points whether the echo EF lies within 2 points of a
  boundary or 5–10 points away. However, the clinical-threshold class disagrees with the echo-derived class for
  67–69 % of ECGs within 5 points of a boundary, against 15–17 % of ECGs ≥ 10 points away
  (`mild_analysis.json → regression_error_near_boundaries`).

## 3. Regression-assisted classification vs direct classification [VALID]
All 837 ECGs, folder labels, mean over repeats. Corrected repeated-CV t-test on 15 matched cells.

| Model | Accuracy | Balanced acc. | Macro-F1 | F1 Mild | F1 Severe | S3 | ΔS3 vs vote (p) |
|---|---|---|---|---|---|---|---|
| Vote ensemble (benchmark) | 0.547 | 0.467 | 0.455 | 0.255 | 0.362 | 0.490 | – |
| MLP fusion (direct 4-class, same features) | 0.544 | 0.466 | 0.440 | 0.186 | 0.373 | 0.483 | −0.007 (0.65) |
| Best `tuned`: MLP regression, C | 0.546 | 0.435 | 0.432 | 0.248 | 0.267 | 0.471 | −0.022 (0.29) |
| MLP regression, A, `tuned` | 0.538 | 0.422 | 0.418 | 0.256 | 0.245 | 0.460 | −0.032 (0.27) |
| MLP regression, C, `clin` (clinical thresholds) | 0.558 | 0.416 | 0.418 | 0.287 | 0.207 | 0.464 | −0.026 (0.22) |
| Ridge ECGF, A, `clin` | 0.547 | 0.397 | 0.394 | **0.309** | 0.149 | 0.446 | −0.044 (0.05) |
| Header only, ridge, A, `tuned` | 0.506 | 0.414 | 0.405 | 0.205 | 0.310 | 0.441 | −0.053 (0.12) |

**Findings:**
1. **No regression-based classifier beats direct classification.**
   - Against the vote ensemble, all 12 primary-decoder (`tuned`) models have lower S3; none is significant after
     Holm (smallest Holm p 0.25).
   - Against the matched direct classifier (MLP fusion), ΔS3 ranges from −0.016 to −0.068
     (`results/phase4_vs_controls.md`).
2. **The clinical-threshold conversion trades Severe for Mild.**
   - Shrinkage means a predicted EF < 31 is rare, so **Severe recall drops to 0.10–0.12** (vote: 0.43) and F1
     Severe to 0.15–0.21.
   - Predictions pile up in 41–55, so Mild recall rises to 0.29–0.31 and F1 Mild to 0.28–0.31.
   - This does not over-predict Mild: about 1.1 × the true Mild count is predicted Mild.
   - Clinically it is the wrong trade, because missing severe LV dysfunction is the costliest error.
3. **The `gauss` decoder is unusable for 4 classes:** argmax almost never selects Moderate (F1 0.07–0.22). It is
   kept only for its class probabilities (AUROC and fusion).
4. **Accuracy is near the vote ensemble** with clinical thresholds (0.547–0.558) because Normal is predicted well
   (F1 0.78). Balanced accuracy is lower by 0.05–0.07.

## 4. Error budget: what EF-prediction quality would the targets need? [ESTIMATE, exploratory]
- **Simulation:** suppose the predicted EF were the nearest-echo EF plus *unbiased* Gaussian error of SD σ
  (classical error), decoded with the clinical rule (200 draws; `mild_analysis.json → error_budget_exploratory`).
  Such a predictor correlates with the echo EF at r = SD_EF / √(SD_EF² + σ²), with SD_EF = 17.6.
- **Use r, not RMSE, to compare.** Our models make the opposite, *shrunk* type of error: well calibrated, slope
  0.42–0.46, so RMSE is not comparable between the two error structures.

| σ (EF points) | 0 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| implied r (R²) | 1.00 (1.00) | 0.91 (0.83) | 0.87 (0.76) | 0.83 (0.68) | 0.78 (0.61) | 0.74 (0.55) | 0.70 (0.49) | 0.66 (0.44) |
| Accuracy | 0.934 | 0.695 | 0.645 | 0.608 | 0.576 | 0.551 | 0.525 | 0.507 |
| Balanced accuracy | 0.911 | 0.625 | 0.570 | 0.532 | 0.502 | 0.476 | 0.453 | 0.436 |
| Macro-F1 | 0.924 | 0.616 | 0.555 | 0.513 | 0.478 | 0.450 | 0.423 | 0.403 |
| Mild F1 | 0.935 | 0.516 | 0.433 | 0.380 | 0.336 | 0.305 | 0.276 | 0.259 |

- **Observed:** Pearson r of the best EF estimates with the nearest-echo EF is **0.65–0.66 on all 837 ECGs**
  (R² 0.43) and 0.74–0.75 on G. That corresponds to σ ≈ 20 (all) and σ ≈ 16 (G).
- **Consistency check:**
  - the simulation at σ ≈ 20 predicts accuracy 0.51 / balanced 0.44. The EF-regression decoders reach 0.53–0.56 /
    0.40–0.44;
  - at σ ≈ 16 it predicts 0.55 on G. The models reach 0.57–0.60 on G.
- **Implication:** the owner's targets (accuracy ≥ 60 %, balanced ≥ 50 %, macro-F1 ≥ 0.48, Mild F1 ≥ 0.35) each need
  an EF estimate with **r ≈ 0.78–0.82** against the echo EF (σ ≲ 12–14, R² ≈ 0.61–0.68). That is far above the
  0.65 achieved.
  - The gap is in **information**, not in the decoder: a better threshold rule cannot create it.
  - The direct 4-class classifiers (vote ensemble: accuracy 0.547, balanced 0.467) already perform slightly
    *above* what their EF-level information would predict.
- σ = 0 shows the ceiling imposed by labels: **93.4 %** of folder labels agree with the class of the nearest echo.

## 5. Limitations
- **Target reliability:** the EF target comes from name-matched echoes without a patient identifier; G is the
  best available proxy for reliable links.
- **Measurement uncertainty:** echo measurement uncertainty cannot be quantified (18 repeat pairs only). The noise
  floor of the target is therefore unknown, and R² on G may be bounded by measurement error, not by the ECG.
- **Model scope:** ridge has one deterministic run per repeat; the MLP has 3 seeds. Only frozen ECGFounder
  features were used; fine-tuning was equal to frozen in Phase 3.
- **Test set:** the locked test was not used. Development-CV estimates of the best of many models are optimistic.
