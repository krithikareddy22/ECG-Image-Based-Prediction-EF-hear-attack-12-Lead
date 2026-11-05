# Phase 4G: Mild LVEF investigation

- **Code:** `phase_4/analysis4.py`.
- **Results:** `results/mild_analysis.json`, `results/phase4_benchmark.csv`.
- **Figure:** `plots/confusion_matrices_key_models.png`.
- **Data:** 837 development ECGs / 794 patients, of which **185 are folder-Mild**. Outer-fold predictions, mean
  over CV repeats 0–2.
- **Privacy:** cells with < 5 ECGs are suppressed.

## 1. The Mild class itself [VALID]
- **EF distribution:** among the 185 folder-Mild ECGs, the nearest-echo EF has quartiles 43.4 / 45.8 / 49.0 %. The
  band is 41 to 52 (male) or 53.9 (female), only 11–13 points wide.
- **Near the boundaries:** **36** lie within 2 points of 41, and **15** within 2 points of the upper boundary
  (52 / 54). That is 51 / 185 = 28 % within 2 points of a class boundary, against 9 / 384 = 2.3 % for Normal (Moderate
  29 %, Severe 14 %).
- **Outside the band:** 19 folder-Mild ECGs have a nearest echo outside the Mild band (9 below 41, 10 above the
  upper boundary). Their folder label came from a different linked echo. Per the protocol, this is **not** claimed
  to make those labels wrong.
- **EF measurement method:** Teichholz only. Differences between methods cannot contribute to variability here.

## 2. Where Mild errors go [VALID]
Mean counts of the 185 folder-Mild ECGs over the 3 repeats:

| Model | → Normal | → Mild | → Moderate | → Severe | Mild predicted (total) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Vote ensemble | 65.3 | 40.7 | 50.3 | 28.7 | 133 | 0.305 | 0.220 | 0.255 |
| MLP fusion | 71.3 | 26.0 | 52.7 | 35.0 | 93 | 0.278 | 0.141 | 0.187 |
| Ridge EF regression, tuned cut-points | 76.0 | 37.7 | 60.3 | 11.0 | 141 | 0.266 | 0.204 | 0.231 |
| Ridge EF regression, clinical thresholds | 90.3 | **55.7** | 36.3 | 2.7 | 205 | 0.271 | **0.301** | **0.285** |
| MLP EF regression, tuned | 72.0 | 44.0 | 55.0 | 14.0 | 158 | 0.281 | 0.238 | 0.258 |
| Multitask, λ selected | 70.7 | 30.7 | 49.7 | 34.0 | 105 | 0.294 | 0.166 | 0.212 |
| MLP fusion trained on EF-derived labels | 73.3 | 28.0 | 48.3 | 35.3 | 99 | 0.280 | 0.151 | 0.197 |
| Vote + ridge-EF member | 83.0 | 33.0 | 42.7 | 26.3 | 111 | 0.295 | 0.178 | 0.222 |

- **Error direction:** errors go **both ways**. For the vote ensemble, 35 % of folder-Mild ECGs are called Normal,
  27 % Moderate and 16 % Severe. Mild is not confused with only one neighbour.
- **Prediction rate:** no model predicts Mild excessively.
  - Clinical-threshold EF regression is the only model above the true count (205 predicted vs 185 true, 1.1×),
    and it gets there by moving Mild ECGs away from Severe.
  - The vote ensemble predicts Mild 0.72× as often as it occurs.
- **Precision:** stuck at 0.27–0.31 for every model. About 70 % of "Mild" calls are Normal or Moderate ECGs.

## 3. ECG characteristics of correctly vs incorrectly classified Mild ECGs [VALID, descriptive]
Vote ensemble; prediction = majority over the 3 repeats; header measurements are means.

| Folder-Mild ECGs predicted as | n | Age | Male | Heart rate | PR | QRS | QTcB | Mean echo EF | Folder = echo class |
|---|---|---|---|---|---|---|---|---|---|
| Mild (correct) | 45 | 61.4 | 0.51 | 84 | 147 | 94 | 452 | 47.1 | 0.93 |
| Normal | 69 | 54.5 | 0.49 | 82 | 151 | 91 | 437 | 49.7 | **0.80** |
| Moderate | 47 | 64.9 | 0.70 | 84 | 152 | 104 | 452 | 45.5 | 0.98 |
| Severe | 24 | 59.5 | 0.79 | 97 | 161 | **119** | 471 | 44.7 | 0.96 |

- **Mild ECGs called Normal** look electrically normal: younger, narrower QRS, shorter QTc. Their echo EF sits
  higher in the band (49.7). These are also where **folder–echo disagreement concentrates**: only 80 % agree, vs
  93–98 % in the other groups. Part of this error is plausibly label ambiguity at the upper boundary.
- **Mild ECGs called Moderate or Severe** carry conduction or repolarisation abnormalities: QRS 104–119 ms, more
  male, Severe-called ECGs faster with longer QTc. [HYPOTHESIS] These ECGs show changes typical of more advanced
  LV disease. The 4-class label then disagrees with the ECG phenotype, not necessarily with the echo.

## 4. Do recording intervals affect Mild classification? [VALID]
Mild recall by |ECG − echo| interval (vote ensemble; folder-Mild ECGs):

| Interval | ECGs | Vote recall | MLP regression (tuned) recall | Ridge regression (clinical thresholds) recall |
|---|---|---|---|---|
| Same day | 105 | 0.23 | 0.28 | 0.40 |
| 1–14 d | 17 | 0.14 | 0.12 | 0.08 |
| 15–90 d | 27 | 0.23 | 0.23 | 0.25 |
| > 90 d | 36 | 0.23 | 0.19 | 0.16 |

- **The interval has no consistent effect** on Mild recall for the benchmark. For all classes there is also no
  monotone trend: accuracy is 0.52 same-day, 0.73 for 1–7 d (n = 90), 0.51–0.55 beyond.
- The EF-regression models are better on **same-day** Mild ECGs (0.28 / 0.40 vs 0.23). [HYPOTHESIS] Continuous
  supervision helps most where the EF target is most contemporaneous. The bins are small; this was not tested
  formally.

## 5. Errors near the category boundaries [VALID]
All classes; accuracy against folder labels by distance of the nearest-echo EF to the closest clinical boundary
(31, 41, 52 M / 54 F):

| Distance to boundary | ECGs | Vote accuracy | Ridge regression accuracy | Multitask accuracy | Folder = echo class |
|---|---|---|---|---|---|
| < 2 points | 125 | 0.365 | 0.376 | 0.360 | 0.840 |
| 2–5 | 255 | 0.365 | 0.359 | 0.358 | 0.976 |
| 5–10 | 106 | 0.553 | 0.465 | 0.503 | 0.953 |
| ≥ 10 | 351 | 0.743 | 0.738 | 0.761 | 0.932 |

- **Accuracy halves within 5 EF points of a boundary.** This holds for every model, whether trained on classes or
  on EF.
- Mild recall is 0.30 within 2 points of a boundary, 0.19–0.22 at 2–10 points. Few Mild ECGs are ≥ 10 points from
  a boundary, by construction.
- **Interpretation:** with a predictive EF error of about 13 points (RMSE), an ECG within 5 points of a threshold
  is close to a coin flip between neighbouring classes. A borderline EF is **not** evidence of a wrong label;
  folder–echo agreement is 98 % at 2–5 points.

## 6. Mild discrimination metrics [VALID, with nominal significance only]
AUROC on all 837 ECGs, folder labels, mean over repeats. Pairwise score = P(Mild) / (P(Mild) + P(other)). Δ vs the
vote ensemble from the corrected repeated-CV t-test on 15 cells, **uncorrected for multiplicity**.

| Model | Mild OvR | Δ (p) | Mild vs Normal | Δ (p) | Mild vs Moderate | Δ (p) |
|---|---|---|---|---|---|---|
| Vote ensemble | 0.548 | – | 0.750 | – | 0.625 | – |
| MLP fusion | 0.547 | +0.002 (0.89) | 0.761 | +0.013 (0.18) | 0.630 | +0.004 (0.79) |
| Ridge EF regression (A) | 0.605 | +0.056 (0.08) | 0.757 | +0.005 (0.74) | **0.672** | +0.042 (0.25) |
| **MLP EF regression (A)** | **0.611** | **+0.062 (0.043)** | **0.769** | +0.020 (0.12) | 0.661 | +0.028 (0.42) |
| Multitask, λ selected | 0.553 | +0.006 (0.77) | 0.767 | +0.016 (0.19) | 0.637 | +0.010 (0.57) |
| MLP fusion on EF-derived labels | 0.564 | +0.013 (0.58) | 0.758 | +0.008 (0.49) | 0.630 | +0.004 (0.80) |
| Vote + ridge-EF member | 0.574 | +0.027 (0.023) | 0.759 | +0.010 (0.07) | 0.641 | +0.016 (0.13) |

- **Does continuous EF supervision improve Mild discrimination? Partly, and only in ranking.**
  - EF-regression scores rank Mild ECGs better (OvR AUROC +0.056 to +0.062) and separate Mild from Moderate
    better (+0.03 to +0.04).
  - Only two comparisons reach nominal p < 0.05, and neither survives a multiplicity correction over the ~30 Mild
    comparisons in this table.
  - Part of the OvR gain is structural. A band-shaped P(Mild) around a continuous EF estimate naturally ranks
    "middle" ECGs higher, which a 4-class softmax trained on noisy labels does not.
  - Status: **HYPOTHESIS** (claim P4-15).
- **The ranking gain does not become better Mild classification.**
  - No model reaches Mild F1 0.35; the best is 0.309 (ridge ECGF, clinical thresholds), at the cost of Severe
    recall.
  - Mild precision stays below 0.31 because Mild and its neighbours overlap heavily in the predicted-EF space.
- **Mild vs Normal is the easier pair** (AUROC 0.75–0.77) and **Mild vs Moderate the hardest** (0.62–0.67).

## 7. Conclusions
1. **Why Mild is hard:**
   - it is a narrow 11–13-point EF band;
   - 28 % of Mild ECGs lie within 2 points of a boundary;
   - the predictive EF error (RMSE ≈ 13) is wider than the band itself.

   No classifier or regressor here can be precise within the band.
2. Label disagreement concentrates in Mild ECGs called Normal (80 % folder–echo agreement). It contributes, but it
   is not the main cause.
3. EF regression gives the best Mild *ranking* and, with clinical thresholds, the best Mild F1 (0.31). It does so
   by giving up Severe detection, so it is **not recommended** as a clinical decision rule.
4. **Target status:** the Mild F1 target (≥ 0.35) was **not reached**.
   - Reaching it would need a materially better EF estimate: by the error-budget simulation
     (`PHASE4_REGRESSION_REPORT.md` §4), r ≈ 0.80 with the echo EF against the 0.65 observed. More informative inputs
     would also do it.
   - It should not be pursued by re-weighting toward Mild.
