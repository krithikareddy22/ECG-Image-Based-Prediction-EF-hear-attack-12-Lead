# Phase 5C: ECG feature engineering (and E1, feature-enhanced MLP)

- **Scripts:**
  - `features/extract.py`: extraction and validation;
  - `features/compare.py`: LR comparison;
  - `experiments/e1_feature_mlp.py`: E1.
- **Results:** `results/feature_validation.json`, `results/feature_comparison.csv`, `results/e1_feature_mlp.csv`.
- **Features:** stored git-ignored in `data/private/features/`, development rows only.

## 1. Feature families (label-free, deterministic, ECG-only)
Each family is computed from the digitised waveforms: 12 leads × 2.5 s, plus a 10-s lead-II rhythm strip at 500 Hz.

| Family | n | Content |
|---|---|---|
| `qrs` | 75 | per lead, from the median beat: R, S, Q amplitudes, QRS area, QRS width, fragmentation (extra slope changes); median width, total fragmentation, R-wave progression slope V1→V4 |
| `st` | 48 | per lead: ST level at J + 60 ms vs PR baseline, ST slope, T amplitude and T area |
| `hrv` | 10 | rhythm strip: mean RR, SDNN, RMSSD, pNN50, CV, min / max RR, beats, ectopy fraction, heart rate |
| `freq` | 60 | per lead Welch PSD: relative power 0.5–4 / 4–15 / 15–40 / 40–100 Hz, spectral entropy |
| `wavelet` | 84 | per lead db4 5-level DWT: relative energy per level + log total energy |
| `leadstat` | 72 | per lead: mean, SD, skewness, kurtosis, peak-to-peak, IQR |
| `sq` | 26 | signal quality (raw): flat-line and clipping fractions, baseline-wander and HF power fractions, kurtosis-SQI, amplitude (mean / min / max over leads); poor-lead count; quality score; rhythm-strip equivalents |
| `clin3` (existing) | 28 | Phase 3 clinical features: voltage, LVH criteria, R-wave progression, rhythm, T-wave |

**Leakage:** none of these features uses a label or the echo EF, and none is fitted. Imputation (in-fold median) and
standardisation are fitted on training rows only.

## 2. Verification of extraction [VALID]
- **Agreement with the printed machine measurements:**

  | Measure | Pearson r | Median absolute difference | Mean difference |
  |---|---|---|---|
  | Heart rate | 0.854 | 0.35 bpm | – |
  | Mean RR | 0.797 | 3.0 ms | – |
  | QRS width | 0.747 | – | −29 ms |

  - The QRS-width estimate (energy-threshold onset / offset) ranks ECGs consistently with the machine QRSD but is
    systematically about 29 ms short. That is a known property of threshold detectors, and the models use it only
    relatively.
- **Missingness:** < 0.2 % (median-beat features when no beat is found in a 2.5-s lead window). There are no constant
  features in v2.
- **Correction before any use (sq v2):** the v1 flat-line criterion flagged 98 % of leads as poor.
  - Vector-digitised traces are piecewise linear, so "flat" samples are a digitisation artefact.
  - The baseline-wander band was also below the spectral resolution.
  - v2 drops the flat-line criterion from the poor-lead flag and uses full-length spectra. No experiment used v1
    (`feature_validation.json → version_notes`).

## 3. Families tested individually: multinomial LR, header baseline recipe, 3 repeats [VALID]
- **Comparisons:** reference header-numeric LR, accuracy 0.489; Δ = candidate − reference; corrected repeated-CV
  t-test on 15 cells; Holm within each of the two sets.
- **Header + family:**

  | Header + family | Accuracy | Balanced acc. | Macro-F1 | Severe recall | Δacc (p; Holm) |
  |---|---|---|---|---|---|
  | **+ qrs** | **0.538** | 0.453 | 0.441 | 0.425 | **+0.049 (0.043; 0.38)** |
  | + all new families | 0.503 | 0.434 | 0.425 | 0.396 | +0.014 (0.67; 1.00) |
  | + clin3 + all new | 0.499 | 0.429 | 0.420 | 0.381 | +0.010 (0.67; 1.00) |
  | + leadstat | 0.499 | 0.426 | 0.417 | 0.388 | +0.010 (0.65; 1.00) |
  | + sq | 0.495 | 0.443 | 0.422 | **0.505** | +0.006 (0.75; 1.00) |
  | + freq | 0.483 | 0.416 | 0.406 | 0.396 | −0.005 (0.79; 1.00) |
  | + st | 0.482 | 0.413 | 0.399 | 0.410 | −0.007 (0.79; 1.00) |
  | + hrv | 0.473 | 0.404 | 0.380 | 0.458 | −0.016 (0.18; 1.00) |
  | + wavelet | 0.469 | 0.411 | 0.398 | 0.399 | −0.020 (0.40; 1.00) |

- **Families alone** (accuracy): ECGF linear (reference point) 0.528; all new 0.507; header + clinical 0.501; qrs 0.501;
  clin3 0.483; leadstat 0.475; st 0.442; freq 0.436; sq 0.436; wavelet 0.428; hrv 0.344.
- **Findings:**
  - **QRS morphology is the most informative engineered family.** With the header numbers it approaches the
    ECGFounder linear model (0.538 vs 0.528). This matches the error analysis (QRS width and R progression carry the
    reduced-EF signal).
  - It is the only nominally significant addition, and it does not survive Holm.
  - **More features do not help:** adding all families (≈ 375 features on about 535 training ECGs) dilutes the QRS
    signal (0.503).
  - HRV from a 10-s strip is close to uninformative for EF class.

## 4. E1: family added to the MLP fusion (header ‖ ECGFounder) [VALID]
- **Setup:** 3 seeds × 3 repeats per variant. Baseline: the identical MLP fusion (accuracy 0.544). Holm over the 8
  variants.

| MLP fusion + | Accuracy | Balanced acc. | Macro-F1 | Severe recall | Min class recall | Δacc (p) | Admissible |
|---|---|---|---|---|---|---|---|
| all new families | **0.554** | **0.477** | **0.455** | 0.483 | 0.164 | +0.010 (0.59) | yes |
| qrs | 0.546 | 0.461 | 0.438 | 0.454 | 0.135 | +0.003 (0.86) | no |
| leadstat | 0.546 | 0.467 | 0.439 | 0.491 | 0.144 | +0.002 (0.87) | no |
| sq | 0.545 | 0.465 | 0.441 | 0.472 | 0.142 | +0.001 (0.94) | no |
| wavelet | 0.544 | 0.465 | 0.444 | 0.443 | 0.160 | +0.001 (0.97) | yes |
| freq | 0.544 | 0.469 | 0.448 | 0.476 | 0.175 | +0.000 (0.98) | yes |
| clin3 (= Phase 3 3-way fusion, re-used) | 0.542 | 0.467 | 0.445 | 0.483 | 0.166 | −0.002 (0.90) | yes |
| hrv | 0.540 | 0.462 | 0.439 | 0.465 | 0.144 | −0.003 (0.78) | no |
| st | 0.535 | 0.454 | 0.435 | 0.440 | 0.159 | −0.009 (0.51) | yes |

**Findings:**
- **No engineered family improves the MLP fusion:** all |Δacc| ≤ 0.010, Holm p = 1.00. The ECGFounder embedding
  already contains the information that QRS morphology adds to the header numbers.
- The best variant (all families) is +0.010 accuracy, +0.011 balanced and +0.015 macro-F1 over its baseline. That is
  consistent in direction but not significant.

## 5. Limitations
- **Data:** features are computed from PDF-digitised traces (2.5 s per lead), so beat-level measurements rest on 2–4
  beats.
- **Timing:** without a common time base across the 4 × 3 layout, inter-lead timing features are not possible.
- **Quality indicators** are confounded with QRS morphology (`PHASE5_ERROR_ANALYSIS.md` §4).
