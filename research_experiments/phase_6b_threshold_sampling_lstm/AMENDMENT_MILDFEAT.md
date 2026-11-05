# Phase 6B exploratory amendment: MILDFEAT (image-only clinician features for Mild)

**Date:** 2026-11-05. Written after the label-free feature extraction and before any model run. Owner request: "Figure
out the best way to classify Mild ... find features from the images only ... try to extract more ... think like a doc".
Code: `mildfeat.py` (features), `mildmodels.py` (models).

## 1. Image-only inputs
- **Allowed:** only the ECG drawing digitised from the printed PDF (`data/signals.npz`).
- **Not allowed:** header text, printed machine measurements, age or sex. The printed machine values are used only to
  validate and calibrate the measurements (label-free; `results/mildfeat_validation.json`).
- **CLIN:** the clinician feature set of `mildfeat.py`, built in four steps:
  1. The 10-s rhythm strip is simultaneous with all four printed columns, so its R peaks time-stamp the beats in every
     column. Earlier phases did not use this.
  2. Each lead's median beat is aligned to the rhythm-II R peak, which gives a synthesised simultaneous median 12-lead
     beat.
  3. From it come global QRS onset / offset, T end and the P wave.
  4. Then the features:
     - rhythm;
     - intervals: QRS, PR, P, QT, QTc, JT, Tpeak–Tend;
     - conduction: R-peak time V1 / V5 / V6, LBBB / RBBB / IVCD-like patterns;
     - scar: pathological Q and fragmented QRS per territory, poor R progression, transition;
     - voltage: Sokolow-Lyon, Cornell, Peguero, QRS sum, low voltage;
     - repolarisation: ST J+60, T amplitude / area per lead and territory, T inversion, flat T, strain, T/R;
     - atrial: P amplitude, P-terminal force V1, P axis;
     - vector (Kors VCG): spatial and frontal QRS-T angle, ventricular gradient, QRS / T vector magnitudes, axes.

  Signal-quality counts (valid leads, beats per column, number of beats) are not model inputs. Every feature is used:
  there is no outcome-based feature selection.
- **ECGF:** the ECGFounder f6 embedding of the same digitised waveforms, as used in the IMG-ONLY amendment.

## 2. Models
All models use Phase 2 folds, 3 repeats and the ORIGINAL labels on all 837 ECGs. Imputation (in-fold median) and
standardisation are fitted on the fold's training rows only. Selection uses inner-val with the S3 score.
- **CLIN-LR:** multinomial logistic regression on CLIN, class-balanced, C from the Phase 5 grid.
- **CLIN-ORD:** the ordinal "doctor" model. LV dysfunction is graded on one scale (Normal < Mild < Moderate < Severe).
  - Three class-balanced binary logistic regressions estimate P(y > k) (Frank & Hall), with one C chosen on inner-val.
  - Class probabilities are the successive differences, clipped at 0 and renormalised.
- **CLIN-GBM:** LightGBM on CLIN, with fixed settings:
  - 7 leaves, learning rate 0.03, at most 400 rounds;
  - early stopping on inner-val (50 rounds);
  - subsample 0.8, colsample 0.5, min child 15, class-balanced.
- **IMG-MLP+CLIN:** the Phase 2 MLP fusion recipe (hidden 64, dropout 0.5, class-balanced CE, L2 on inner-val) on the
  block-standardised [CLIN ‖ ECGF], 3 seeds. This is the header block replaced by CLIN.
- **IMG-vote+CLIN:** the image-only analogue of C1. It is an equal-weight mean of:
  - ECGF-MLP (3 seeds);
  - ECGF linear (distance-penalised CE);
  - CLIN-LR (replacing the header + clinical LR).
- **IMG-C3+CLIN:** IMG-vote+CLIN plus the nested class-prior rule (protected objective, Phase 5 nested protocol). It
  uses nested inner OOF of all three members.
- **Baselines, recomputed here from the same inputs:**
  - ECGF-MLP and IMG-vote (ECGF-MLP + ECGF linear);
  - IMG-C3 (IMG-vote + nested rule). Reproducibility check against the cloud aggregate: 0.559.
  - Reference with header (not image-only): C3, 0.561.

## 3. Evaluation (development CV; the locked test set is not used)
- **Four-class metrics:** accuracy (corrected repeated-CV CI), balanced accuracy, macro-F1, per-class recall /
  precision / F1, minimum recall, admissibility (every recall ≥ 0.15).
- **Mild-specific metrics:**
  - Mild recall, precision and F1;
  - AUROC Normal-vs-Mild among true Normal + Mild ECGs, score P(Mild) / (P(Normal) + P(Mild));
  - AUROC Mild-vs-Moderate, score P(Moderate) / (P(Mild) + P(Moderate));
  - AUROC reduced-vs-Normal, score 1 − P(Normal).
- **Paired comparisons:** corrected repeated-CV t-test on accuracy, macro-F1, Mild F1 and Mild recall.
  - Comparisons: IMG-vote+CLIN vs IMG-vote; IMG-C3+CLIN vs IMG-C3; IMG-MLP+CLIN vs ECGF-MLP; CLIN-ORD vs CLIN-LR.
  - Holm correction over all comparisons.
  - A genuine improvement needs Holm p < 0.05, the same sign in all 3 repeats, and admissibility.
- **Descriptive "doctor's table" (does not feed any model):** per CLIN feature:
  - the class medians;
  - univariate AUROC Normal-vs-Mild and Mild-vs-Moderate, with patient-cluster bootstrap 95 % CI (1,000 resamples);
  - a Benjamini–Hochberg flag.
- **Status:** EXPLORATORY and reported separately.
