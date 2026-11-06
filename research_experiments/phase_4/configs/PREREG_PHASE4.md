# Phase 4D–4H pre-registration (written before any Phase 4 model was trained)

Committed to git before the first run; its SHA-256 is recorded in `PREREG_PHASE4.sha256`.

## Question
Can continuous echo EF supervision and better label quality improve 4-class LVEF prediction (Normal / Mild /
Moderate / Severe) from the 12-lead ECG PDF, beyond the Phase 3 benchmarks?
- **Benchmarks:**
  - vote ensemble: accuracy 0.547, balanced accuracy 0.467, macro-F1 0.455, S3 0.490;
  - calibrated MLP fusion: accuracy 0.568 (its per-ECG predictions were not stored, so it is reported as a
    reference value only).

## Data and validation (unchanged from Phases 2–3)
- **Development set D:** 837 ECGs / 794 patients. Locked test T (210 ECGs) is **not predicted and not scored**
  in Phase 4.
- **Folds:**
  - outer 5-fold patient-grouped CV from `phase_2/configs/folds_D.csv`, repeats 0, 1 and 2;
  - inner-train / inner-val split as defined there;
  - every model is fitted on inner-train, and hyper-parameters, early stopping and decision rules are chosen
    on inner-val only.
- **Inputs:** the Phase 2 features only, header numeric (16) and ECGFounder "aligned" f6 (1,024),
  block-standardised when concatenated.
- **EF target:** EF of the nearest-in-time linked echo, chosen label-agnostically (ties → mean;
  `phase_4/cohorts.py`).
- **EF → class:** the exact recovered rule. Sex comes from the PDF header (`male`), which agrees with the echo
  sheet for all 1,047 ECGs.
- **Training cohorts** (cohorts restrict *training and inner-val selection only*):
  - A: all;
  - C: nearest echo ≤ 14 days;
  - G: high confidence.

## Primary evaluation (identical for every model, including the benchmark)
- **Set and labels:** all 837 development ECGs, **folder labels**, outer-fold predictions.
- **Score:** mean of pooled-OOF metrics over repeats 0–2, seed-averaged probabilities within a repeat.
- **Primary metric:** S3 = (accuracy + balanced accuracy + macro-F1) / 3, as pre-registered in Phase 2.
- **Comparisons:**
  - each candidate − vote ensemble on the 15 matched (repeat, fold) cells, corrected repeated-CV t-test
    (Bouckaert–Frank, n_train / n_test = 535 / 167);
  - patient-cluster paired bootstrap on repeat 0;
  - Holm correction across the candidates in Section "Model family" within each metric.
- **"Genuine improvement":** all-D folder-label S3 (or accuracy) better than the vote ensemble with Holm-adjusted
  p < 0.05 **and** the same sign in all three repeats.
- **Not counted as model improvements**, but reported as evaluation-set / label-quality effects:
  - gains that appear only on G;
  - gains that appear only against EF-derived labels;
  - gains that appear only for a single repeat or seed.

## Secondary evaluations (reported for every model, including the vote ensemble, on the same ECGs)
- **Subsets:**
  - G subset vs folder labels and vs EF-derived class;
  - non-G subset vs folder labels.
- **Classification metrics:** accuracy, balanced accuracy, macro-F1, per-class F1, QWK and macro AUROC, with
  patient-cluster bootstrap 95 % CIs.
- **Regression metrics:** MAE, RMSE, R², Pearson and Spearman, against the nearest-echo EF on A, C and G.
- **Clinically standard binary endpoints:**
  - EF < 41 (Moderate + Severe);
  - EF < 31 (Severe);
  - AUROC.

## Model family (complete list; anything else is labelled exploratory)

### 4D Regression
- **Ridge:** features {header, ECGF, header ‖ ECGF} × training cohorts {A, C, G}. α is chosen by inner-val MAE
  (in-cohort).
- **MLP regression:** header ‖ ECGF, 1 hidden layer of 64 units with GELU and dropout 0.5 (Phase 2 MLP fusion
  body), Huber loss on z-scored EF. L2 and epoch are chosen by inner-val MAE. Cohorts {A, C, G}; seeds 42 / 43 / 44.
- **EF → class decoders:**
  - **`tuned` (PRIMARY):** three cut-points on predicted EF, chosen on inner-val by S3 against folder labels;
    the female Normal cut-point = male + 2, as in the clinical rule;
  - `clin`: the clinical rule on the predicted EF;
  - `recal`: inner-val linear recalibration, then the clinical rule;
  - `gauss`: argmax of Gaussian class probabilities, with σ = the inner-val residual SD after recalibration.
    These probabilities are also used for AUROC and fusion.

### 4E Multitask
- **Architecture:** the Phase 2 MLP fusion body plus two heads:
  - 4-class head, class-balanced CE;
  - EF head, Huber on z-scored EF, masked to the EF cohort.
- **Fixed λ:** λ ∈ {0, 0.3, 1, 3} with EF cohort A. λ = 0 is the control and must reproduce the Phase 2 MLP
  fusion (implementation check).
- **Selected λ:** `mt_sel`, λ picked per fold by inner-val S3 from the four fixed-λ runs.
- **Other variants:**
  - λ = 1 with EF cohorts C and G;
  - uncertainty weighting (Kendall et al. 2018) with EF cohort A.
- **Seeds and repeats:** 42 / 43 / 44 × repeats 0–2.

### 4F Label quality
MLP fusion (Phase 2 configuration, `p2heads`), seeds 42 / 43 / 44 × repeats 0–2:
- `lq_trainG`, `lq_trainC`: train and select on cohort G / C only;
- `lq_dropdisagree`: train and select without the ECGs whose folder label ≠ EF-derived class;
- `lq_relabel`: train and select with the EF-derived class instead of the folder label.

Evaluation is always on all D (folder labels), plus the secondary sets.

### 4F Regression-assisted ensemble
- **`vote+reg_*`:** the vote ensemble with one extra equal-weight member, the Gaussian class probabilities of a
  regression model (4D ridge A and MLP A, header ‖ ECGF), or the class head of the multitask model.
- No weights are tuned.

### 4G Analyses (no model selection)
- **G cohort characterisation:**
  - composition;
  - agreement 95.8 % and why it is high by construction;
  - performance of each model on G vs non-G, adjusted for class mix.
- **Mild study:**
  - Mild AUROC: one-vs-rest, vs Normal, vs Moderate;
  - errors by distance of the nearest-echo EF to 41 / 52 / 54 and by ECG–echo interval.
- **Repeat-echo variability:** no measurement-uncertainty claim from the 18 repeat pairs.

## Privacy
- Per-ECG outputs (anything keyed by `ecg_id`, EF predictions) go to `phase_4/data/private/runs/`, which is
  git-ignored.
- Committed outputs contain aggregates only, with no identifiers, dates or per-patient values.
