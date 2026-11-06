# Phase 5 plan and pre-registration (fixed before any Phase 5 model run; SHA-256 in `configs/PLAN.sha256`)

## 1. Objective
Improve **four-class** LVEF classification from the 12-lead ECG printout (Normal / Mild / Moderate / Severe, the
hospital's sex-specific thresholds, unchanged). **Primary target: ≥ 65 % overall accuracy.** This is an
experimental objective, not a promised result.

## 2. Data and validation (unchanged from Phases 2–4)
- **Data:** development set D, 837 ECGs / 794 name-based patient keys. The locked test (210 ECGs) is **never
  predicted, scored or used**.
- **Folds:**
  - outer: patient-grouped StratifiedGroupKFold(5), repeats 0–2 (`phase_2/configs/folds_D.csv`);
  - inner: inner-train / inner-val per outer fold, as in Phases 2–4.
- **Nested inner CV** for anything fitted on predictions (stacking, decision rules, calibration). Within each outer
  training portion:
  - a patient-grouped 4-fold inner CV with `random_state = 1000 + 10·r + k`;
  - every inner model uses its own 80/20 grouped split, `random_state = 2000 + 10·k + j` (as in
    `phase_2/nested_calibration.py`);
  - inner out-of-fold predictions are the **only** data a meta-learner or decision rule sees.
- **Fitting inside folds:** all preprocessing, imputation, standardisation, feature selection and hyper-parameter
  choice are fitted on training rows of the relevant fold.
- **Echo EF is never a model input.** It is used only to *analyse* errors.

## 3. Metrics, comparators and the admissibility rule
- **Primary metric:** overall four-class accuracy on all 837 ECGs, pooled outer-fold predictions, mean over repeats
  0–2. Seeds are combined within a repeat: probability mean, argmax or the model's decision rule.
- **Full suite (always reported):**
  - per-class precision, recall and F1;
  - balanced accuracy, macro-F1, QWK, macro AUROC;
  - confusion matrix;
  - **Severe recall and precision**.
- **Admissibility (pre-registered):** a model may be selected as "best" only if **every class recall ≥ 0.15** (mean
  over repeats).
  - This operationalises "no class made effectively undetectable".
  - A sensitivity analysis at 0.10 and 0.20 is reported.
  - Non-admissible models are reported but not selected.
- **Comparators:**
  - **C1 (primary):** the Phase 3 vote ensemble (accuracy 0.547; admissible, min recall 0.22);
  - **C2:** the calibrated MLP fusion (56.8 %, reproduced in 5A with stored predictions; Mild recall about 0.11, so
    *not* admissible). It is reported as the accuracy reference.
- **Statistics:**
  - **primary:** corrected repeated-CV t-test (Bouckaert–Frank, n_train / n_test = 535 / 167) on the 15 matched
    (repeat, fold) cells, candidate − comparator;
  - **secondary:** patient-cluster paired bootstrap (2,000 resamples, repeat 0) and exact McNemar;
  - **Holm** over the planned confirmatory comparisons of each experiment family (§5), and globally over all
    confirmatory candidates for the final selection.
- **Genuine improvement:** accuracy higher than C1 (and, separately, than C2) with Holm-adjusted p < 0.05, the same
  sign in all 3 repeats, and the model admissible.

## 4. Decision gates (development-CV accuracy of the best admissible model)

| Accuracy | Action |
|---|---|
| < 58 % | investigate feature and data limitations |
| 58–60 % | assess complementary models |
| 60–62 % | refine promising experiments |
| 62–65 % | consolidate and test robustness |
| ≥ 65 % | nominate a candidate for final (authorised) validation |

## 5. Experiments (deterministic IDs; registry in `registry.py`)

| ID | Hypothesis | Baseline / comparison |
|---|---|---|
| **5A** `P5A-import-*` | Existing models re-evaluated accuracy-first with CIs (no training) | – |
| **5A** `P5A-calmlp-{protected,accuracy}` | Reproduces 56.8 % with stored predictions (nested prior calibration) | reproduction check: within ±0.005 of 0.571 / 0.569 / 0.565 |
| **5B** error analysis | Errors concentrate at the Normal/Mild and Mild/Moderate boundaries and in low-quality recordings | descriptive |
| **5C** `P5C-lr-<family>`, `P5C-lr-hdr+<family>` | A validated engineered-feature family adds accuracy beyond header numbers | LR on header numeric; ECGF linear |
| **E1** `P5D-E1-mlp-<family>` | Adding a feature family to the MLP fusion raises accuracy | MLP fusion (same seeds and folds) |
| **E2** `P5D-E2-{xgb,lgbm}-<set>` | Gradient boosting on engineered features beats linear / MLP | MLP fusion; header + clinical LR |
| **E3** `P5D-E3-stack-*` | A nested-trained regularised stacker beats the equal-weight vote | vote ensemble (C1) |
| **E4** `P5D-E4-hier-*` | A two-stage (Normal vs abnormal → Mild / Moderate / Severe) model beats the flat 4-class model | MLP fusion |
| **E5** `P5D-E5-<cand>-<rule>` | A nested accuracy-max decision rule **under the recall floor** raises accuracy while staying admissible | the same model with argmax |
| **E6** `P5D-E6-*` | Signal-quality-aware training or inputs improve accuracy | MLP fusion |
| **E7** `P5D-E7-<strategy>` | One training change (focal, smoothing, mixup, cosine LR, accuracy-based early stopping, waveform augmentation) raises accuracy | MLP fusion; Phase 2 runs re-used for weighting, sampler, dropout and width |

- **One change at a time:** each experiment changes one factor relative to its stated baseline. Combinations are only
  built from components that were individually positive, and are labelled as such.
- **Seeds and repeats:** 42 / 43 / 44 × repeats 0–2 for stochastic models; deterministic models use repeats 0–2.

## 6. Resumability and immutability
- **Records:** every run writes its configuration, config hash, seed, repeat and aggregate metrics
  (`results/experiments/<run_id>.json`), plus git-ignored per-ECG predictions (`data/private/runs/<run_id>/`).
- **DONE** = both files exist and the prediction file hash matches the aggregate.
- **INTERRUPTED:** a run marker (`results/running/<run_id>.json`) without a live process. The resume command is
  printed by `python -m research_experiments.phase_5.resume5`.
- **No overwrites:** completed runs are never overwritten. A correction gets a new version suffix (`-v2`) and the
  original is kept.
- **Privacy:** nothing keyed by an ECG id, timestamp or patient hash is committed (`p5common.privacy_scan` before
  every push).
