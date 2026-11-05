# Phase 7 plan (pre-registration)

**Date:** 2026-11-05 (laptop session, RTX 3060 6 GB). Written before any Phase 7 run. Owner brief (verbatim intent):
"genuine improvement in four-class LVEF classification, with reliable Mild detection".
- **Recurrent models are discontinued:** LSTM, BiLSTM, GRU, CNN-LSTM, ECGFounder-LSTM fusion and any other recurrent
  network. None is running (checked 2026-11-05: no Python process, GPU idle) and none will be launched.
- **Scope:** development set only (837 ECGs, 794 patient keys). The locked 210-ECG test cohort is not loaded, inspected
  or scored without explicit owner authorisation.

## 1. Primary task, targets and reference
- **Primary task:** the original four-class EF definitions (Normal M > 52 / F ≥ 54, Mild ≥ 41, Moderate 31–< 41,
  Severe < 31), scored on all 837 ECGs. The evaluation population is never changed. Alternative thresholds and subsets
  are sensitivity analyses only and are reported separately.
- **Owner research targets** (not permission to manipulate data or evaluation):

  | Metric | Target |
  |---|---|
  | Accuracy | ≥ 0.70 |
  | Mild recall | ≥ 0.60 |
  | Mild precision | ≥ 0.50 |
  | Mild F1 | ≥ 0.55 |
  | Macro-F1 | ≥ 0.60 |
  | Balanced accuracy | ≥ 0.60 |

- **Reference:** C3 (C1 vote + nested prior rule), accuracy 0.562 [0.519, 0.605], macro-F1 0.449, Mild P / R / F1
  about 0.31 / 0.18 / 0.22. For pilots, the reference is the same-recipe member, the Phase 2 MLP fusion on [header ‖
  ECGFounder f6], seed 42 (`P6A-mlpfusion-B1-v1__s42_r*`).
- **Prior evidence** (Phase 6B, MMCLEAN, MILDFIX, MILDFIX2):
  - the ECG–echo EF error is about 13 points SD (11.8 on clean links);
  - Mild-vs-Moderate AUROC is about 0.6;
  - a calibrated EF estimator with 13-point error caps four-class accuracy at about 0.63;
  - 70 % needs about 9 points.

  Phase 7 is therefore expected to improve Mild separation only modestly. The targets will be reported as met or not
  met, as found.

## 2. Validation protocol (unchanged from Phases 5–6B)
- **Folds:** Phase 2 patient-grouped 5-fold × 3 repeats (`fold_vec`), identical for every model; paired comparisons on
  the same folds.
- **Within each outer fold:** the Phase 6A split (`split_indices`) into inner-train and inner-val.
  - Inner-val alone selects the hyper-parameters and the early-stopping epoch.
  - Nested inner OOF (4-fold, `existing6b.inner_oof` splits) is used for any decision rule, stacking or fusion weight.
- **Seeds:** 42, 43, 44 for full runs; pilots use seed 42.
- **Statistics:**
  - corrected repeated-CV 95 % CI (Nadeau–Bengio);
  - paired corrected t-test vs C3 (ORIG) on all 837 ECGs (accuracy, macro-F1, Mild F1);
  - Holm correction over all Phase 7 confirmatory comparisons.
  - A genuine improvement needs Holm p < 0.05, the same sign in all 3 repeats, and admissibility (every class recall
    ≥ 0.15).
- **Metrics reported:** accuracy, balanced accuracy, macro-F1, per-class precision / recall / F1, confusion matrices.
- **No Mild boosting:** decision rules may not multiply Mild probabilities or force Mild predictions. The C3 prior rule
  (protected objective) is the only decision rule on fused models, as before.
- **Verified continuous EF:** where EF supervision is used, the target is the nearest-echo EF of clean-link ECGs only.
  These are the NF set of `AMENDMENT_MMCLEAN.md`: exact name match, echo ≤ 30 days (true dates), folder label =
  nearest-echo class, no other-class link, EF spread ≤ 10. That is 366 of 837. Other ECGs contribute their class label
  only.

## 3. Experiments (ordered by expected value / cost; each pilot gates its full run)
**Pilot gate (fixed now).** A pilot (seed 42, 3 repeats, all 837 ECGs) advances to the full run only if, against its
same-recipe reference on the same folds:
- accuracy ≥ +0.010, **or** Mild F1 ≥ +0.03;
- **and** macro-F1 is not lower by more than 0.010.

The gates are screens, not claims.

- **E1, ECGFounder representations (no new training of the backbone).** Features come from the existing ECGFounder
  cache (aligned input; stage 4: 400 × 79, stage 5: 1024 × 40 positions over 10 s). Each candidate replaces f6 in the
  MLP fusion recipe [header ‖ representation]:
  - **E1-msp:** stage-5 mean ‖ max ‖ std over time (3,072).
  - **E1-col:** stage-5 mean per printed column. The four 2.5-s windows are 10 positions each; they are concatenated
    (4,096), so each lead group keeps its own summary.
  - **E1-s45:** stage-4 mean ‖ f6 (1,424): multi-scale.
  - **E1-dx:** ECGFounder's 150 pre-trained diagnostic outputs (sigmoid), one forward pass of the frozen network on
    the same aligned input. These are interpretable labels (conduction, infarct, hypertrophy patterns).
  - Efficient fine-tuning is **not** repeated: Phase 6 A1 / A2 failed their screen (r 0.650 vs 0.652).
- **E2, continuous-EF supervision and Mild separation** (heads on [header ‖ f6]; Phase 2 MLP architecture: hidden 64,
  dropout 0.5, AdamW, early stopping on inner-val S3, L2 from {1e-4, 1e-3, 1e-2}):
  - **E2-mt, multitask:** class-balanced CE + λ · Huber(EF / 10) from the hidden layer, on verified-EF rows only.
    λ ∈ {0.1, 0.5, 1.0} is chosen on inner-val.
  - **E2-ldl, boundary-aware label distribution:** verified-EF rows are trained against soft targets P(class | EF ~
    N(EF_echo, σ²)), integrated over the sex-specific class bands. Other rows use their hard label. σ ∈ {3, 5, 7} EF
    points is chosen on inner-val. This encodes measurement uncertainty: a 41.5 % Mild is partly Moderate.
  - **E2-scl, supervised contrastive (EF-weighted):** CE + β · SupCon on the L2-normalised hidden representation.
    - Positive weight between two ECGs: exp(−|ΔEF| / 5) when both have verified EF; otherwise 1 for the same class and
      0 for a different class.
    - Temperature 0.1; β ∈ {0.1, 0.3} chosen on inner-val.
    - This pulls ECGs with similar EF together, so Mild is learned as a position on the EF scale.
  - **E2-ce, control:** the same trainer with plain class-balanced CE. It separates implementation effects from loss
    effects. E2 gates are applied against both the reference member and E2-ce, and both must pass.
  - E1 also re-runs f6 through its own pipeline (**E1-f6**), as a reproduction check against the reference member.
- **E3, lightweight CNN on the digitised waveforms (GPU).** Raw digital ECGs are not available (`DATA_ACCESS.md`), so
  the waveforms digitised from the PDFs are used. No signal is invented: the "short" construction, each lead's own
  2.5 s.
  - **E3-cnn:** a multi-lead 1D ResNet.
    - Input: 12 × 625 (250 Hz) for the leads, plus a rhythm-II branch of 1 × 2,500.
    - Four residual blocks (32-64-128-128 channels, kernel 7) with global average pooling, then CE + 0.5 · Huber(EF)
      on verified rows.
    - ≤ 60 epochs, early stopping on inner-val S3; class-balanced CE; AdamW 1e-3.
  - **E3-cnn+hdr:** the same network with the header vector concatenated before the head.
  - **Pilot:** repeat 0 only (5 folds, seed 42) against the reference member on the same folds. It advances only if it
    passes the gate on repeat 0 and reaches accuracy ≥ 0.50.
- **E4, fusion (only if complementary).** For each candidate that passes its full run:
  - nested inner OOF is computed;
  - complementarity with the C1 vote is checked on the inner OOF: error correlation < 0.8 and ≥ 10 % disagreement;
  - complementary candidates join the C1 vote (equal weight), and the C3 prior rule is re-fitted nested.
  - Non-complementary candidates are not fused.
- **E5, data quality and data needs (report only).** A summary of the matching, label-reliability and borderline-EF
  analyses (6B audit, VER / NF / MMCLEAN), plus the concrete data required for phase 2. The original dataset stays
  immutable.

## 4. Final-model rule and deliverables
- **Best Phase 7 model:** the admissible confirmatory candidate with the highest mean four-class accuracy on all 837
  ECGs. It is compared with C3 (Holm). If none beats C3 genuinely, C3 remains the best overall model.
- **Deliverables** in `research_experiments/phase_7/`:
  - `PHASE7_EXPERIMENT_INVENTORY.md`;
  - `PHASE7_MODEL_COMPARISON.csv`;
  - confusion matrices, accuracy and macro-F1 graphs, Mild precision / recall / F1 graphs;
  - `PHASE7_CLAIMS_REGISTER.md`;
  - `PHASE7_FINAL_REPORT.md`, stating whether 70 % was reached and which data are needed if not.
- **Storage:** per-ECG predictions only in the git-ignored `phase_7/data/private/`; aggregates only in git.
- **Integrity guard:** `phase_6/integrity6.py` excludes additive work through `PHASE6_DIRS`, in the same way as for
  phase_6a / 6b. `research_experiments/phase_7` is added to that list, so Phase 7 files are not counted as changes.
  Phase 1–5 files and the original data stay fully checked.
- **Checks before every commit:**
  - `phase_7.tests.test_phase7` (pre-registration, write guard, no recurrent layers, no ECG identifiers in tracked
    files);
  - the Phase 6 / 6A / 5 suites and `integrity6 verify phase6_before`.

## Amendments
### Amendment 1 (2026-11-05, after the E1–E3 pilots, before E5-lc)
**Pilot outcome.** No E1, E2 or E3 pilot passed its gate, so no full run and no E4 fusion follows; this is the
pre-registered consequence.

**E5-lc, learning curve.** This adds quantitative evidence to the data-needs section; it is not a candidate model.
- **Training size:** in every outer fold, the inner-train patients are subsampled to 25 / 50 / 75 / 100 % (whole
  patients, seeded 7000 + 10r + k; nested, so 25 % ⊂ 50 % ⊂ 75 %). Inner-val and the outer test fold are unchanged.
- **Models:**
  - the reference member (Phase 2 MLP fusion [header ‖ f6], seed 42): four-class accuracy, macro-F1 and Mild F1 on
    all 837 ECGs;
  - the R1 ridge EF regressor ([header ‖ f6 ‖ clinical]; `mildfix2.py`), trained on the same rows: out-of-fold EF
    residual SD on all 837 ECGs and on verified-EF ECGs.
- **Reporting:** the gain per doubling of training size (least-squares slope on log2 n) for each metric, and a
  descriptive extrapolation of the training size at which accuracy 0.70 and EF SD 9 would be reached.
  - The extrapolation is labelled indicative: few points, the same label quality assumed, and no claim that more data
    of this kind reaches the targets.

### Amendment 2 (2026-11-05, written before computing E6; owner: "combine the 69 % one with the Mild ≈ 0.3 model", then "don't use the header")
**E6, header-free two-stage cascade.** A complete four-class classifier is scored on all 837 ECGs.
- **No header information anywhere:**
  - no printed machine measurements, no age, no sex;
  - no Phase 3 "clinical" set (clin3), because 4 of its columns (QRS–T angle, JT, JTc, QRS ≥ 120 ms) come from the
    printed header.
- **Allowed inputs:** ECGFounder f6 of the digitised waveform, and the 121 image-only clinician features of
  `mildfeat.py`. The printed values were used there only for one global, label-free calibration choice; they are never
  per-ECG inputs.
- **Header-free EF regressors** (nested, as in `AMENDMENT_MILDFIX2.md`, target = nearest-echo EF):
  - **H1:** ridge on [f6 ‖ image features], block-standardised;
  - **H2:** LightGBM on [image features ‖ f6 PCA-32], fixed settings as R3;
  - **H3:** mean of H1 and H2.

  Each is trained unweighted, with noisy links × 0.3, or on clean links only. Nested selection per outer fold uses the
  inner-OOF RMSE on clean-link rows.
- **Stage 1** (thresholds set inside the training folds; target 0.70; ≥ 20 answered inner rows):
  - **G2T-img:** EF grading with the selected regressor, cut points d0 / t1 / t2 tuned on the inner OOF by the floor
    objective, then a confidence threshold on |EF̂ − nearest cut|;
  - **AGT-img:** answer only when IMG-C3 agrees with G2-img, plus the confidence threshold.
    - IMG-C3 = image-only vote of ECGFounder MLP + ECGFounder linear, plus the nested prior rule, 0.558.
    - Its inner decision uses its recorded per-fold α.
- **Stage 2:** every referred ECG is classified by an existing image-only model:
  - IMG-vote, Mild P / R / F1 0.297 / 0.238 / 0.264 (the "Mild ≈ 0.3" image-only model);
  - ECGFounder f6 linear (0.288 / 0.247 / 0.266);
  - CLIN-ORD, the ordinal model on image features (F1 0.251);
  - IMG-vote + CLIN (F1 0.236);
  - IMG-C3 (F1 0.227).
- **Grid and statistics:** 2 × 5 = 10 cascades, plus the two stage-1 abstention models on their own.
  - Primary comparison: vs IMG-C3, the best header-free four-class model (accuracy, macro-F1, Mild F1; paired
    corrected t-test; Holm over the 30 tests).
  - C3 (with header, 0.562) is shown as context only.
- **Final rule:** the admissible cascade with the highest all-837 accuracy, plus the cascade with the highest Mild F1.
  Both are maxima over candidates, so they are optimistic.

### Amendment 3 (2026-11-05, written before computing E6 or E7; owner: "we don't have to take all of the data ... 45-60 %, but Mild should be predicted and 4-class accuracy above 69 %")
**E7, header-free selective prediction that must still detect Mild.**
- **Mechanism:** the model answers only the ECGs it is confident about; the others are referred to echo. Which ECGs are
  answered is decided by the model's confidence on each ECG, never by the true label or by echo information.
- **Base predictor:** IMG-vote (image only; argmax). The **AG** variant also requires agreement with the header-free
  EF grading of E6.
- **Thresholds:** one confidence threshold per predicted class. Candidates are quantiles 0–90 % of that class's inner
  confidence, or "never answer".
- **Choice, per outer fold, on that fold's training portion only** (nested inner OOF):
  - maximise the answered share;
  - subject to accuracy on answered ≥ 0.70;
  - and **overall Mild recall ≥ m**, where m ∈ {0.10, 0.15, 0.20}. Overall Mild recall = true-Mild ECGs answered and
    predicted Mild, divided by **all** true-Mild ECGs. Referred Mild patients count as missed.
  - If no setting is feasible, that fold answers nothing (reported).
- **Reported on all 837 ECGs** for the 2 variants × 3 Mild requirements:
  - answered share;
  - accuracy on answered (corrected CI);
  - overall Mild recall and Mild precision among answered;
  - per-class answered share and recall;
  - the predicted mix.

  The headline is always "accuracy X on the Y % answered". The all-837 four-class numbers (E6) are reported next to
  it, and selective accuracy is never presented as accuracy on all patients.
- **Owner rule:** accuracy on answered > 0.69 with Mild detected (overall Mild recall ≥ 0.10). This is checked on the
  outer folds and is optimistic as the best of 6.

### Amendment 4 (2026-11-05, written before computing E8; owner: "think like a doctor, more image factors, predict Mild properly, take the best percentage of data, don't let Normal / Mild / Moderate overlap, focus on Severe too")
**E8, doctor-style hierarchical classification with refer zones** (header-free).
- **Image factors already measured** (`mildfeat.py`, mild-factor study):
  - Normal vs Mild: QRS width AUROC 0.73, spatial QRS–T angle 0.70;
  - Mild vs Moderate: ≤ 0.63 (T wave in III, R-peak time V1, anterior Q).
- **Inputs:** [ECGFounder f6 ‖ the 121 image clinician features ‖ the 150 ECGFounder diagnostic outputs (dx)],
  block-standardised on the fitting rows. No header, age, sex or clin3.
- **Specialists** (class-balanced logistic regression; C ∈ {0.003, 0.01, 0.03, 0.1} chosen on inner-val by AUROC,
  then reused for the inner OOF):
  - **A:** abnormal (Mild / Moderate / Severe) vs Normal, on all rows;
  - **S:** Severe vs Mild + Moderate, on abnormal rows;
  - **M:** Mild vs Moderate, on Mild + Moderate rows.
- **Decision**, as a clinician reads it, with a refer zone at every step (no overlap between neighbouring classes):
  1. Normal if pA ≤ a_lo; abnormal if pA ≥ a_hi; otherwise refer.
  2. For abnormal: Severe if pS ≥ s_hi; not severe if pS ≤ s_lo; otherwise refer.
  3. For not severe: Mild if pM ≥ m_hi; Moderate if pM ≤ m_lo; otherwise refer.
- **Thresholds** are chosen per outer fold on the nested inner OOF (4-fold, `existing6b` splits). Candidates for each
  are the 10–90 % quantiles of that probability on the inner rows, with lo ≤ hi.
- **Selective objective:** maximise the answered share subject to (inner):
  - accuracy on answered ≥ T, with T ∈ {0.65, 0.70};
  - overall Mild recall ≥ mr, with mr ∈ {0, 0.05, 0.10};
  - overall Severe recall ≥ sr, with sr ∈ {0, 0.15}.

  "Overall" divides by **all** patients of that class, so referred patients count as missed. That gives 12 variants.
  A fold with no feasible setting answers nothing (reported).
- **H-full (no refer zone, all patients):** lo = hi at each step, chosen by the floor objective (accuracy subject to
  every recall ≥ 0.15).
- **Reporting:** for each variant, the answered share, accuracy on answered (corrected CI), per-class recall among
  answered, Mild and Severe recall overall, Mild precision, and the predicted mix. For H-full, the four-class metrics
  on all 837 ECGs.
  - References: IMG-C3 (0.558 on all), the header-free triage (0.694 on 56 %) and its 6 % Mild tilt (0.690).
  - Every operating point is chosen inside the training folds. The best of 12 on the outer folds is optimistic.

### Amendment 5 (2026-11-06, written before freezing; owner: "can I create a dataset which I can use to test" → "yes, build it", on a new branch `research/phase7-testkit`)
**Purpose:** a frozen copy of the header-free triage model (AGT-img, the 56 %-answered / 0.69 operating point) and a
one-shot scorer for a test set the owner collects. No new modelling, nothing tuned.

- **Frozen model** (`testkit/freeze.py`), built from all 837 development ECGs with the E6 recipe applied one level up
  (outer CV OOF in place of inner OOF):
  - **IMG-vote members:** the ECGFounder-f6 MLP (seeds 42 / 43 / 44) and the linear distance-CE head (seed 42), each fit
    on all 837 with its own grouped 80/20 selection split (`fit_predict` recipe; split seed 5000).
  - **Prior-rule alpha:** `choose_alpha(..., "protected")` on the committed IMG-vote outer OOF, pooled over the 3 repeats.
  - **EF regressor:** the E6 variant (H1 / H2 / H3 × W1 / W.3 / W0) with the lowest pooled outer-OOF RMSE on the
    verified (clean-link) ECGs; then refit on all 837 (H2 split seed 5001).
  - **Grading cuts and answer threshold:** the floor-objective cuts (d0, t1, t2) and the distance threshold (answered
    accuracy ≥ 0.70, ≥ 20 answered) chosen on the pooled outer OOF.
  - **Answer rule unchanged:** answer only when the EF grade agrees with IMG-C3 and the estimated EF is at least the
    threshold away from every cut; otherwise refer.
  - The frozen file stays in `phase_7/data/private/testkit/` (git-ignored). Its SHA-256 and every chosen value go in the
    committed card `results/testkit_frozen_card.json`.
- **Checks before use (development data only, aggregates only):**
  - the PDF reader reproduces the cached f6 and the 121 image features on a random sample of development PDFs, and the
    frozen model gives the same answer from PDF-derived and cached inputs;
  - the member and EF code paths reproduce the E6 / IMG-vote recipe on outer fold r0 (OOF accuracy and RMSE vs the
    committed runs).
- **Scorer** (`testkit/score.py`). The test-set rules are fixed before any test data exist:
  - **Manifest:** `ecg_file`, `patient_id` (pseudonymised), `sex`, `ef` (echo, numeric), `echo_gap_days`, `ef_method`.
    The label comes from `ef` and `sex` by the project rule (`data_check.rule`).
  - **Primary analysis:** rows with an echo gap ≤ 30 days and a readable PDF. Excluded rows are counted and reported.
  - **Refusals:** a frozen file whose hash differs from the card; any ECG that belongs to the development set (same
    file-name timestamp, or near-identical ECGFounder embedding, cosine ≥ 0.999); any locked-test ECG (path or
    file-name timestamp from `data/manifest.csv`, metadata only) unless the owner passes `--locked-test-authorised`;
    a manifest already scored, unless `--again "<reason>"` (the reason is logged and reported).
  - **Reported** (with 95 % CIs: Wilson, and a patient-cluster bootstrap):
    - triage: answered share, accuracy on answered, class mix, the composition of the "Normal" calls, rule-out of
      EF < 41 %, and Mild caught;
    - four-class IMG-C3 for every readable ECG: accuracy, balanced accuracy, macro-F1, per-class P / R / F1, confusion;
    - EF-estimate error;
    - side by side with the CV expectations (56 % [48, 63] answered at 0.694 [0.655, 0.736]; IMG-C3 0.558 [0.504, 0.611]).
  - **Outputs:** per-ECG outputs go to `data/private/testkit/runs/<name>/` (row numbers only, never file names); the
    aggregate JSON goes to `results/testkit/`. Nothing is printed that identifies a patient.
- **Not done here:** no locked-test scoring, and no change to any model after the first test result. A model changed
  after seeing test results needs a new freeze and a new test set.

### Amendment 6 (2026-11-06, written before any locked-test ECG is read; owner: "now use this data to train and give accuracy final for class 4", then, asked how, chose "Score locked 210 (Recommended)": "I authorise using the locked test set once")
**One-shot locked-test evaluation of the frozen model.** The locked test (210 ECGs, set aside in Phase 2) is used
once, with the owner's explicit authorisation. Afterwards it is no longer unseen.

- **Model:** only the frozen header-free triage model of Amendment 5 (SHA-256 `09808d12…`, trained on all 837
  development ECGs). No other model is scored on the locked test, and nothing is changed after scoring.
- **Primary result, "final four-class accuracy":** IMG-C3, the frozen model's four-class prediction for every scored
  test ECG.
- **Secondary result:** the triage (answered share, accuracy on answered, class mix, composition of the "Normal" calls,
  rule-out of EF < 41 %, Mild caught).
- **Labels:** the folder labels in `data/manifest.csv`, the same definition as every development result.
- **Sex for the Normal / Mild cut:** the sex printed on the ECG (`header_features.csv: male`). It agrees with the record
  sex on 837 / 837 development ECGs. Sex is not a model input.
- **Primary set:** the 210 test ECGs minus:
  - (a) ECGs of a test patient key that forms a near-identical name pair with a development key. This uses the Phase 5
    criterion (difflib ratio ≥ 0.90, or containment of ≥ 8 letters), recomputed in memory with nothing name-derived
    written. The count must equal the audit's 11 straddling pairs, or nothing is scored;
  - (b) ECGs whose ECGFounder embedding is near-identical to a development ECG (cosine ≥ 0.999). The Phase 5 audit found
    no near-duplicate rhythm strips across the boundary.
- **Sensitivity set:** all readable test ECGs. Unreadable PDFs are counted and excluded.
- **CIs:**
  - Wilson intervals, and a patient-cluster bootstrap (2,000 resamples);
  - clusters are patient keys, with the 3 within-test near-identical name pairs merged.
- **Comparison:** each key number next to its CV expectation:
  - IMG-C3 0.558 [0.504, 0.611];
  - triage 56 % [48, 63] answered at 0.694 [0.655, 0.736].
- **Outputs:**
  - aggregates in `results/testkit/locked_test_scores.json`;
  - per-ECG outputs (test row numbers only) in `data/private/testkit/runs/locked_test/`;
  - a ledger entry. The script refuses a second run.
- **Interpretation rules:**
  - the result is reported whatever it is;
  - a difference from CV is not tuned away;
  - any later model change needs new test data (`testkit/TESTKIT.md`).
