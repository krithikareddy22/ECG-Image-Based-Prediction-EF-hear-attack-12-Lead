# Phase 6 plan and pre-registration

**Status:** fixed before any Phase 6 model run. SHA-256 in `configs/PLAN.sha256`. Any later change goes into a dated
**Amendments** section at the end, with a reason; the text above it is never edited.

- **Branch:** `research/phase6`, created from `research/phase5` @ `27cbdbd`. Phase 6 is additive: no Phase 1–5 file,
  result or report is modified (`integrity6`: git diff against `27cbdbd` + SHA-256 snapshot `phase6_before`).
- **Task:** four-class LVEF (Normal / Mild / Moderate / Severe; Teichholz EF; Normal M > 52 / F ≥ 54, Mild ≥ 41 and
  below Normal, Moderate 31–< 41, Severe < 31) from the 12-lead ECG PDF printout.

## 1. Starting point (fixed facts; verified, not re-derived)
| Item | Value |
|---|---|
| Development set D | 837 ECGs / 794 name-derived patient keys (Normal 384, Mild 185, Moderate 177, Severe 91) |
| Locked test T | 210 ECGs / 199 keys (96 / 47 / 44 / 23). Scored once in Phase 1; **not loaded, predicted, scored or inspected in Phase 6**; no T PDF or waveform is run through any model, even unlabelled |
| Folds | `phase_2/configs/folds_D.csv`: StratifiedGroupKFold(5), repeats 0–2, grouped by patient key; inner-train / inner-val per outer fold. Nested 4-fold inner CV as in `phase_5/experiments/nested.py` |
| C1 | Phase 3 vote ensemble: accuracy 0.547, balanced 0.467, macro-F1 0.455 (admissible) |
| C2 | calibrated MLP fusion: 0.568 ± 0.003 (Mild recall 0.11; not admissible; accuracy reference) |
| C3 | Phase 5 selected `P5D-E5-vote-prior-protected-v1`: 0.561 (95 % CI 0.520–0.603), balanced 0.456, macro-F1 0.448, Mild F1 0.221, min recall 0.177 |
| Not admissible | class-unweighted CE MLP 0.572 (Mild recall 0.144) |
| Error budget (Phase 4) | predicted vs echo EF r = 0.66 (R² 0.43). 60 % four-class accuracy needs r ≈ 0.82; 65 % needs r ≈ 0.87 |
| Inputs | frozen ECGFounder f6 (1,024-d, "aligned" 12 × 5000 input), 16 header, 28 clinical features, digitised waveforms (12 × 2.5 s + 10-s lead II, 500 Hz). FP32, CPU |

**Already done, not repeated:** Phase 4 EF regression, regression → class, multitask on frozen embeddings,
label-quality training, Mild study, matched pool; Phase 5 E1–E7 (43 candidates, 0 genuine improvements); Phase 2–3
last-block (stage-6 block 3) fine-tuning on repeat 0, masking / coverage variants of the ECGFounder input.

## 2. Hypotheses
- **H-A (representation):** fine-tuning ECGFounder beyond the last block (stage 6; then stages 5–6), jointly on the
  four-class label and continuous EF, raises the out-of-fold EF correlation above the frozen embedding (r 0.66) and
  four-class accuracy above C3.
- **H-B (alternative encoders):** an openly licensed pretrained ECG encoder gives a frozen embedding with higher EF
  correlation than ECGFounder f6.
- **H-D (input fidelity):** digitisation errors limit performance: EF error and four-class errors increase with
  signal-quality flags and reduced lead coverage, and the 10-s lead II carries EF information beyond its 2.5-s window.
- **H-E (diversity):** a new representation that passes the screen adds accuracy when combined with the C1 members
  through the existing nested stacking or the E5 prior-protected rule.
- **H-N (grouping):** merging the 15 within-D near-identical name pairs into one group does not change the
  development-CV estimates materially (|Δ accuracy| < 0.01).

## 3. Two-stage design
### Stage 1: screen (cheap; seed 42 × repeats 0–2, or deterministic × repeats 0–2)
- **Primary screen metric:** Pearson r between the candidate's out-of-fold predicted EF and the nearest-echo EF
  (Phase 4 label-agnostic target), pooled over all 837 ECGs per repeat, mean over repeats 0–2.
- **Secondary screen metric:** four-class macro-F1 of the candidate's class output (argmax), mean over repeats.
- **Pass rule (pre-registered):** r ≥ **0.69** (≥ +0.03 over 0.66), **or** macro-F1 ≥ **0.465** (C1 + 0.010) with
  every class recall ≥ 0.15. Candidates that fail are reported and stop there.
- Also reported: r on cohorts C (≤ 14 d) and G (high confidence), MAE, R², calibration slope, accuracy,
  balanced accuracy, per-class recall.
- **Same-code references:** every family has a reference built with the same code and folds, differing only in the
  factor under test (§4). Δr vs the reference is reported with a patient-cluster bootstrap CI (repeat 0) and per-fold
  differences (15 cells).

### Stage 2: full evaluation (only Stage-1 passers)
- 3 repeats × 3 seeds (42 / 43 / 44) for stochastic models; seeds combined within a repeat by probability mean.
- Full metric suite: accuracy (primary), balanced accuracy, macro-F1, per-class precision / recall / F1, QWK, macro
  AUROC, confusion matrix, Severe recall and precision, EF r.

## 4. Experiments (deterministic ids; registry in `registry6.py`)

| ID | Change (one factor) | Same-code reference | Notes |
|---|---|---|---|
| **R0** `P6R0-joint-frozen-v1` | – (reference): joint class + EF head on header ‖ frozen ECGFounder stage-6 output, backbone frozen | Phase 4 multitask (r ≈ 0.66, accuracy ≈ 0.55) | implementation check (§4.1) |
| **A1** `P6A1-ft-s6-joint-v1` | stage 6 (4 blocks) trainable, progressive unfreezing | R0 | inputs: cached stage-5 activations (exact: Net1D has no batch-norm / dropout) |
| **A2** `P6A2-ft-s56-joint-v1` | stages 5–6 trainable, progressive unfreezing | R0 and A1 | inputs: cached stage-4 activations |
| **B0** `P6B0-ecgf-f6-v1` | – (reference): frozen ECGFounder f6 through the B pipeline | – | |
| **B-x** `P6B-<encoder>-v1` | frozen embedding of another encoder through the B pipeline | B0 | only encoders with a clear permissive licence (§4.2) |
| **D1** `P6D1-quality-strata` | – (analysis, no training): EF error and four-class accuracy of R0 and C3 by sq-v2 poor-lead count, quality score, lead coverage | – | |
| **D2** `P6D2-leadII-2p5s-v1` | aligned input with the 10-s lead II replaced by its own 2.5-s window (frozen f6 → B pipeline) | B0 | the only input variant tested |
| **E** `P6E-<combo>-v1` | Stage-2 passer added as a member to the C1 vote, then (i) E5 prior-protected nested rule, (ii) E3 nested stacking (LR, balanced) | C1, C3 | only after A, B and D; only with Stage-2 passers |
| **N** `P6N-<model>-merged-v1` | outer and inner folds regenerated with the 15 within-D name pairs merged into one group | same recipe on the original folds | C1 and C3 recipes; plus the selected Phase 6 model if any |

### 4.1 Experiment A: ECGFounder fine-tuning (CPU, FP32)
- **Network:** ECGFounder (Net1D, MIT licence, checkpoint SHA-256 `ee199f37…7997`). Stages below the trainable part
  are frozen and run once per ECG (development ECGs only); their outputs are cached in the git-ignored private store.
  Because Net1D uses neither batch-norm nor dropout, training on the cached activations is exactly equivalent to
  training the full network with those stages frozen.
- **Head (the Phase 4 multitask body):** mean-pooled stage-6 output (1,024) ‖ header numeric (16; medians and
  standardisation fitted on inner-train) → Dropout 0.5 → Linear(1,040 → 64) → GELU → Dropout 0.5 → class head
  (4-way, class-balanced CE on the folder label) and EF head (Huber δ = 1 on z-scored nearest-echo EF; z-scoring
  fitted on inner-train). Loss = CE + λ · Huber with **λ = 1** (fixed; Phase 4 showed λ makes no difference).
- **Progressive unfreezing:** head only for 5 epochs, then the last block of stage 6 for 5 epochs, then all of
  stage 6 (A1); for A2, then stage 5 as well from epoch 15. Discriminative learning rates: head 1e-3, backbone 1e-4
  (stage 5: 5e-5); AdamW, weight decay 1e-2 on the backbone and 0.1 on the head (= the Phase 2 L2 1e-3; no L2
  grid, to bound compute); batch 32; at most 40 epochs.
- **Early stopping and selection:** on the fold's **inner-val** only (patient-grouped, from `folds_D.csv`):
  criterion = mean of inner-val S3 and inner-val EF Pearson r, patience 8 epochs, after a minimum of 10 epochs.
  The outer fold is never used for any choice.
- **Outputs per outer fold:** class probabilities, predicted EF (de-standardised).
- **Compute:** estimated from a timing run (real weights, development ECGs, forward / backward time only, no
  metric recorded) before any A run; the estimate is recorded as an amendment. Full-backbone fine-tuning is **not**
  run on CPU (owner decision).
- **R0 check:** R0 uses the same code with the backbone frozen and is expected near Phase 4 (r ≈ 0.66, accuracy
  ≈ 0.55). A deviation > 0.01 in r is reported and investigated before A runs; R0 is never tuned toward a number.

### 4.2 Experiment B: alternative pretrained encoders
- **Licence gate (before any download):** an encoder is used only if its weights carry a clear permissive licence
  (MIT, Apache-2.0, BSD, CC-BY-4.0 or equivalent), recorded with source URL, revision and SHA-256 in
  `configs/encoders.json`. Non-commercial, research-only, unclear or missing licences → **stop and ask the owner**.
  A blocked download is reported, not worked around.
- **Candidates (subject to the gate):** ECG-FM, HuBERT-ECG, ST-MEM, ECG-JEPA, MERL, and any other openly licensed
  12-lead encoder found.
- **Input:** the same aligned 12-lead construction (true lead timing, no synthesised signal), resampled / cropped only
  as the encoder requires (recorded). Development ECGs only.
- **B pipeline (identical for B0, B-x, D2):** (i) EF: ridge regression on header ‖ embedding (block-standardised on
  inner-train; α ∈ {0.1, 1, 10, 100, 1000} chosen on inner-val MAE); (ii) class: the Phase 2 MLP fusion (header ‖
  embedding, `p2heads` MLP, class-balanced CE, L2 and epoch on inner-val S3), seed 42 in Stage 1.

### 4.3 Experiment C: extra unlabelled / matched ECGs: **not run**
The repository holds only the 1,047 study PDFs; the 3,130 additional matched ECGs exist only as spreadsheet rows
(99 % Normal) and no extra PDFs or waveforms are available. Confirmed blocked by the owner.

### 4.4 Experiment D: digitisation and input fidelity
- **D1 (analysis):** for R0 (EF) and C3 (class), absolute EF error and accuracy by stratum: sq-v2 poor-lead count
  (0 / 1–2 / ≥ 3), quality-score tertiles, rhythm-strip quality flag, number of leads with usable signal. Test:
  Spearman correlation of |EF error| with each quality variable; accuracy by stratum with Wilson CIs. Strata < 5 ECGs
  are suppressed.
- **D2 (one input variant):** see table. Phase 3 already showed that no masking / coverage variant beats "aligned"
  for classification; D2 asks only whether the 10-s lead II adds **EF** information beyond its 2.5-s window.

### 4.5 Experiment E: diverse ensembling (only after A–D)
- Members: the three C1 members (MLP fusion 3 seeds, ECGFounder linear distance-CE, header + clinical LR) plus each
  Stage-2 passer. Inner out-of-fold predictions use the Phase 5 nested protocol; a fine-tuned member uses seed 42 only
  for its inner OOF (as single-seed members in Phase 5). Expected compute is recorded before running; if it exceeds
  12 CPU-hours the owner is asked first.
- No new decision-rule tuning on the existing embeddings alone.

### 4.6 N: name-pair sensitivity
- The 15 near-identical name pairs **within D** (Phase 5 criterion: difflib ratio ≥ 0.90 or containment, names read
  in memory only from development rows) are merged into one group (union-find). The mapping is stored only in the
  git-ignored private store; committed outputs contain counts only.
- Folds are regenerated exactly as `phase_2/protocol.py` does (outer StratifiedGroupKFold(5), random_state 2025 + r;
  inner split random_state 100 + k) but with the merged groups; nested inner CV and every inner 80/20 split also use
  the merged groups.
- No ECG is moved between D and T, dropped or relabelled.
- The 11 pairs that straddle D and T and the 3 within T are only **counted** (no names); they must be resolved by the
  owner before any locked-test evaluation.

## 5. Metrics, admissibility, eligibility and statistics
- **Primary metric:** four-class accuracy on all 837 ECGs (pooled outer-fold predictions), mean over repeats 0–2,
  **subject to admissibility**.
- **Admissibility:** every class recall ≥ 0.15 (mean over repeats). Sensitivity at 0.10 and 0.20 reported.
- **Eligibility:** all 3 repeats (× 3 seeds for stochastic models). Never selected on a single repeat or seed.
- **Comparison set:** C1, C2, C3 (regenerated per-ECG predictions; must reproduce the committed aggregates within
  0.005 in accuracy, balanced accuracy and macro-F1).
- **Statistics:**
  - corrected repeated-CV t-test (Bouckaert–Frank, N_train 535 / N_test 167) on the 15 matched (repeat, fold) cells,
    candidate − comparator;
  - Holm over all Stage-2 candidates for each comparator (global), and within each family (A, B, E);
  - patient-cluster paired bootstrap (2,000 resamples, repeat 0) and exact McNemar (repeat 0) for the top three
    admissible candidates and any genuine improvement.
- **Genuine improvement over a comparator:** Holm-adjusted p < 0.05 **and** the same sign in all 3 repeats **and**
  the candidate admissible. Required separately for C1, C2 and C3.
- **Selection:** the best admissible, eligible model by mean accuracy (ties: higher macro-F1). Non-admissible models
  are reported, never selected.

## 6. Decision gates (development-CV accuracy of the best admissible model)
| Accuracy | Action |
|---|---|
| < 58 % | report as a negative result; investigate data limitations (no further tuning on the same inputs) |
| 58–60 % | assess complementary models (E) |
| 60–62 % | refine the passing representation (pre-registered amendment required) |
| 62–65 % | consolidate and test robustness (N, seeds) |
| ≥ 65 % | nominate a frozen candidate for the owner's decision on a one-time locked-test evaluation |

A model is proposed for freezing only if it is a genuine improvement over C3. In every case the 11 D/T name pairs
must be resolved and the owner must authorise in writing before any locked-test access.

## 7. Leakage controls and privacy
- Echo EF is a training **target** (A heads, B ridge) and an analysis variable only; never an input.
- All preprocessing, standardisation, imputation, early stopping, hyper-parameters and decision rules are fitted on
  training rows of the relevant (inner) fold; patient-disjointness of every split is asserted.
- Phase 6 code imports only the development-only loader (`phase_5.p5common`); it never imports `cv_features`,
  `accuracy_tradeoff`, `p3common`, `p3_robust`, `p4common` or `phase_3.clinical_features` (tested).
- Per-ECG predictions, embeddings, cached activations, features, inner caches, name-pair groups and any trained
  weights stay in the git-ignored `phase_6/data/private/`. Committed: code, aggregate metrics, aggregate plots,
  reports. `privacy_scan` (all ECG ids, timestamps, patient keys) and a staged-diff review before every push.

## 8. Resumability
- Every run: configuration, config hash, seed, repeat, aggregate metrics (`results/experiments/<run_id>.json`) and
  private predictions (`data/private/runs/<run_id>/oof.csv`, hash recorded in the aggregate).
- `resume6` classifies runs (DONE / INTERRUPTED / RUNNING / PENDING / CORRUPT), writes `checkpoint.json` and the
  status table in `PHASE6_STATUS.md`, and prints the next command. Completed runs are skipped; a correction gets a
  new version suffix (`-v2`), never an overwrite.
- Expected compute is recorded in `PHASE6_STATUS.md` before each stage starts.

## Amendments

### Amendment 1 (2026-11-05): compute estimate for experiment A (from the M1 timing run; no recipe change)
- **Reason:** §4.1 requires the timing-based estimate to be recorded before any A run.
- **Measured** (`results/timing_A.json`; real weights, 64 development ECGs, batch 32, FP32 CPU; compute only):
  train step per ECG with 4 threads / 1 thread: A1 (stage 6) 8 / 21 ms; A2 (stages 5–6) 21 / 55 ms; full backbone
  45 / 139 ms. Caching the frozen part for all 837 development ECGs takes < 1 min.
- **Projected wall-clock** (4 single-thread fold jobs in parallel; 25–40 epochs per fold):
  | Experiment | Stage 1 (15 folds) | Stage 2 extra (30 folds) | E inner OOF (60 fits) |
  |---|---|---|---|
  | A1 stage 6 | 0.3–0.5 h | 0.6–1.0 h | 1.0–1.5 h |
  | A2 stages 5–6 | 0.8–1.3 h | 1.6–2.6 h | 2.4–3.9 h |
  | full backbone (not run; owner decision) | 2.1–3.3 h | 4.1–6.6 h | 6.2–9.9 h |
- **Consequence:** A1 and A2 run on CPU as pre-registered; no GPU is needed for them. Full-backbone fine-tuning
  would also be feasible on CPU (≈ 6–10 h for Stages 1 + 2), but it stays out of scope unless the owner decides
  otherwise. Projections ignore Python overhead and memory-bandwidth contention (factor up to ≈ 1.5×).

### Amendment 2 (2026-11-05): exploratory combined-encoder candidate (added after the B-ECG-FM screen)
- **Reason:** ECG-FM alone failed the Stage-1 screen (r 0.609 vs 0.638 for ECGFounder through the same pipeline).
  Two encoders pre-trained on different data could still be complementary. The owner asked for continued four-class
  optimisation. This candidate was added **after** seeing the ECG-FM result, so it is labelled exploratory.
- **Candidate `P6B-f6+ecgfm-v1`:** the unchanged B pipeline on header numeric ‖ ECGFounder f6 ‖ ECG-FM
  (three blocks, block-standardised on inner-train). Same Stage-1 screen and pass rule; reference B0. It enters
  Stage 2 / E only if it passes, and is then counted in the Holm family like every other candidate.
