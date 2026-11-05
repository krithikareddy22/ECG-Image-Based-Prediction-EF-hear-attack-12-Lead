# Phase 7 final report: ECG-based four-class LVEF classification with reliable Mild detection

**Data:** development set (837 ECGs, 794 patient keys) for sections 1–7. The locked 210-ECG test cohort was scored
once, with the owner's authorisation, by the frozen model only (section 8).
**Protocol:** Phase 2 patient-grouped 5-fold × 3 repeats, identical folds for every model; nested selection
(inner-val); corrected repeated-CV CIs; Holm correction.
**Pre-registration:** `PHASE7_PLAN.md` (commit 0d14775), Amendment 1 (commit a1a7eef), Amendments 2–3 (commit 66a85e5) and Amendment 4 (commit a16f56f). Every number below is an
aggregate on all 837 ECGs with the original four-class labels; the evaluation population was never changed.
**Recurrent models** (LSTM / BiLSTM / GRU / CNN-LSTM / ECGFounder-LSTM) were discontinued. None was running, none was
launched, and a test blocks recurrent layers in Phase 7 code.

## 1. Answers
| Question | Answer |
|---|---|
| Best original four-class accuracy | **0.562 [0.519, 0.605]**, C3 (C1 vote + nested prior rule); unchanged by Phase 7 |
| Best overall model | **C3**: balanced accuracy 0.457, macro-F1 0.449 |
| Mild precision / recall / F1 (C3) | **0.300 / 0.178 / 0.224** |
| Best Mild F1 of any model | Header + clinical LR, 0.272 (P 0.305 / R 0.247; accuracy 0.503). Admissible ensemble: C1 vote, 0.257 (accuracy 0.548) |
| Statistical significance | No Phase 7 candidate beats its reference or C3: 44 paired comparisons, smallest Holm p 0.51 |
| 70 % reached? | **No.** None of the six owner targets is met by any model |
| Phase 7 candidate passing its gate | None. Under the pre-registered rule, no full run and no fusion followed |
| Selective prediction with Mild detected and > 0.69 | Infeasible: with Mild detected, the best achievable accuracy is 0.59–0.64 (section 2b); the doctor-style hierarchy with refer zones is infeasible too (section 2c) |

Per-class performance of C3 (precision / recall / F1):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.710 | 0.846 | 0.772 |
| Mild | 0.300 | 0.178 | 0.224 |
| Moderate | 0.457 | 0.460 | 0.458 |
| Severe | 0.344 | 0.344 | 0.343 |

## 2. What was tried (pilots: seed 42, 3 repeats; gate = accuracy ≥ +0.010 or Mild F1 ≥ +0.03, macro-F1 ≥ −0.010)
| Exp. | Candidate | Accuracy | Macro-F1 | Mild P / R / F1 | Gate |
|---|---|---|---|---|---|
| ref | MLP fusion [header ‖ f6], seed 42 | 0.544 | 0.436 | 0.283 / 0.132 / 0.179 | — |
| E1 | f6 through the Phase 7 pipeline (reproduction) | 0.544 | 0.436 | identical | Δ 0.000 |
| E1 | stage-5 mean ‖ max ‖ std | 0.517 | 0.409 | 0.263 / 0.146 / 0.187 | fail |
| E1 | per-printed-column stage-5 mean | 0.527 | 0.431 | 0.271 / 0.157 / 0.199 | fail |
| E1 | stage-4 mean ‖ f6 (multi-scale) | 0.529 | 0.425 | 0.250 / 0.141 / 0.178 | fail |
| E1 | 150 pre-trained ECGFounder diagnostic outputs | 0.531 | 0.412 | 0.287 / 0.117 / 0.165 | fail |
| E2 | CE control (own trainer) | 0.541 | 0.432 | 0.280 / 0.112 / 0.159 | — |
| E2 | multitask CE + verified-EF regression | 0.545 | 0.440 | 0.289 / 0.142 / 0.191 | fail |
| E2 | boundary-aware soft labels (EF measurement uncertainty) | 0.548 | 0.433 | 0.303 / 0.123 / 0.174 | fail |
| E2 | EF-weighted supervised contrastive | 0.546 | 0.436 | 0.276 / 0.130 / 0.176 | fail |
| E3 | multi-lead 1D ResNet, digitised waveforms (repeat 0) | 0.496 | 0.384 | 0.264 / 0.178 / 0.213 | fail (ref 0.559 on repeat 0) |
| E3 | 1D ResNet + header (repeat 0) | 0.485 | 0.378 | 0.243 / 0.189 / 0.213 | fail |

**Not repeated, with reasons:**
- ECGFounder fine-tuning: Phase 6 A1 / A2 failed (r 0.650 vs 0.652).
- Image ResNet: 0.40–0.49 in Phases 1–3.
- Mild-probability boosting: excluded by the brief, and shown in Phase 6B to trade accuracy for Mild calls that are
  right only about 1 time in 4.

The full table, including existing models (C2 0.568 is not admissible: Mild recall 0.112), is in
`PHASE7_MODEL_COMPARISON.csv`. Graphs:
- `plots/phase7_accuracy_macro_f1.png`;
- `plots/phase7_mild_performance.png`;
- `plots/phase7_confusion_matrices.png`;
- `plots/phase7_learning_curve.png`.

## 2b. Header-free combination of the "69 %" triage with a Mild-capable model (Amendments 2-3)
The owner asked to combine the ~69 % abstention model with the "Mild ≈ 0.3" model, without the header, and allowed
answering only a share of ECGs as long as Mild is predicted.

**Inputs:** the ECGFounder embedding of the digitised waveform and the 121 image-only clinician features. Not used:
printed machine measurements, age, sex, and the Phase 3 "clinical" set (4 of its columns come from the header).

**E6, two-stage cascade** (complete four-class classifier, all 837 ECGs):
- **Stage 1:** header-free EF grading answers the confident ECGs:
  - **G2T-img:** 0.693 accuracy on the 53 % answered;
  - **AGT-img** (agrees with IMG-C3): 0.694 on 56 %.
  - Both answer almost only Normal (≈ 395 of ≈ 450 answers), and Mild recall among answered is 0.00–0.02.
- **Stage 2:** an image-only model classifies the referred ECGs. It is right only 35–42 % of the time there, because
  those are the hard ECGs.
- **Results vs IMG-C3:**

  | Model | All-837 accuracy | Mild P / R / F1 | Admissible |
  |---|---|---|---|
  | IMG-C3 (header-free reference) | 0.558 [0.504, 0.611] | 0.290 / 0.187 / 0.227 | yes |
  | IMG-vote ("Mild ≈ 0.3" model, alone) | 0.541 [0.502, 0.580] | 0.297 / 0.238 / **0.264** | yes |
  | G2T-img → IMG-C3 (highest accuracy) | **0.568** [0.523, 0.612] | 0.281 / 0.137 / 0.184 | no (Mild recall < 0.15) |
  | G2T-img → IMG-vote | 0.557 [0.515, 0.599] | 0.285 / 0.157 / 0.202 | yes |
  | AGT-img → IMG-vote (best cascade Mild F1) | 0.548 [0.500, 0.596] | 0.289 / 0.195 / 0.232 | yes |

  - AGT-img → IMG-C3 is identical to IMG-C3 by construction: stage 1 only answers where it agrees with IMG-C3.
  - No cascade is a genuine improvement (30 tests, Holm).
  - Cascading lowers Mild detection compared with IMG-vote alone, because stage 1 confidently calls many Mild ECGs
    Normal.
- **Header-free EF error** (nested-selected regressor): SD 13.5 on all ECGs, 12.4 on verified-EF ECGs. This is the same
  as with the header.

**E7, answer a share of ECGs, keep accuracy ≥ 0.70, and detect ≥ 10 / 15 / 20 % of all Mild patients:**
- **Infeasible in all 15 training folds, for every variant.** No per-class confidence threshold met both conditions
  on the training-fold predictions, so the model answered nothing (`results/e7_selective_mild.csv`).
- **Why** (`e7_feasibility.py`, training-fold predictions):
  - even the most confident Mild calls are right only 20–50 % of the time (top 10), and 27–32 % overall;
  - catching 10 % of all Mild patients takes 48–58 Mild calls, of which about 70 % are wrong;
  - combined with the most accurate non-Mild answers, the best achievable accuracy on answered ECGs is 0.59–0.64.
- **The arithmetic:** with Mild calls right about 30 % of the time, keeping accuracy above 0.69 allows Mild to be under
  1 % of the answers, i.e. one or two Mild patients detected. "Accuracy > 0.69" and "Mild detected" cannot both hold
  with the information in this dataset, whatever share of ECGs is answered.

## 2c. Doctor-style hierarchical classification with refer zones (Amendment 4, E8; header-free)
**Design:**
- three image-only specialists (abnormal vs Normal; Severe vs Mild + Moderate; Mild vs Moderate) on ECGFounder f6 + the
  121 image clinician features + the 150 ECGFounder diagnostic outputs;
- a refer zone at each step, so neighbouring classes never overlap;
- thresholds chosen in the training folds to answer as many patients as possible under accuracy and Mild / Severe
  requirements.

| Option | Answered | Accuracy on answered [95 % CI] | Mild caught (of all Mild) | Mild precision | Severe recall, answered / of all Severe |
|---|---|---|---|---|---|
| No refer zone (all patients) | 100 % | 0.544 [0.514, 0.573] | 19 % | 0.25 | 0.33 / 0.33 |
| Accuracy ≥ 0.65 | 70 % | 0.650 [0.615, 0.686] | 0 % | – | 0.54 / 0.26 |
| Accuracy ≥ 0.70 | 44 % | 0.677 [0.623, 0.739] | 0 % | – | 0.63 / 0.20 |
| Accuracy ≥ 0.65 and Mild ≥ 5 % | 47 % | 0.633 [0.572, 0.700] | 3 % (3 folds infeasible) | 0.26 | 0.62 / 0.21 |
| Accuracy ≥ 0.65 and Mild ≥ 10 % | 5 % | (14 of 15 folds infeasible) | 0 % | – | – |
| Accuracy ≥ 0.70 and Mild ≥ 5 or 10 % | 0 % | infeasible in all 15 folds | – | – | – |

**Findings:**
- **Severe:** the step-wise design detects Severe much better among the answered patients (recall 0.54–0.63, vs 0.17
  for the triage).
- **Mild:** it does not help. The Mild-vs-Moderate step is close to chance, matching the factor study (AUROC ≤ 0.63)
  and the other session's HIER model (stage-2 accuracy 0.53–0.58).
- **All patients:** without refer zones, the hierarchy (0.544) does not beat IMG-C3 (0.558) or HIER-ext (0.570).

## 3. Why Mild separation did not improve
- **Mild is spread across all four predictions.** In the C3 confusion matrix, true Mild goes to Normal 43 %, Mild 18 %,
  Moderate 28 % and Severe 11 %. Every Phase 7 representation reproduces the same pattern (38–47 % to Normal).
- **The ECG carries little Mild-specific information at this sample size and label quality.**
  - Pairwise AUROC: Normal-vs-Mild about 0.75; Mild-vs-Moderate about 0.6.
  - The ECG–echo EF error is about 13 points SD (11.8 on clean links), against a Mild band 12 points wide.
  - Richer representations, EF supervision, uncertainty-aware labels and contrastive pulling all redistribute the same
    weak signal. None adds information.
- **More data of the same kind does not fix Mild.** In the learning curve, Mild F1 is flat while accuracy rises
  slowly (next section).

## 4. Learning curve and what additional data are needed
The reference member and the ridge EF regressor were retrained on 25 / 50 / 75 / 100 % of each fold's training
patients (nested subsamples; inner-val and test folds unchanged):

| Training ECGs per fold | Accuracy | Macro-F1 | Mild F1 | EF SD, all ECGs | EF SD, verified-EF ECGs |
|---|---|---|---|---|---|
| 134 | 0.515 | 0.415 | 0.204 | 14.39 | 13.39 |
| 267 | 0.527 | 0.421 | 0.151 | 13.92 | 12.58 |
| 401 | 0.537 | 0.439 | 0.188 | 13.64 | 12.27 |
| 536 | 0.544 | 0.436 | 0.179 | 13.55 | 12.08 |

**Per doubling of training data:** accuracy +1.4 points, macro-F1 +1.2, Mild F1 −0.9 (no trend). EF SD falls by 0.44
on all ECGs and by 0.66 on verified-EF ECGs.

**Indicative extrapolation** (log-linear from 4 points over 2 doublings; uncertain far out):

| Scenario | Accuracy | EF SD |
|---|---|---|
| ≈ 100,000 training ECGs, current label quality | ≈ 0.65 | ≈ 10.3 |
| ≈ 100,000 training ECGs, verified-quality EF targets | — | ≈ 7.1, below the ≈ 9 needed for 70 % |
| 70 % at current label quality | needs ≈ 1 million ECGs | — |
| EF SD 9 with verified-quality labels | — | needs ≈ 13,000 training ECGs |

**Conclusion: label quality matters more than quantity.** The phase-2 dataset should provide, in order of expected
value:
1. **Verified ECG–echo linkage.** Link by patient identifier (not names), with the echo within a few days (ideally the
   same admission) of the ECG. In this dataset only 366 / 837 ECGs (44 %) have a clean link, and echoes > 30 days away
   correlate with the ECG at r 0.42 vs 0.75 for clean links.
2. **A better EF reference.** Biplane Simpson EF (guideline method), not Teichholz M-mode. Repeat measurements or
   core-lab reads near the 41 % and 50–54 % boundaries, where the Mild band lies.
3. **Raw digital ECGs** (e.g. vendor XML / DICOM-ECG at 500 Hz, all 12 leads simultaneous for 10 s) instead of
   PDF-digitised traces. Today each lead has only 2.5 s, not simultaneous with the others. ECGFounder was pre-trained
   on simultaneous 10-s ECGs.
4. **Volume:** about 13,000+ ECGs with verified labels, with Mild deliberately well represented. With 100,000 patients
   of verified quality, the learning curve indicates that the EF error could fall below the ≈ 9 points needed for
   70 %. That is a hypothesis to test, not a result.
5. **Keep the protocol:** a pre-registered primary task on the original four classes, patient-grouped splits, and the
   locked test cohort scored once at the end.

## 5. Data quality (summary of Phases 4–6B; the original dataset unchanged)
- **Matching:**
  - every development ECG has ≥ 1 linked echo;
  - the folder label agrees with the nearest echo for 93.4 % (55 inconsistent);
  - 85 ECGs have a linked echo of another class;
  - 77 have an EF spread > 10 points across linked echoes;
  - 391 have a name-match score < 100; 192 have the nearest echo > 30 days away.
- **Borderline EF:** 221 ECGs lie within 3 points of a class boundary.
- **Label cleaning, relabelling and noise-free training** (VER, BRES, ECHO, CLEAR, MMC / MMNC / MN240 / MN200,
  NF / NFS) change the C3 recipe's accuracy on all 837 ECGs by −0.021 to +0.005. Label noise is real but explains
  little of the gap.
- **Raw digital ECG:** no access is recorded (`DATA_ACCESS.md`). All waveform work uses traces digitised from the
  printed PDFs.

## 6. Reproducibility
- **Code:**
  - `p7common.py`;
  - `e1_ecgf.py` (`--pilot`), `e2_heads.py` (`--pilot`), `e3_cnn.py` (`--pilot`);
  - `e5_curve.py`;
  - `report7.py` (CSV, statistics, plots, inventory).
- **Inputs:** the existing ECGFounder cache (Phase 6), ECGFounder f6 (Phase 2), header and clinical features, and
  `data/signals.npz` (development rows only).
- **Outputs:**
  - aggregates in `results/` and `results/experiments/`;
  - per-ECG predictions in the git-ignored `data/private/`.
- **Pre-commit checks:** `tests/test_phase7.py` (pre-registration present, write guard, no recurrent layers, no ECG or
  patient identifier in tracked files), plus the Phase 6 / 6A / 5 suites and `integrity6 verify phase6_before`.
- **Integrity guard:** `phase_7` was added to the additive-work exclusions in `phase_6/integrity6.py`, as was done
  for 6a / 6b. Phase 1–5 files and the data stay fully checked.

## 7. Frozen test kit (Amendment 5; branch `research/phase7-testkit`)
- **What:** the header-free triage model (56 % answered at 0.694 in CV) frozen on all 837 development ECGs, plus a
  one-shot scorer for a test set the owner collects. Instructions: `testkit/TESTKIT.md`. Card (every chosen value and
  the file hash): `results/testkit_frozen_card.json`.
- **Frozen values:**
  - EF estimator H3 W.3 (pooled outer-OOF RMSE 12.2 on verified ECGs);
  - cuts: Normal above the sex boundary, Mild ≥ 45.5, Moderate ≥ 35;
  - answer threshold 3.05 EF points;
  - prior-rule α 0.5.
- **On the data used to choose these values:** 55 % answered at 0.700. This is not an estimate. The CV estimate stays
  0.694 [0.655, 0.736] on 56 % [48, 63].
- **Checks (development data only):**
  - members reproduce the committed IMG-vote OOF (100 % same class), and the EF code reproduces E6 (≤ 10⁻¹³);
  - 25 development PDFs read from scratch give identical image inputs and identical decisions;
  - a 40-ECG pipeline test ran end to end, and all 40 were detected as development duplicates.
- **Refusals:** development ECGs; locked-test ECGs without `--locked-test-authorised`; a re-scored manifest without a
  logged reason; a changed frozen file.
- **Not done:** no test data has been scored, and the locked test is untouched.

## 8. Locked-test result (Amendment 6; owner-authorised, scored once on 2026-11-06)
The frozen header-free model (section 7, SHA-256 `09808d12…`, trained on all 837 development ECGs) was run once on
the locked test set. The rules were committed before any test ECG was read (commit c6bb2f0).
- **Primary set:** 199 of the 210 test ECGs. 11 were left out because their patient key forms a near-identical name
  pair with a development patient. No test ECG was a near-duplicate of a development ECG, and none was unreadable.
- **Clusters:** 186 patients (3 within-test name pairs merged).

**Final four-class accuracy (IMG-C3, every primary test ECG):**

| Metric | Locked test (95 % CI, patient-cluster bootstrap) | CV expectation |
|---|---|---|
| **Accuracy** | **0.553 [0.485, 0.621]** (Wilson [0.483, 0.620]) | 0.558 [0.504, 0.611] |
| Macro-F1 | 0.453 [0.375, 0.525] | 0.441 |
| Balanced accuracy | 0.441 [0.372, 0.514] | 0.445 |
| Mild precision / recall / F1 | 0.32 / 0.32 [0.18, 0.46] / 0.32 | 0.29 / 0.19 / 0.23 |

| True class (n) | Precision | Recall | F1 |
|---|---|---|---|
| Normal (95) | 0.70 | 0.80 | 0.75 |
| Mild (41) | 0.32 | 0.32 | 0.32 |
| Moderate (43) | 0.39 | 0.35 | 0.37 |
| Severe (20) | 0.50 | 0.30 | 0.37 |

Confusion (rows true Normal / Mild / Moderate / Severe; columns predicted): `[[76, 11, 6, 2], [13, 13, 13, 2],
[13, 13, 15, 2], [6, 4, 4, 6]]`.

**Triage on the locked test:**
- **Answered:** 49.7 % [42.9, 56.7] of patients, with accuracy **0.747 [0.660, 0.837]** on them (74 of 99). The CV
  expectation was 56 % at 0.694.
- **Calls:** 88 Normal, 10 Moderate and 1 Severe; no Mild call.
- **Confident "Normal" answers:**
  - 85 % [77, 92] had no moderate or severe dysfunction (EF ≥ 41 %);
  - 21 % [11, 31] of all EF < 41 % patients received a confident "Normal" (CV: 15 %);
  - 24 % of Mild patients were called Normal.

**Sensitivity, all 210 test ECGs:** four-class accuracy 0.538 [0.472, 0.607]; triage 48 % answered at 0.743.

**Reading:**
- Every CV expectation lies inside its locked-test CI, so the development estimates carried over to unseen patients.
- The four-class accuracy is about 0.55, and 70 % is not reached on all patients.
- The triage point estimate (0.75 on 50 %) is higher than in CV, but its CI includes 0.66–0.70, so the honest
  statement is "about 0.70–0.75 on about half the patients".
- The triage still gives no Mild answers.
