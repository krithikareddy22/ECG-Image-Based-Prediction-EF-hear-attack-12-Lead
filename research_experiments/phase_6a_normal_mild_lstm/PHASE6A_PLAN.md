# Phase 6A plan and pre-registration: Normal/Mild boundary and LSTM study

**Status:** fixed before any Phase 6A model run (faculty-guide directions, 2026-11-05). SHA-256 of the text above
`## Amendments` in `configs/PLAN.sha256`; later changes only as dated amendments.

- **Branch:** `research/phase6` (Phase 6A is a controlled sub-study of Phase 6). Additive only: no Phase 1–5 file and
  no Phase 6 result is modified; `phase_6/integrity6` covers both directories.
- **Population:** development set D only (837 ECGs / 794 name-derived patient keys). The 210-ECG locked test is not
  loaded, predicted, scored or inspected; no locked-test PDF or waveform enters any model.
- **Folds:** `phase_2/configs/folds_D.csv` (patient-grouped StratifiedGroupKFold(5), repeats 0–2, inner-train /
  inner-val), identical for every model; nested inner CV as in `phase_5/experiments/nested.py`.

## 1. Label definition (inspected, not assumed)
**Source:** `classify_lv` in the original matching notebook (code cells only; recovered from git history under the
owner's Phase 4 authorisation) and `ecg_ef/config.py`; identical to the Phase 4 audit. EF = Teichholz.

| Class | Male | Female |
|---|---|---|
| Normal | EF > 52 | EF ≥ 54 |
| Mild | 41 ≤ EF ≤ 52 | 41 ≤ EF < 54 |
| Moderate | 31 ≤ EF < 41 | 31 ≤ EF < 41 |
| Severe | EF < 31 | EF < 31 |

**Borderline assignment:** male 52.0 → Mild, 52.1 → Normal; female 53.9 → Mild, 54.0 → Normal; 41.0 → Mild,
40.9 → Moderate; 31.0 → Moderate, 30.9 → Severe. Unknown sex with EF ≥ 41 → "Unknown" (no D ECG is affected: sex is
known for all 837). Folder labels were made from fuzzy **name** matches between ECG and echo (no patient ID, no date
window); the continuous EF used below is the Phase 4 label-agnostic **nearest-in-time linked echo** (not verified).

## 2. Cohort facts (development set; computed before this plan)
| Set | ECGs N / Mi / Mo / S | Patient keys N / Mi / Mo / S |
|---|---|---|
| Original fixed train split | 669: 307 / 148 / 141 / 73 | 635: 305 / 145 / 138 / 68 |
| Original fixed validation split | 168: 77 / 37 / 36 / 18 | 159: 77 / 35 / 34 / 18 |
| Development set D (train + val) | 837: 384 / 185 / 177 / 91 | 794: 382 / 180 / 172 / 86 |
| CV inner-train per outer fold (fitting set) | ≈ 535: 245–247 / 118–119 / 112–114 / 58–59 | – |
| CV inner-val per outer fold | ≈ 134: 61–62 / 29–30 / 28–29 / 14–15 | – |
| CV outer-training portion (inner-train + inner-val) | ≈ 670: 307–308 / 148 / 141–142 / 72–73 | – |
| CV outer test fold | ≈ 167, of which 76–77 Normal | – |

25 patient keys have ECGs in more than one class. The locked test (96 / 47 / 44 / 23) is not used.

## 3. Experiment A: Normal training set of 270
- **Discrepancy (recorded):** no project file specifies 270. The original fixed training split already contains
  **307** Normal ECGs (≥ 270), so "expanding to 270" cannot refer to it, and reducing it to 270 would discard valid
  data; this is **not** done. In the CV protocol every model is fitted on the inner-train part, which contains only
  **245–247** Normal ECGs; 270 is attainable there without new data, duplication or discarding.
- **A270 construction (per repeat r and outer fold k):** inner-train Normal is raised to exactly 270 ECGs by moving
  whole Normal **patients** from the same fold's inner-val into inner-train: inner-val Normal patients are visited in
  a random order (NumPy `default_rng(6270 + 10 r + k)`); a patient is moved if the count stays ≤ 270; stop at 270.
  Non-Normal ECGs, the outer test fold, labels and the dataset are unchanged; no patient is in both inner-train and
  inner-val (asserted). Side effect, reported: inner-val Normal shrinks from ≈ 61 to ≈ 37 ECGs.
- **Models:** MLP fusion (3 seeds), the C1 vote (members retrained), the C3 rule on that vote (its nested inner OOF
  is that of B1, because the nested inner models do not use the inner-train / inner-val split), and LSTM-fixed (§5).
- **Comparator:** the same models on the original split (B1, same code).

## 4. Experiment B: Normal/Mild boundary label configurations
- **B1 (primary baseline):** original folder labels.
- **B2 (boundary sensitivity):** sex-independent Normal/Mild boundary at **EF 50 %**. Clinical justification:
  "preserved" LVEF is ≥ 50 % in the heart-failure classifications (Universal Definition of HF 2021; ESC HF guideline
  2021; AHA/ACC/HFSA 2022: HFmrEF 41–49 %, HFpEF ≥ 50 %), and the British Society of Echocardiography (2020) calls
  50–54 % "borderline low". Under B2 the Mild class equals the HFmrEF range (41–49 %). Only **clearly defined**
  borderline Mild ECGs move to Normal: folder label Mild **and** nearest-echo EF ≥ 50 **and** the nearest echo itself
  in the Mild class under the original rule (consistent link). Expected: 27 ECGs / 27 patient keys (sizing count).
  No other label changes. B2 is a sensitivity analysis; **no clinical validity is claimed** for the four-class
  scheme with this boundary, and the boundary was fixed on clinical grounds before any model was trained with it.
- **B3 (exclusion sensitivity):** original labels; "uncertain borderline" ECGs are removed **from inner-train only**
  (inner-val, outer test and every evaluation keep all 837 ECGs). Uncertain = folder label Normal or Mild **and**
  (nearest-echo EF within ± 2.0 points of the patient's sex-specific Normal/Mild boundary (52 M / 54 F), **or** a
  linked echo of the other class among {Normal, Mild}). The ± 2-point band follows the Phase 4 Mild analysis.
- **Borderline audit (descriptive):** for Mild ECGs near the boundary: name-match score, ECG–echo interval, number of
  links and EF spread, high-confidence cohort G membership, patient keys with repeated ECGs; whether the EF is
  verified (it is not: name matching only). Counts of ECGs / patients affected by alternative boundaries
  (sex-independent 48–55 %, and the original sex-specific rule shifted by −4…+2 points) are tabulated and plotted.
  These counts never use model output.
- **Evaluation of label configurations:**
  - every model is scored against the **original labels** on all 837 ECGs (primary; this is the only scale on which
    accuracy is comparable across configurations);
  - B2 models are also scored against the **B2 labels**; the change is decomposed into a **definition effect**
    (B1 model re-scored with B2 labels − B1 model with original labels) and a **training effect** (B2 model − B1
    model, both scored with B2 labels);
  - a model's misclassifications are never treated as label errors.
- **Models:** MLP fusion, C1 vote, C3 (vote + nested prior rule; nested inner OOF re-fitted with the configuration's
  training labels / training rows), LSTM-fixed.

## 5. Experiment C: standalone LSTM (no ECGFounder, no ResNet)
- **Input data:** digitised printout waveforms, 500 Hz: 12 leads × 2.5 s (each lead in its own printed column:
  I/II/III 0–2.5 s, aVR/aVL/aVF 2.5–5 s, V1–V3 5–7.5 s, V4–V6 7.5–10 s) and the 10-s lead II rhythm strip
  (simultaneous with all columns). Preprocessing: ECGFounder band-pass (0.67–40 Hz, 50 Hz notch, median baseline
  removal) per segment, then decimation 500 → 100 Hz; amplitudes kept in mV (no per-ECG normalisation), clipped at
  ± 5 mV.
- **No fabrication:** each lead is placed only in its recorded 2.5-s window on a common 10-s time axis and is zero
  elsewhere, with a **mask channel** (1 = recorded, 0 = not recorded). No lead is derived or copied into windows where
  it was not recorded. Missing leads (≥ 90 % flat after digitisation; 3 of 10,044 segments) and flat runs ≥ 20 ms
  get mask 0.
- **Inputs:** `rhythm` = lead II strip only (1 × 1,000); `aligned` = 13 signals (12 leads + rhythm II) + 13 masks
  (26 × 1,000).
- **Variants** (one change at a time):

  | ID | Input | Layers | Direction | Hidden | Inter-layer dropout |
  |---|---|---|---|---|---|
  | L0 | rhythm | 1 | uni | 64 | – |
  | L1 | aligned | 1 | uni | 64 | – |
  | L2 | aligned | 2 | uni | 64 | 0 |
  | L3 | aligned | 1 | bi | 64 | – |
  | L4 | aligned | 2 | bi | 64 | 0.3 |

- **Head:** [last hidden state (both directions) ‖ mask-weighted mean of outputs] → Dropout 0.3 → Linear(→ 4).
- **Training:** class-balanced cross-entropy (Phase 2 weights), AdamW lr 1e-3, weight decay 1e-2, batch 32, gradient
  clipping 1.0, ≤ 60 epochs; early stopping on the fold's inner-val S3 (Phase 2 selection score), patience 12, no
  stop before epoch 10; best inner-val epoch restored. Seeds 42 / 43 / 44 × repeats 0–2 (9 runs × 5 folds).
- **Primary LSTM estimate (nested selection, "LSTM-sel"):** in each (repeat, outer fold), the variant with the highest
  mean inner-val S3 over the three seeds is chosen; its outer predictions are used. Selection never sees the outer
  fold.
- **LSTM-fixed (for A, B2, B3):** the variant chosen most often by the nested selection over the 15 B1 folds (ties:
  fewer parameters). Fixed before any A / B2 / B3 LSTM run.
- **CNN-LSTM (conditional):** run only if LSTM-sel is admissible (every class recall ≥ 0.15) **and** its macro-F1
  ≥ 0.389 (the original project's header + ResNet-18 default). Architecture: 3 × [Conv1d k7 stride 2 → GELU]
  (26 → 32 → 64 → 64 channels; 1,000 → 125 steps) → 1-layer BiLSTM-64 → same head and training.
- **ECGFounder–LSTM fusion (conditional on complementarity):** complementarity = adding LSTM-sel as a fourth member
  to the C1 vote raises accuracy **or** macro-F1 in all three repeats. If met: (i) vote of C1 members + LSTM-sel and
  (ii) the same with the C3 nested prior-protected rule (LSTM inner OOF: seed 42, LSTM-fixed). Error-overlap
  statistics (LSTM correct where MLP fusion is wrong) are always reported.

## 6. Experiment E: comparison with existing models (original labels; stored predictions, no retraining)
Header + ResNet-18 (repeat 0 only; descriptive), ECGFounder frozen linear, ECGFounder fine-tuned (Phase 6 A1,
seed 42 × 3 repeats), header + ECGFounder MLP fusion, C1 vote, C2 calibrated MLP, C3 Phase 5 selected model.

## 7. Metrics and statistics
- **Primary:** four-class accuracy on all 837 ECGs (pooled outer folds, seeds combined by probability mean, mean over
  repeats 0–2), subject to admissibility (every class recall ≥ 0.15). Also: balanced accuracy, macro-F1, per-class
  precision / recall / F1 (Mild F1 and recall highlighted), QWK, macro AUROC, confusion matrices, training and
  inference time, parameter counts.
- **Same-code baseline check:** B1 retrained by Phase 6A code must reproduce the stored MLP fusion, C1 and C3 within
  0.005 (accuracy, balanced accuracy, macro-F1); otherwise stop and report.
- **Tests:** corrected repeated-CV t-test (N_train 535 / N_test 167) on the 15 matched (repeat, fold) cells; Holm over
  the Phase 6A confirmatory family (LSTM-sel vs the six 3-repeat existing models; A270 vs B1, B2 vs B1, B3 vs B1 for
  MLP fusion, C1, C3, LSTM-fixed; conditional CNN-LSTM vs LSTM-sel and fusions vs C1 / C3), separately for accuracy
  (primary), macro-F1 and Mild F1; patient-cluster bootstrap (2,000, repeat 0) and exact McNemar for the top
  candidates. **Genuine improvement:** Holm p < 0.05, same sign in all 3 repeats, admissible.
- Decision rules / thresholds are chosen only inside training folds (nested); no single repeat or seed is selected.

## 8. Privacy, storage, reproducibility
- Per-ECG labels, EF, link data, predictions and model checkpoints stay in the git-ignored
  `phase_6a_normal_mild_lstm/data/private/`; committed: code, configs, aggregates, plots, reports.
- Run ids `P6A-<model>-<config>-v1__s<seed>_r<repeat>`; aggregate JSON + private OOF with SHA-256; fold-level
  caches make every run resumable; completed runs are never overwritten.

## Amendments

### Amendment 1 (2026-11-05, before any LSTM run): definition of missing samples in the LSTM input
- **Reason:** building the inputs showed that the pre-registered rule "flat runs ≥ 20 ms get mask 0" marks ≈ 40 % of
  samples as missing, including ≈ 39 % of the 10-s rhythm strip. PDF vector digitisation draws straight segments
  (e.g. the baseline) as repeated values (the same artefact Phase 5 removed from its signal-quality flag, sq v2), so
  interior constant runs are recorded signal, not missing data. Found while building the inputs, before any LSTM
  was trained; no result informed the change.
- **Change:** mask 0 only for (i) leads that are ≥ 90 % flat (missing lead) and (ii) constant runs ≥ 20 ms at the
  start or end of a segment (undrawn part of the printed trace, analogous to the Phase 2 polyline-gap masking).
  Interior constant runs keep mask 1. Everything else in §5 is unchanged.

### Amendment 2 (2026-11-05, execution record; no change to any analysis rule)
- All LSTM-family models were trained on the owner's laptop GPU (CUDA, deterministic cuBLAS); the cloud's partial,
  uncommitted LSTM runs were discarded and the grid was run once.
- The conditional CNN-LSTM and the vote + LSTM fusions were executed by Phase 6B (identical definitions, unconditional
  by the owner's direction). `compare.py` reads those runs and adds them to the Phase 6A confirmatory family only if the
  §5 conditions hold; they did not, so they are reported descriptively.
- Private predictions of committed runs are regenerated within a 0.005 tolerance (records in `results/regeneration/`).
