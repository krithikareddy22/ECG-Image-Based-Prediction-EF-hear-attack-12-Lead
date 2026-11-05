# Phase 6B plan and pre-registration: EF thresholds, Normal training size and LSTM models (four-class)

**Status:** fixed before any Phase 6B model run (owner directions, 2026-11-05). SHA-256 of the text above
`## Amendments` in `configs/PLAN.sha256`; later changes only as dated amendments.

- **Objective (owner):** the best scientifically valid **four-class** LVEF accuracy without sacrificing detection of
  Mild, Moderate and Severe. No binary reformulation. 60 % is investigated, not manufactured.
- **Primary baseline:** original EF thresholds, original dataset, original labels. Every alternative threshold or
  class-distribution experiment is a **separate controlled sensitivity analysis**.
- **Scope / re-use:** extends Phase 6 and Phase 6A on branch `research/phase6`; completed 6A runs are re-used, not
  repeated (B1 = ORIG, B2 = T50, B3 = XUNC, A270 = A270X; LSTM L0–L4 on original labels). Development set only
  (837 ECGs / 794 keys); the locked test is not loaded, predicted or scored; folds `phase_2/configs/folds_D.csv`.

## 1. Audit (step 1; no model)
`PHASE6B_AUDIT.md` compiles, from result files only: the exact labelling rule (recovered matching-notebook code),
EF provenance (Teichholz from fuzzy name-matched echoes; not verified), class / patient / fold distributions,
patient grouping, matching quality (owner-supplied sheets; 2021 ECG-date swap, C6A-1), borderline cases, repeated
patients and multiple links. Borderline folder-Mild ECGs (nearest-echo EF within 3 points of the sex-specific
Normal/Mild boundary) are separated into, in this order of precedence:
1. **matching uncertainty:** name score < 100, or |ECG–echo| > 30 days (true acquisition dates), or a linked echo of
   another class;
2. **potentially unreliable EF:** EF within ± 2 points of the boundary (measurement band), or EF spread > 10 points
   across links;
3. **high-confidence borderline:** none of the above (the best available proxy for "verified"; no EF is verified);
4. **genuine but difficult ECG pattern (descriptive):** high-confidence borderline ECGs misclassified by the C3 model in
   all three repeats. Used for description only; never for training or evaluation choices.

## 2. Configurations
**Label definitions (four classes).** Each is trained on its own labels and scored on its own labels **and** on the
original labels; definitions are reported separately and never claimed to improve the original task.

| ID | Normal/Mild boundary | Justification | Relabelled |
|---|---|---|---|
| ORIG | Normal M > 52 / F ≥ 54 % (Mild ≥ 41, Moderate 31–< 41, Severe < 31) | original rule; ASE/EACVI 2015 sex-specific normal ranges | – |
| T50 | ≥ 50 %, sex-independent | heart-failure classifications: preserved EF ≥ 50 %, HFmrEF 41–49 % (Universal Definition 2021; ESC 2021; AHA/ACC/HFSA 2022) | 27 Mild → Normal (consistent links only) |
| T55 | ≥ 55 %, sex-independent | British Society of Echocardiography 2020: normal ≥ 55 %, 50–54 % "borderline low" | 10 Normal → Mild (consistent links only) |

Only consistently linked ECGs move (folder label equals the class of the nearest echo under the original rule), so
link ambiguity is not propagated. Caveat recorded: guideline ranges refer to biplane (Simpson) EF; these labels are
Teichholz (M-mode), so no threshold here is claimed to be clinically validated for this dataset.

**Training-only configurations (original labels; validation and evaluation populations identical).**

| ID | Change to the fitting rows only |
|---|---|
| XUNC | uncertain Normal/Mild borderline ECGs removed (45 ECGs; Phase 6A B3) |
| N300 / N270 / N240 / N200 | Normal ECGs in the outer-training portion (inner-train + inner-val; 307–308, the CV analogue of the original 669-ECG training split with 307 Normal) reduced to the target by removing whole Normal patients from inner-train (7–8 / 37–38 / 67–68 / 107–108 ECGs); seeded order per fold (`default_rng(6200 + 10 r + k)`), nested across targets; inner-val (early stopping / selection) and outer test unchanged |
| A270X | Phase 6A A270 (inner-train Normal expanded to 270 from inner-val; changes inner-val) - reported, not re-run |

"Original training distribution" = 307–308 Normal (ORIG). All requested sizes are feasible without new data,
duplication or touching validation / evaluation. Removed rows are excluded from every fit (outer members, nested inner
models, decision-rule and prior fitting).

## 3. Models
| ID | Model | Notes |
|---|---|---|
| EXIST | MLP fusion (3 seeds), C1 vote recipe, C3 recipe (vote + nested prior rule) | every configuration |
| L0–L4 | standalone LSTMs (Phase 6A §5: rhythm / measured-aligned 26 × 1,000 @ 100 Hz with masks; uni / multi-layer / bidirectional) | ORIG; re-used from 6A |
| LSTM-sel | nested inner-val selection over L0–L4 per outer fold | primary standalone LSTM estimate |
| LSTM-fixed | variant chosen most often by the nested selection | every configuration |
| L5 CNN-LSTM | 3 × Conv1d(k7, s2) → BiLSTM-64 (6A §5) | ORIG; now unconditional (owner) |
| L6 CNN-LSTM + header | L5 encoder; pooled features ‖ standardised header numeric (16) → head | ORIG |
| F1 ECGF + LSTM (early) | LSTM-fixed encoder pooled features ‖ header ‖ ECGFounder f6 (block-standardised on inner-train) → Phase 2 MLP-fusion body → 4 classes, trained jointly | ORIG |
| F2 ECGF + CNN-LSTM (early) | same with the L5 encoder | ORIG |
| F3 vote + LSTM (late) | C1 members + LSTM-sel, equal-weight vote | ORIG |
| F4 F3 + nested prior rule | C3 rule; LSTM inner OOF with LSTM-fixed, seed 42 | ORIG |

LSTM / fusion training as in Phase 6A §5 (class-balanced CE, AdamW 1e-3, batch 32, clip 1.0, ≤ 60 epochs, inner-val S3
early stopping, patience 12). Seeds 42 / 43 / 44 × repeats 0–2. **Compute rule:** LSTM-fixed under the eight
sensitivity configurations uses 3 seeds if the estimated total is ≤ 10 CPU-hours (wall clock with 4 parallel jobs);
otherwise seed 42 only, compared with the seed-42 ORIG run (labelled; not eligible for selection).
Previously unsuccessful combinations are not repeated (gradient boosting, stacking variants, fine-tuning, ECG-FM).

## 4. Evaluation
- **Primary:** four-class accuracy on all 837 ECGs against the **original labels** (pooled outer folds, seeds combined,
  mean over 3 repeats), subject to admissibility (every class recall ≥ 0.15) and eligibility (3 repeats × 3 seeds).
- **Alternative definitions** additionally scored on their own labels, with the definition / training decomposition
  (Phase 6A §4).
- **Reported:** accuracy, balanced accuracy, macro-F1, per-class precision / recall / F1, confusion matrices, QWK,
  corrected 95 % CIs, training / inference time, parameters.
- **Tests:** corrected repeated-CV t-test on the 15 shared cells; Holm within each family per metric
  (a: LSTM-family vs the existing models; b: each configuration vs ORIG for the same model, original labels;
  c: alternative definitions on their own labels vs ORIG re-scored on those labels); cluster bootstrap and McNemar
  for the top candidates. **Genuine improvement:** Holm p < 0.05, same sign in all 3 repeats, admissible.
- **Two conclusions:** (1) best admissible eligible model under ORIG labels; (2) best under a clinically justified
  alternative definition (T50 or T55), scored on that definition, explicitly not comparable to (1).

## 5. Privacy and storage
Per-ECG data, predictions, checkpoints and logs in the git-ignored `data/private/`; committed outputs are
aggregates (counts 1–4 suppressed in audit tables).

## Amendments

### Amendment 1 (2026-11-05, before any Phase 6B model run): operational details fixed before results
Written on the owner's GPU laptop before any 6B model was trained or any LSTM result was seen; nothing above is changed.
- **Where runs come from.** ORIG / T50 / XUNC / A270X for EXIST and LSTM-fixed are the Phase 6A runs under B1 / B2 / B3 /
  A270 (6A code). T55 and N300-N200 are 6B runs (`existing6b.py`, `lstm6b.py --models FIX`). L5, L6, F1, F2, F3, F4 are
  6B runs under ORIG (`lstm6b.py`), unconditional (owner); Phase 6A's own conditional rules for CNN-LSTM and fusion are
  still evaluated and reported in Phase 6A. GPU training (CUDA, deterministic cuBLAS) for every LSTM-family model.
- **Static inputs.** L6: header numeric (16; medians and z-scores from inner-train). F1 / F2: `make_X(["hdr", "ecgf"])`
  (Phase 2 block standardisation, statistics from inner-train) concatenated with the encoder's pooled features
  [last hidden || masked mean]; body = Phase 2 MLP-fusion body (Dropout 0.5 -> Linear 64 -> GELU -> Dropout 0.5) ->
  Linear(4). F3 members are the 6A B1 runs (MLP fusion seed-mean, ECGF linear distance-CE, header + clinical LR) and
  LSTM-sel seed-mean; F4's LSTM inner OOF uses the 6A `fusion.lstm_inner` protocol (splits, seed keys) on the GPU.
- **Compute rule.** The 10-hour threshold is applied to the measured wall clock of this machine: estimate = 8 configs x
  45 fold fits x mean LSTM-fixed fold time under ORIG / number of parallel GPU workers (`lstm6b.py --estimate`,
  recorded in `results/compute_rule.json` before the configuration runs start).
- **Eligibility.** 3 repeats and 3 seeds (seed-combined), or an ensemble / decision rule whose members are; deterministic
  models (ECGF linear, header + clinical LR) count as eligible. Single-seed models (ECGFounder fine-tuned A1; LSTM-fixed
  seed 42 only, if the compute rule requires it) are reported but not eligible for selection.
- **Test families** (corrected repeated-CV t-test on the 15 shared (repeat, fold) cells; Holm within each family and
  metric; metrics: accuracy (primary), macro-F1, Mild F1, Mild recall):
  (a) each of LSTM-sel, LSTM-fixed, L5, L6, F1, F2, F3, F4 (ORIG) vs C1 (Phase 3 vote) and vs C3 (Phase 5 selected);
  (b) each of T50, T55, XUNC, A270X, N300, N270, N240, N200 vs ORIG for MLP fusion, C1 recipe, C3 recipe and LSTM-fixed,
      on the original labels;
  (c) T50 and T55 vs ORIG for the same four models, both scored on the alternative definition's labels; reported with
      the definition effect (ORIG model re-scored on the alternative labels minus on the original labels) and the
      training effect (alternative-trained minus ORIG-trained, both on the alternative labels).
  Cluster bootstrap (2,000 patient resamples, repeat 0) and exact McNemar (repeat 0) for the three largest accuracy
  gains of each family.
- **Conclusions.** (1) highest mean accuracy on the original labels among admissible, eligible models trained on the
  original labels (ORIG, XUNC, A270X, N-sizes), ties by macro-F1; (2) the same among T50- and T55-trained models scored
  on their own labels (admissibility on those labels), explicitly not comparable with (1). "60 %" is reported as the
  mean accuracy and its corrected 95 % CI; it is called reached only if the mean is >= 0.60, and robustly only if the
  CI's lower bound is >= 0.60.
