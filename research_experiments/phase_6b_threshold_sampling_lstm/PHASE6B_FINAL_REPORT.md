# Phase 6B final report: EF thresholds, Normal training size and LSTM models (four-class LVEF)

**Objective (owner):** the best scientifically valid four-class LVEF accuracy without sacrificing Mild, Moderate and
Severe detection; no binary reformulation; 60 % is investigated, not manufactured.

**Protocol:** development set only (837 ECGs / 794 patient keys; patient-grouped 5-fold CV × 3 repeats). The 210-ECG
locked test was never loaded, predicted, scored or inspected. Pre-registration is `PHASE6B_PLAN.md`; Amendment 1 fixed
the operational details (test families, eligibility, 60 % criterion, compute rule) before any 6B LSTM-family result.
Accuracy is pooled over the outer folds, seed-combined and averaged over the 3 repeats. 95 % CIs are corrected
repeated-CV intervals over the 15 (repeat, fold) cells. Admissible means every class recall is ≥ 0.15; eligible means
3 repeats × 3 seeds (or an ensemble / rule built from such members, or a deterministic model).

**Outputs:**
- `PHASE6B_AUDIT.md`: section 1 audit.
- `PHASE6B_MODEL_COMPARISON.csv`: 54 models.
- `results/phase6b_results_per_repeat.csv`: results CSV.
- `results/phase6b_comparisons.csv`: 224 paired tests.
- `results/phase6b_summary.json`: conclusions, decomposition, bootstrap / McNemar.
- `results/compute_rule.json`.
- `plots/`: `accuracy_comparison.png`, `macro_f1_comparison.png`, `per_class_precision_recall_f1.png`,
  `confusion_matrices.png`, `threshold_sensitivity.png`, `normal_size_curve.png`, `architecture_lstm_cnn_fusion.png`.

## 1. Audit (summary of `PHASE6B_AUDIT.md`)
- **Labels:** sex-specific rule (Normal M > 52 / F ≥ 54, Mild ≥ 41, Moderate 31–< 41, Severe < 31). EF is Teichholz
  from fuzzy **name-matched** echo reports, and no EF is verified.
- **Classes:** 384 / 185 / 177 / 91 ECGs (Normal / Mild / Moderate / Severe).
- **Borderline Mild** (EF within 3 points of the boundary): 30 ECGs. Of these, 15 have matching uncertainty, 10 a
  potentially unreliable EF, 5 are high-confidence, and "< 5" are genuinely difficult (descriptive only).
- **Link inconsistencies:** 10 further Mild ECGs have a nearest echo far above the boundary (class Normal).
- **2021 ECG-date swap (C6A-1)** changes intervals only, not labels or configurations.

## 2. What was run
| Group | Configurations | Models | Runs |
|---|---|---|---|
| Existing recipe (MLP fusion 3 seeds, C1 vote, C3 rule) | ORIG, T50, XUNC, A270X (Phase 6A B1 / B2 / B3 / A270) + T55, N300, N270, N240, N200 (6B) | 3 | 9 configs × 3 repeats |
| Standalone LSTMs L0–L4, LSTM-sel | ORIG | 6 | 45 + 9 runs (Phase 6A) |
| LSTM-fixed (= L3, chosen by nested selection) | all 9 configurations | 1 | 9 × 9 runs |
| L5 CNN-LSTM, L6 CNN-LSTM + header, F1 / F2 early fusion with ECGFounder f6 | ORIG | 4 | 4 × 9 runs |
| F3 late fusion (C1 members + LSTM-sel), F4 = F3 + nested prior rule | ORIG | 2 | 2 × 3 runs |

**Compute rule:** L3 needs 0.088 min per fold on the RTX 3060, so all 8 sensitivity configurations × 3 seeds take an
estimated 0.53 h (≤ 10 h). Every configuration therefore ran with 3 seeds and every model is eligible, except the
seed-42-only ECGFounder fine-tuned reference.

## 3. Results on the original labels (key models)
| Model | Accuracy [95 % CI] | Bal. acc. | Macro-F1 | Mild F1 | Mild recall | Min recall | Params | Train min / run | ms / ECG |
|---|---|---|---|---|---|---|---|---|---|
| C3 Phase 5 selected (reference) | 0.561 [0.520, 0.603] | 0.456 | 0.448 | 0.221 | 0.177 | 0.177 | – | – | – |
| C3 recipe [ORIG] (same model, regenerated) | 0.562 [0.519, 0.605] | 0.457 | 0.449 | 0.224 | 0.178 | 0.178 | – | – | – |
| C1 Phase 3 vote | 0.547 [0.504, 0.590] | 0.467 | 0.455 | 0.255 | 0.220 | 0.220 | – | – | – |
| F4 vote + LSTM-sel + prior rule | 0.559 [0.517, 0.600] | 0.460 | 0.449 | 0.228 | 0.182 | 0.182 | LSTM part 24k–147k + C1 | 3.52 | 0.06 |
| F3 vote + LSTM-sel | 0.553 [0.508, 0.597] | 0.466 | 0.453 | 0.240 | 0.198 | 0.198 | LSTM part 24k–147k + C1 | 2.29 | 0.06 |
| F2 CNN-LSTM + header + ECGF f6 | 0.551 [0.517, 0.585] | 0.472 | 0.452 | 0.213 | 0.166 | 0.166 | 198,820 | 0.26 | 0.03 |
| F1 LSTM-fixed + header + ECGF f6 | 0.540 [0.507, 0.573] | 0.459 | 0.439 | 0.193 | 0.151 | 0.151 | 130,372 | 0.29 | 0.05 |
| L6 CNN-LSTM + header | 0.497 [0.456, 0.539] | 0.399 | 0.389 | 0.223 | 0.182 | 0.182 | 116,644 | 0.27 | 0.03 |
| L5 CNN-LSTM | 0.484 [0.435, 0.532] | 0.380 | 0.368 | 0.219 | 0.191 | 0.184 | 116,580 | 0.38 | 0.03 |
| LSTM-sel (nested) | 0.470 [0.424, 0.515] | 0.376 | 0.344 | 0.122 | 0.081 | 0.081 (not admissible) | 24,068–147,460 | 2.29 | 0.06 |
| LSTM-fixed L3 | 0.467 [0.417, 0.516] | 0.365 | 0.337 | 0.102 | 0.067 | 0.067 (not admissible) | 48,132 | 0.44 | 0.05 |

Training minutes are per 5-fold run (one seed, one repeat) on the RTX 3060. LSTM-sel's cost includes training all five
variants, and F4 adds its nested inner LSTM fits. Existing models were not re-timed.

## 4. Thresholds (T50 / T55) and training-set changes
**Existing recipe on the original labels** (accuracy / Mild recall):

| Config | MLP fusion | C1 vote recipe | C3 recipe | LSTM-fixed |
|---|---|---|---|---|
| ORIG | 0.544 / 0.141 | 0.548 / 0.222 | 0.562 / 0.178 | 0.467 / 0.067 |
| T50 (≥ 50 %) | 0.548 / 0.148 | 0.547 / 0.205 | 0.554 / 0.142 | 0.466 / 0.068 |
| T55 (≥ 55 %) | 0.544 / 0.166 | 0.544 / 0.207 | **0.569** / 0.186 | 0.459 / 0.097 |
| XUNC (uncertain removed) | 0.545 / 0.124 | 0.542 / 0.169 | 0.559 / 0.103 | 0.462 / 0.077 |
| A270X | 0.542 / 0.159 | 0.544 / 0.222 | 0.561 / 0.193 | 0.461 / 0.079 |
| N300 | 0.546 / 0.142 | 0.543 / 0.220 | 0.559 / 0.184 | 0.464 / 0.068 |
| N270 | 0.545 / 0.142 | 0.547 / 0.225 | 0.558 / 0.195 | 0.458 / 0.086 |
| N240 | 0.539 / 0.155 | 0.545 / 0.227 | 0.558 / 0.209 | 0.474 / 0.072 |
| N200 | 0.541 / 0.150 | 0.546 / **0.234** | 0.553 / 0.220 | 0.470 / 0.067 |

- **Normal-size curve** (`normal_size_curve.png`): reducing the Normal training ECGs from 307 to 200 lowers C3 accuracy
  slightly (0.562 → 0.553; Δ −0.010, same sign in all repeats, Holm p 1.00). Mild recall rises (C3 0.178 → 0.220;
  vote 0.222 → 0.234) and Normal recall falls (C3 0.846 → 0.802). This is the expected prior shift, not a gain in
  discrimination: macro-F1 is flat (C3 0.447–0.452).
- **Definitions** (accuracy on the definition's own labels):

  | Model | ORIG model on alt | alt model on alt | Definition effect | Training effect (Holm p) |
  |---|---|---|---|---|
  | MLP fusion T50 | 0.558 | 0.563 | +0.015 | +0.005 (1.00) |
  | C1 vote T50 | 0.555 | 0.561 | +0.007 | +0.006 (1.00) |
  | C3 T50 | 0.576 | 0.574 | +0.014 | −0.003 (1.00) |
  | LSTM-fixed T50 | 0.485 | 0.483 | +0.019 | −0.002 (1.00) |
  | MLP fusion T55 | 0.540 | 0.543 | −0.003 | +0.002 (1.00) |
  | C1 vote T55 | 0.548 | 0.544 | −0.000 | −0.004 (1.00) |
  | C3 T55 | 0.560 | 0.567 | −0.002 | +0.006 (1.00) |
  | LSTM-fixed T55 | 0.460 | 0.456 | −0.007 | −0.004 (1.00) |

  Under T50 the higher scores are a **definition effect**: the same model scores higher because 27 hard borderline Mild
  ECGs become Normal. Training on T50 adds nothing and costs C3 Mild recall (0.178 → 0.142 on the original labels).
  Under T55 nothing changes materially.

## 5. Statistics (corrected repeated-CV t-test; Holm within each family and metric)
- **(a) LSTM family vs C1 / C3** (16 tests per metric): **no improvement.** The only Holm-significant results are
  **inferiority** of the standalone LSTMs:
  - LSTM-fixed vs C3, accuracy −0.094 (p 0.016);
  - LSTM-sel vs C3, accuracy −0.091 (p 0.045);
  - macro-F1 −0.10 to −0.12 vs C1 and C3 for both (p 0.0001–0.011).

  Fusions are indistinguishable from the existing models: F4 vs C3 −0.002 [−0.017, 0.012]; F4 vs C1 +0.012
  [−0.006, 0.029]; F3 vs C1 +0.006; F2 vs C1 +0.004. All Holm p = 1.00. Bootstrap for F4 vs C1 (repeat 0):
  +0.012 [0.000, 0.035], McNemar p 0.081.
- **(b) each configuration vs ORIG, same model, original labels** (32 tests per metric): all Holm p = 1.00. The largest
  gains are C3 T55 +0.007 (same sign in all 3 repeats; bootstrap [−0.011, 0.023], McNemar p 0.49) and LSTM-fixed N240
  +0.007 (signs differ across repeats).
- **(c) alternative definition vs ORIG on the alternative labels** (8 tests per metric): all Holm p = 1.00.
- **Genuine improvements** (Holm p < 0.05, same sign in all 3 repeats, admissible, eligible): **none** in any family or
  metric.

## 6. Conclusions

### Conclusion 1: best model under the ORIGINAL labels
The best admissible, eligible model trained on the original labels is **C3**, the Phase 5 model (vote of MLP fusion,
ECGFounder linear and header + clinical LR, plus the nested class-prior rule). It reaches **0.561 [0.520, 0.603]**
(re-trained copy: 0.562 [0.519, 0.605]), balanced accuracy 0.456, macro-F1 0.448, and Mild / Moderate / Severe recall
0.18 / 0.46 / 0.34. No LSTM, CNN-LSTM, fusion, Normal-size or exclusion configuration improves on it. The C1 vote is the
better-balanced alternative (balanced accuracy 0.467, Mild recall 0.220) at 0.014 lower accuracy.

**60 %: not reached.** The mean is 0.56 and the corrected 95 % CI [0.52, 0.60] only reaches 0.60 at its upper bound.
The C3 recipe trained on T55 labels scores 0.569 [0.538, 0.601] on the original labels. That is the highest
admissible figure seen, but it belongs to a different training definition (so it is not eligible here by Amendment 1),
its gain over ORIG is not significant (+0.007, Holm p 1.00), and it is still below 0.60.

### Conclusion 2: best under a clinically justified alternative definition (scored on that definition; NOT comparable with 1)
- **T55** (BSE 2020: normal ≥ 55 %, 10 Normal → Mild): the best admissible, eligible model is the **C3 recipe trained on
  T55**, at **0.567 [0.535, 0.598]** on T55 labels (macro-F1 0.453, Mild recall 0.188). Its training effect over the
  ORIG-trained C3 on T55 labels is +0.006 (Holm p 1.00).
- **T50** (heart-failure guidelines: preserved ≥ 50 %, 27 Mild → Normal): the C3 recipe trained on T50 scores 0.574 on
  T50 labels. However, its Mild recall on those labels is **0.1498**, just below the pre-registered 0.15 floor, so it is
  **not admissible**. The rule was applied as written; the gap is 0.0002. MLP fusion trained on T50 fails the same way
  (Mild recall 0.1498). The only admissible T50-trained model is the C1 vote recipe, at 0.561 on T50 labels, below the
  T55 result.

**60 % on the alternative definition: not reached** (0.567, CI upper bound 0.598). These thresholds refer to biplane EF
in the guidelines while the labels are Teichholz, so no clinical validity is claimed for either alternative.

## 7. Interpretation
The four-class accuracy ceiling on these data sits around 0.55–0.57 for every model family, threshold definition and
class-balance setting tried. The limiting factor is the Mild class: it is confused in both directions, and its labels
come from name-matched, unverified Teichholz EF (Phase 6A: an 80 % target would need an EF estimate with r ≈ 0.975;
the models reach r ≈ 0.66). Waveform LSTMs learn less than the ECGFounder embedding. Combining them with ECGFounder
recovers the existing performance but does not exceed it.

## 8. Reproducibility and deviations
- Run on the owner's laptop (Windows, RTX 3060, CUDA 13.2 build of torch 2.15.0; deterministic cuBLAS). GPU runs are
  not guaranteed bit-identical across GPUs.
- Re-used Phase 6A runs were regenerated locally within the 0.005 tolerance (`../phase_6a_normal_mild_lstm/results/regeneration/`).
- Windows environment notes are in `phase_6/PHASE6_RESUME.md`.
