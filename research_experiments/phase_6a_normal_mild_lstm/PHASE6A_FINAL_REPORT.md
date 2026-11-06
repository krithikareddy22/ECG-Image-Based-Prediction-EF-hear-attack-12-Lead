# Phase 6A final report: Normal/Mild boundary and LSTM study

Development set only: 837 ECGs / 794 patient keys, patient-grouped 5-fold CV × 3 repeats (`phase_2/configs/folds_D.csv`).
The 210-ECG locked test was never loaded, predicted, scored or inspected. Accuracy is four-class accuracy on the
**original labels**, pooled over the outer folds and averaged over the 3 repeats; seeds are combined by probability mean.
95 % CIs are corrected repeated-CV intervals over the 15 (repeat, fold) cells (N_train 535 / N_test 167).
A model is **admissible** if every class recall is ≥ 0.15. Pre-registration: `PHASE6A_PLAN.md`; the per-model table is
`PHASE6A_MODEL_COMPARISON.csv`, the tests `results/phase6a_comparisons.csv`, and the summary `results/phase6a_summary.json`.
Plots are in `plots/`. Every run is listed in `PHASE6A_EXPERIMENT_INVENTORY.md`.

**Where it ran:** the existing-model configurations ran in the cloud session (CPU). The LSTM grid, nested selection,
LSTM-fixed under A270 / B2 / B3 and all LSTM-family models ran on the owner's laptop (RTX 3060, CUDA). The cloud runs'
private predictions were regenerated on the laptop and checked against the committed aggregates. All 84 runs are within
0.005 (max 0.0047: header + clinical LR, B1 repeat 1, 3 ECGs). This is why "C3 recipe [B1]" (regenerated, 0.562) and
"C3 Phase 5 selected" (byte-identical baseline, 0.561) differ by 0.001: they are the same model.

## Key results (original labels)

| Model | Admissible | Accuracy [95 % CI] | Balanced acc. | Macro-F1 | Mild F1 | Mild recall |
|---|---|---|---|---|---|---|
| C3 Phase 5 selected (vote + nested prior rule) | yes | 0.561 [0.520, 0.603] | 0.456 | 0.448 | 0.221 | 0.177 |
| C1 Phase 3 vote ensemble | yes | 0.547 [0.504, 0.590] | 0.467 | 0.455 | 0.255 | 0.220 |
| C2 Phase 3 calibrated MLP | no | 0.568 [0.534, 0.603] | 0.463 | 0.442 | 0.160 | 0.112 |
| Header + ECGFounder MLP fusion | no | 0.544 [0.505, 0.583] | 0.466 | 0.440 | 0.186 | 0.140 |
| ECGFounder fine-tuned (Phase 6 A1, seed 42) | yes | 0.536 [0.492, 0.581] | 0.452 | 0.436 | 0.202 | 0.164 |
| ECGFounder frozen, linear | yes | 0.528 [0.484, 0.573] | 0.442 | 0.434 | 0.244 | 0.214 |
| C3 recipe [A270] | yes | 0.561 [0.518, 0.604] | 0.452 | 0.446 | 0.237 | 0.193 |
| C3 recipe [B2] | no | 0.554 [0.511, 0.596] | 0.446 | 0.434 | 0.187 | 0.142 |
| C3 recipe [B3] | no | 0.559 [0.518, 0.600] | 0.444 | 0.426 | 0.144 | 0.103 |
| LSTM L0 (rhythm only) | no | 0.426 [0.387, 0.466] | 0.293 | 0.257 | 0.065 | 0.040 |
| LSTM L1 | no | 0.452 [0.399, 0.504] | 0.357 | 0.324 | 0.084 | 0.058 |
| LSTM L2 | no | 0.483 [0.437, 0.529] | 0.389 | 0.352 | 0.114 | 0.076 |
| LSTM L3 (= LSTM-fixed) | no | 0.467 [0.417, 0.516] | 0.365 | 0.337 | 0.102 | 0.067 |
| LSTM L4 | no | 0.472 [0.424, 0.519] | 0.368 | 0.339 | 0.117 | 0.081 |
| **LSTM-sel (nested selection)** | no | **0.470 [0.424, 0.515]** | 0.376 | 0.344 | 0.122 | 0.081 |
| CNN-LSTM L5 (run in 6B) | yes | 0.484 [0.435, 0.532] | 0.380 | 0.368 | 0.219 | 0.191 |
| Vote + LSTM-sel (6B F3) | yes | 0.553 [0.508, 0.597] | 0.466 | 0.453 | 0.240 | 0.198 |
| Vote + LSTM-sel + nested prior rule (6B F4) | yes | 0.559 [0.517, 0.600] | 0.460 | 0.449 | 0.228 | 0.182 |

Nested selection over L0–L4 chose L3 in 7 of the 15 B1 folds, L4 in 4, L2 in 3 and L1 in 1, so **LSTM-fixed = L3**
(1-layer bidirectional LSTM-64, 48,132 parameters).

## Answers

1. **Is Mild the main difficulty? Yes, but not the only one.** Mild has the lowest recall and F1 in every model (C3: Mild
   recall 0.177 and F1 0.221, against Normal 0.845, Moderate 0.460 and Severe 0.344 recall). Of the 185 Mild ECGs, C3
   predicts on average 79 as Normal, 52 as Moderate, 21 as Severe and only 33 as Mild, so Mild is confused in both
   directions. Severe (recall 0.34) and Moderate (0.46) are also weak.
2. **Does Normal 270 (A270) help? No.** Accuracy changes by −0.002 (MLP fusion), −0.004 (vote), −0.002 (C3) and −0.006
   (LSTM-fixed); all Holm p = 1.00. Mild recall rises slightly for C3 (0.178 → 0.193; Mild F1 +0.014, not significant).
3. **Does moving borderline Mild to Normal (B2, EF ≥ 50 %) help? No.** On the original labels accuracy changes by +0.004,
   −0.001, −0.009 and −0.001 (MLP, vote, C3, LSTM-fixed; Holm p = 1.00). On the B2 labels the apparent gain is a
   **definition effect** (+0.007 to +0.019: the unchanged B1 models score higher simply because the task is easier). The
   **training effect** is −0.003 to +0.006 (p 0.43–0.92).
4. **Does it hurt Mild recall? Yes, for the C3 model (consistently, but not significantly).** C3 Mild recall is 0.178
   (B1), 0.142 under B2 (below the 0.15 floor) and 0.103 under B3 (removing uncertain borderline ECGs); both are
   inadmissible. C3 Mild F1 changes by −0.036 (B2) and −0.079 (B3), with the same sign in all three repeats but Holm
   p = 1.00. The vote loses less (0.222 → 0.205 under B2). A270 does not hurt Mild recall.
5. **Does an LSTM beat the existing models? No.** LSTM-sel reaches 0.470 [0.424, 0.515] and is not admissible (Mild
   recall 0.081). It is 0.058–0.098 below every existing model, in all three repeats. Accuracy is Holm-significant vs C2
   (Δ −0.098, p 0.013) and borderline vs C3 (Δ −0.091, p 0.052). Macro-F1 is 0.09–0.11 lower, Holm-significant vs the
   ECGFounder linear model, C1, C2 and C3 (p 0.002–0.041). The best single variant (L2, 0.483) is also far below.
6. **Is the LSTM complementary to ECGFounder? Partly, but not enough to help.** LSTM-sel is right on 24–25 % of the ECGs
   that the ECGFounder MLP fusion gets wrong (error correlation 0.38–0.43). Adding it to the C1 vote changes accuracy by
   +0.010 / +0.006 / −0.002 and macro-F1 by −0.002 / −0.001 / −0.008, so the pre-registered criterion (a gain in all
   three repeats) is **not met**. Phase 6B ran the fusions anyway: vote + LSTM-sel 0.553 vs C1 0.547 (+0.006, Holm
   p 1.00), and with the nested prior rule 0.559 vs C3 0.561 (−0.002, Holm p 1.00).
7. **Does CNN-LSTM add value? Not relative to the existing models.** The 6A condition for running it (LSTM-sel admissible
   with macro-F1 ≥ 0.389) was **not met**, so it is not part of the 6A confirmatory analysis. Phase 6B ran it
   unconditionally: L5 reaches 0.484 [0.435, 0.532] and is admissible (min recall 0.184; Mild F1 0.219). In an
   exploratory, uncorrected comparison it is better balanced than LSTM-sel (+0.014 accuracy, +0.023 macro-F1, +0.098
   Mild F1; p 0.66 / 0.43 / 0.09). It remains 0.08 below C3 and 0.06 below C1.
8. **What is significant?** Only the **inferiority of the standalone LSTMs** (item 5). No label configuration (A270, B2,
   B3), no Normal-size change and no fusion produces a significant change; there is **no genuine improvement**
   (Holm p < 0.05, same sign in all 3 repeats, admissible). Bootstrap / McNemar for the six largest gains: all 95 %
   intervals include 0 and all McNemar p ≥ 0.13.
9. **Which model is best balanced?** Among admissible models, **C3** has the highest accuracy (0.561) and **C1 (vote)** is
   the best balanced: highest balanced accuracy 0.467, macro-F1 0.455, Mild F1 0.255 and Mild recall 0.220, at 0.014
   lower accuracy. C2 has higher accuracy (0.568) but is not admissible (Mild recall 0.112).

## Training cost of the LSTM family (RTX 3060, per 5-fold run = one seed and one repeat)
| Model | Parameters | Training min / run | Inference ms / ECG |
|---|---|---|---|
| L0 | 17,668 | 0.31 | 0.03 |
| L1 | 24,068 | 0.33 | 0.03 |
| L2 | 57,348 | 0.47 | 0.05 |
| L3 (LSTM-fixed) | 48,132 | 0.44 | 0.05 |
| L4 | 147,460 | 0.74 | 0.10 |
| LSTM-sel | 24,068–147,460 (chosen variant) | 2.29 (all five variants must be trained) | 0.06 |
| CNN-LSTM L5 | 116,580 | 0.38 | 0.03 |

## Deviations and notes
- The cloud session's partial LSTM runs (L4 29/45, L1 1/45) were never committed; the whole grid was run here once.
- CNN-LSTM and the fusions were run in Phase 6B (unconditional, owner) with identical definitions. Phase 6A reads those
  runs (`compare.py`) and includes them in its Holm family only if its own pre-registered conditions hold; they did not.
- GPU training is fast enough (≈ 5–12 s per fold) that the compute fallbacks of the plan were not needed.
- The regenerated LSTM inputs differ from the cloud copy only in float32 rounding (amplitude percentiles in the 7th
  digit); mask fractions and missing-lead counts are identical.
- No EF value is verified (fuzzy name matching), so a misclassification is never treated as a label error.

## Limitations
Single centre and a single ECG machine; labels come from name-matched echo reports with Teichholz EF; 837 development
ECGs; the corrected CIs are wide (±0.04). The locked test has not been used, so all figures are cross-validated
development estimates.
