# Phase 6B exploratory report: ULT (large diverse header-free ensemble, owner objective)

Declared in `AMENDMENT_ULT.md` (commit 3f8273c). Code `ult.py`; results `results/ult_exploratory.json`.
- **Data:** 837 development ECGs, original folder labels, patient-grouped 5-fold CV × 3 repeats.
- **Tuning:** all nested inside the training folds.
- **Inputs:** no header, age or sex. The locked test set was not used.
- **Status:** EXPLORATORY.

**Objective (owner):** maximum accuracy with every class, Mild included, ≥ 15 % detected. The decision rule maximises
inner accuracy subject to every inner recall ≥ 0.18, a margin that keeps the outer recall ≥ 0.15.

## Results (means over 3 repeats; recall = % detected)
| Model | Accuracy [95 % CI] | Macro-F1 | Recall N / Mi / Mo / S | Admissible |
|---|---|---|---|---|
| **ULT-stack** (meta-LR on 10 members + predicted EF) | **0.571 [0.542, 0.600]** | 0.439 | 0.905 / 0.164 / 0.420 / 0.286 | yes |
| ULT-vote (10 members) | 0.564 [0.531, 0.596] | 0.440 | 0.885 / 0.182 / 0.395 / 0.311 | yes |
| ULT-select (declared primary; greedy selection) | 0.559 [0.525, 0.592] | 0.426 | 0.894 / 0.166 / 0.392 / 0.267 | yes |
| ULT-stack, inner floor 0.15 (diagnostic) | 0.570 | 0.435 | 0.903 / 0.150 / 0.437 / 0.282 | borderline |
| HIER-ext | 0.570 [0.527, 0.612] | 0.448 | 0.884 / 0.162 / 0.433 / 0.341 | yes |
| ENH-vote | 0.569 [0.527, 0.610] | 0.444 | 0.878 / 0.173 / 0.456 / 0.286 | yes |
| IMG-C3 | 0.559 [0.506, 0.613] | 0.444 | 0.847 / 0.191 / 0.452 / 0.300 | yes |
| C3 (with header) | 0.561 [0.520, 0.603] | 0.448 | 0.845 / 0.177 / 0.460 / 0.344 | yes |

- **ULT-stack details:**
  - precision N / Mi / Mo / S: 0.701 / 0.278 / 0.453 / 0.380;
  - per-repeat accuracy 0.577 / 0.563 / 0.574;
  - Normal→Mild errors 5.7 % (C3: 9.2 %), Mild→Normal 48.5 % (C3: 42.9 %).
- **Comparisons:** 12 paired tests. ULT-stack vs C3 / IMG-C3 / ENH-vote / HIER-ext: +0.010 / +0.012 / +0.002 /
  +0.001. All Holm p = 1.00. **No genuine improvement.**

## New members alone (argmax; diagnostics)
| Member | Accuracy | Recall N / Mi / Mo / S | Selected parameter |
|---|---|---|---|
| M8 multi-scale ECGFounder LR | 0.543 | 0.79 / 0.21 / 0.42 / 0.42 | C 0.32–1 |
| M9 kNN on embeddings | 0.540 | 0.98 / 0.02 / 0.40 / 0.02 | k 60 (grid edge) |
| M10 gradient boosting | 0.540 | 0.83 / 0.18 / 0.44 / 0.25 | 100 iterations (grid edge) |
| M12 confusion-zone specialist | 0.469 | 0.64 / 0.27 / 0.56 / 0.00 (Normal / Mild / Moderate only) | C 0.32 |

- **Greedy selection weights** (mean over folds):
  - EF-ordinal M6 0.48;
  - gradient boosting M10 0.21;
  - HIER-ext 0.13;
  - synth ECGFounder M2 0.08;
  - M5 0.06;
  - the rest ≤ 0.02.
- **Stacker C:** 0.01 in all folds, the bottom of the grid. The stacker wants to be close to an average. A smaller C
  would move it towards ULT-vote (0.564).

## Interpretation
Ten members spanning five representations and four learner types, three combiners, and the owner's objective without
the macro-F1 guard all land at 0.56–0.57. That is the same plateau as every earlier model. The largest ensemble mostly
leans on the EF-based members: the remaining signal is "how low is the EF", and it is estimated to about ±13 points.
More model complexity cannot recover information the ECG printouts and labels do not contain.
