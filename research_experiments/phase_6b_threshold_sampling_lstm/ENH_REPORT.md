# Phase 6B exploratory report: ENH (new image-only features, better pre-processing, learned combination)

Declared in `AMENDMENT_ENH.md` (commit b62d659), plus Diagnostic D1 (commit 676abdf). Code: `enh_build.py`,
`enh_rocket.py` (separate aeon venv) and `enh.py`. Results: `results/enh_exploratory.json` and `results/enh_build.json`.

All 837 development ECGs (Normal 384, Mild 185, Moderate 177, Severe 91), original labels, patient-grouped 5-fold CV ×
3 repeats. All tuning is nested inside the training folds. No header, age or sex input. The locked test set was not
used. Status: **EXPLORATORY**.

## Pre-processing (P1): beat-synchronous 10-s 12-lead reconstruction
- R peaks were found on the rhythm strip in 99.9 % of ECGs.
- 99.5 % of leads have at least one complete template cycle.
- Template vs recorded signal inside each lead's own column: median r **0.982** (IQR 0.937–0.992).
- The reconstruction is faithful, but it **does not add EF information**. The same LR head scores 0.528 on the synth
  ECGFounder features (M2) and 0.537 on the original aligned input (M2ref, diagnostic D1). Repeating a lead's own beat
  across 10 s gives ECGFounder a more familiar input, but no new information about the heart.

## Results (means over 3 repeats; recall = % of each class detected)
| Model | Accuracy [95 % CI] | Macro-F1 | Recall N / Mi / Mo / S | Precision N / Mi / Mo / S | Admissible |
|---|---|---|---|---|---|
| **ENH-vote** (6 members, tuned class weights) | **0.569 [0.527, 0.610]** | 0.444 | 0.878 / 0.173 / 0.456 / 0.286 | 0.697 / 0.296 / 0.465 / 0.372 | yes |
| ENH-stack (declared primary) | 0.566 [0.529, 0.603] | 0.415 | 0.921 / 0.146 / 0.424 / 0.201 | 0.678 / 0.291 / 0.433 / 0.362 | **no** (Mild < 0.15) |
| HIER-ext (two-stage, earlier) | 0.570 [0.527, 0.612] | 0.448 | 0.884 / 0.162 / 0.433 / 0.341 | 0.702 / 0.281 / 0.465 / 0.390 | yes |
| IMG-C3 (header-free reference) | 0.559 [0.506, 0.613] | 0.444 | 0.847 / 0.191 / 0.452 / 0.300 | 0.713 / 0.292 / 0.441 / 0.353 | yes |
| C3 (with header) | 0.561 [0.520, 0.603] | 0.448 | 0.845 / 0.177 / 0.460 / 0.344 | 0.710 / 0.297 / 0.455 / 0.344 | yes |

Members alone (argmax; diagnostics):

| Member | Accuracy | Macro-F1 | Recall N / Mi / Mo / S |
|---|---|---|---|
| M1 IMG-vote | 0.541 | 0.450 | 0.773 / 0.238 / 0.437 / 0.377 |
| M2 ECGFounder on synth | 0.528 | 0.437 | 0.760 / 0.182 / 0.444 / 0.414 |
| M2ref same head, aligned input (D1) | 0.537 | 0.444 | 0.782 / 0.209 / 0.411 / 0.418 |
| M3 median-beat shape | 0.467 | 0.386 | 0.681 / 0.193 / 0.360 / 0.333 |
| M4 MiniRocket | 0.504 | 0.411 | 0.740 / 0.222 / 0.375 / 0.330 |
| M5 aligned f6 ‖ waveform features | 0.543 | **0.454** | 0.781 / 0.207 / 0.433 / 0.436 |
| M6 EF-ordinal ridge | 0.534 | 0.341 | 0.954 / 0.292 / 0.000 / 0.289 |

**Tuning notes:** MiniRocket's C was at the lowest grid value (10⁻⁴) in 4 of 15 folds. All other members chose
interior values.

**Comparisons (accuracy, corrected t-test over 15 cells, Holm over 6):**
- ENH-vote vs IMG-C3 / C3 / HIER-ext: +0.010 / +0.008 / −0.001.
- ENH-stack vs the same: +0.007 / +0.005 / −0.004.
- All Holm p = 1.00, and no difference has the same sign in all repeats. **No genuine improvement.**

## Neighbour confusion (share of the true class predicted as the neighbour)
| Model | Normal→Mild | Mild→Normal | Mild→Moderate | Moderate→Mild | Moderate→Severe | Severe→Moderate | Error rate | Errors ≥ 2 classes away |
|---|---|---|---|---|---|---|---|---|
| ENH-vote | 0.071 | 0.476 | 0.267 | 0.183 | 0.134 | 0.333 | 0.431 | 0.305 |
| ENH-stack | 0.027 | 0.528 | 0.247 | 0.226 | 0.081 | 0.388 | 0.434 | 0.330 |
| HIER-ext | 0.068 | 0.465 | 0.267 | 0.202 | 0.139 | 0.300 | 0.430 | 0.308 |
| C3 (with header) | 0.092 | 0.429 | 0.283 | 0.153 | 0.173 | 0.333 | 0.439 | 0.306 |

The better models reduce Normal→Mild errors (0.09 → 0.03–0.07) but pay for it with more Mild→Normal errors
(0.43 → 0.47–0.53). Mild↔Moderate confusion is unchanged (about 0.25 / 0.20). Tuning moves errors between neighbours;
it does not remove them.

## Why the errors happen (descriptive; ENH-stack, C3 in brackets)
- **Distance of the echo EF from the nearest class boundary**, the dominant factor:

  | Distance | ECGs | Error rate |
  |---|---|---|
  | < 3 points | 221 | 0.66 (0.62) |
  | 3–6 | 202 | 0.68 (0.66) |
  | 6–10 | 64 | 0.39 (0.32) |
  | ≥ 10 | 350 | **0.16** (0.21) |

  The ECG-predicted EF has an error SD of about 13 points. For the half of the ECGs that lie within 6 points of a
  boundary, the class is close to a guess.
- **Echo link:** error 0.40 with an exact name match and an echo ≤ 30 days, vs 0.46 for other links (C3: 0.40 / 0.47).
- **Digitisation quality:** no effect (0.436 vs 0.431).
- **Inside the Mild band** (Mild band 41 to 52 / 54):

  | Position | Predicted Normal | Mild | Moderate | Severe |
  |---|---|---|---|---|
  | Upper third (near Normal) | 68 % | 11 % | 19 % | 2 % |
  | Middle third | 55 % | 14 % | 24 % | 7 % |
  | Lower third (near Moderate) | 41 % | 18 % | 29 % | 13 % |

  The Mild band is about 12 EF points wide, narrower than the ECG's EF error, so Mild ECGs are spread over the
  neighbouring classes.

## Interpretation
- **No real gain:** every new representation (synth ECGFounder, median-beat shape, MiniRocket, EF-ordinal) and both
  learned combinations land at 0.57 ± 0.01, the same as C3, IMG-C3 and HIER-ext.
- **The best image-only result** remains about 0.57: ENH-vote 0.569 and HIER-ext 0.570, admissible, not significantly
  better than C3 (0.561). The declared primary, ENH-stack, is not admissible.
- **The cause is information, not method:** half the ECGs have an echo EF within 6 points of a class boundary, the
  ECG–echo EF error is about 13 points, and the Mild band is 12 points wide.
- **Where the models do well:** ECGs whose EF is ≥ 10 points from every boundary (42 % of the set) are classified with
  about 84 % accuracy. Raising accuracy on the rest needs more precise EF labels (same-admission echo, biplane Simpson,
  repeat reads) and simultaneous 10-s digital ECGs, not more modelling of these printouts.
