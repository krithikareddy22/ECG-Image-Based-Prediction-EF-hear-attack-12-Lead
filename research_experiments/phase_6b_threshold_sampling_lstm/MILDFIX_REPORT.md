# Phase 6B exploratory report: MILDFIX (one fix per reason Mild is not predicted) and final accuracies

Development set (837 ECGs), patient-grouped 5-fold CV × 3 repeats, original labels; the locked test set was not used.
Fixes were declared before running (`AMENDMENT_MILDFIX.md`, commit 7012ba7). Noise-free training (NFS / NF) was
declared in `AMENDMENT_MMCLEAN.md` (commit 2dab35e). Code: `mildfix.py`, `mmclean.py`. Results:
`results/mildfix_results.json`, `mildfix_results.csv` / `_comparisons.csv` / `_selective.csv`,
`results/mmclean_nf_*`. Plot: `plots/mildfix_summary.png`. All figures are aggregates: means over 3 repeats with
corrected repeated-CV 95 % CIs. Status: **EXPLORATORY**. Every tuned parameter was chosen inside the training folds;
the outer test fold never informed a choice.

## Final accuracies
| Question | Answer | 95 % CI | Catch |
|---|---|---|---|
| Best four-class model, all 837 ECGs | **C3 unchanged: 0.562** | [0.519, 0.605] | No fix beats it |
| Best MILDFIX four-class model, all 837 (declared rule: best admissible fix) | F2a ordinal decision, 0.556 | [0.513, 0.598] | −0.006 vs C3 |
| Best Mild detection, all 837 | F1 Mild-aware rule: Mild recall 0.218, Mild F1 0.243 (C3: 0.178 / 0.224) | accuracy 0.537 [0.489, 0.585] | Costs 0.025 accuracy, not significant |
| Abstention, target 0.70 (declared rule: best answering ≥ 20 %) | F2b ordinal model, NFS training: **0.708** | [0.652, 0.769] | Answers 34 %, and only Normal (230) or Severe (53); never Mild or Moderate |
| Abstention that still grades a few Mild / Moderate | C3, NFS training: 0.702 | [0.643, 0.790] | Answers 35 %; Mild recall among answered 0.09 |
| Clean, clear-cut ECGs (NF_clear, 264; descriptive) | F3b EF grading: 0.720 | [0.643, 0.798] | The unchanged C3 already scores 0.707 [0.617, 0.801] there |

**70 % four-class accuracy on all ECGs is not reached by any fix.** It appears only on a hand-picked subset (clean
echo link and EF away from the boundaries; 32 % of ECGs), or when the model is allowed to refer about two-thirds of
ECGs to echo. In those cases the accuracy comes from Normal and Severe, not from Mild.

## What each fix did, all 837 ECGs
| Reason | Fix | Training | Accuracy | Macro-F1 | Mild recall | Mild F1 | Admissible |
|---|---|---|---|---|---|---|---|
| (reference) | C3 | ORIG | **0.562** [0.519, 0.605] | 0.449 | 0.178 | 0.224 | yes |
| 1. Mild looks Normal | F1 Mild-aware rule (macro-F1 offsets) | ORIG | 0.537 [0.489, 0.585] | 0.447 | **0.218** | **0.243** | yes |
| 2. Neighbours overlap | F2a ordinal decision (expected grade + cut points) | ORIG | 0.556 [0.513, 0.598] | 0.438 | 0.160 | 0.212 | yes |
| 2. Neighbours overlap | F2b ordinal model (Frank & Hall LR) | ORIG | 0.542 [0.509, 0.575] | 0.384 | 0.031 | 0.056 | no |
| 3. EF too imprecise | F3a EF stacking (ridge EF̂ + vote → LR + prior rule) | ORIG | 0.560 [0.518, 0.602] | 0.445 | 0.126 | 0.173 | no |
| 3. EF too imprecise | F3b EF grading (tuned EF̂ cut points) | ORIG | 0.556 [0.527, 0.585] | 0.426 | 0.146 | 0.193 | no |
| 5. Label noise | C3 trained noise-free (NFS) | NFS | 0.552 [0.521, 0.583] | 0.431 | 0.150 | 0.196 | yes |
| 5. Label noise | F1 / F2a / F2b / F3a / F3b on NFS members | NFS | 0.549 / 0.535 / 0.505 / 0.550 / 0.546 | 0.435 / 0.408 / 0.322 / 0.426 / 0.427 | 0.159 / 0.150 / 0.022 / 0.105 / 0.141 | — | F1, F2a only |
| 4. Ceiling | Abstention: see the next table | | | | | | |

- **Statistics:** 33 paired comparisons vs C3 (ORIG) on accuracy, macro-F1 and Mild recall.
  - No genuine improvement; no Holm-adjusted p < 0.05.
  - The only unadjusted p < 0.05 results are losses for F2b: Mild recall −0.15 [−0.23, −0.07]; NFS accuracy −0.057.
- **Noise-free training alone** (`results/mmclean_nf_exploratory.json`, top-3 recipes):
  - All 837 ECGs: MLP fusion 0.541, C1 0.548, C3 0.552 (NFS); C3 0.546 (NF). Holm p 1.00 against the ORIG models.
  - On the 264 NF_clear ECGs, the NFS-trained C3 scores 0.702 [0.624, 0.784] vs 0.707 for the unchanged C3.

## Abstention (reason 4): answer only when confident (nested threshold, target 0.70)
| Model | Training | Answered | Accuracy on answered | Answered by true class N / Mi / Mo / S | Predicted N / Mi / Mo / S |
|---|---|---|---|---|---|
| C3 | ORIG | 25 % | 0.670 [0.564, 0.839] | 32 / 19 / 17 / 18 % | 153 / 6 / 19 / 27 |
| F3b EF grading | ORIG | **52 %** | 0.694 [0.648, 0.750] | 73 / 41 / 29 / 34 % | 375 / 1 / 43 / 20 |
| F2b ordinal model | ORIG | 36 % | 0.697 [0.629, 0.785] | 49 / 26 / 20 / 36 % | 236 / 0 / 1 / 65 |
| C3 | NFS | 35 % | 0.702 [0.643, 0.790] | 48 / 26 / 22 / 21 % | 236 / 7 / 16 / 33 |
| F1 Mild-aware | NFS | 35 % | 0.701 [0.643, 0.789] | 49 / 26 / 22 / 21 % | 237 / 7 / 16 / 33 |
| **F2b ordinal model** | **NFS** | 34 % | **0.708 [0.652, 0.769]** | 47 / 23 / 19 / 30 % | 230 / 0 / 0 / 53 |
| F3a EF stacking | NFS | 44 % | 0.694 [0.507, 0.843] | 62 / 34 / 24 / 27 % | 311 / 2 / 12 / 42 |
| F3b EF grading | NFS | 46 % | 0.693 [0.633, 0.751] | 64 / 37 / 25 / 32 % | 335 / 0 / 7 / 45 |

Noise-free training helps abstention: C3 answers 35 % instead of 25 % at the same target.

The useful clinical reading is the EF-grading triage (F3b, ORIG), which answers 52 % of ECGs at 0.69:
- Normal recall 0.99 among answered;
- Moderate recall 0.34;
- Severe recall 0.32;
- Mild is essentially never answered correctly.

## Conclusion: why the fixes do not reach 70 % on four classes
1. **Label noise is real but small in effect** (reason 5). Removing it narrows the ECG–echo EF error from 13.2 to
   11.7 points, and four-class accuracy does not move (0.552 vs 0.562).
2. **Decision fixes trade accuracy for Mild** (reasons 1–3). Every rule that predicts more Mild loses Normal
   accuracy, and every rule that gains accuracy predicts less Mild. The ordinal and EF-based routes match C3, and do
   not beat it, because they rest on the same information.
3. **The ceiling is in the signal** (reason 4). The ECG separates Normal from reduced EF well (AUROC 0.88–0.90) but
   Mild from Moderate barely (0.59–0.63, also in the image-only clinician features, `MILDFEAT_REPORT.md`).
   Abstention turns that into a valid triage: about 70 % accuracy when answering a third to a half of ECGs, almost
   all Normal or Severe.
4. **What would move Mild:**
   - more patients (phase 2);
   - the raw digital ECG instead of printed PDFs;
   - echo labels linked by patient ID within days (Simpson biplane);
   - an EF error of 8–9 points, against 12–13 now (see the diagnostic below).

## Diagnostic behind the five reasons (laptop session; aggregates; script outside the repo, results reproduced here)
- **Where true-Mild ECGs go (C3, ORIG):** Normal 43 %, Mild 18 %, Moderate 28 %, Severe 11 %.
  - Lower third of the Mild band (near Moderate): predicted Normal 31 %, Moderate 33 %.
  - Upper third (near Normal): predicted Normal 51 %.
- **Pairwise AUROC of the C3 vote probabilities:**
  - Normal-vs-Mild 0.75, Mild-vs-Moderate 0.63, Moderate-vs-Severe 0.64;
  - Normal-vs-Moderate 0.88, Normal-vs-Severe 0.90.
- **A1 EF regression vs nearest-echo EF:** r 0.66, residual SD 13.2, MAE 10.3.
  - Predicted EF of true-Mild ECGs (25th–75th percentile): 41.9–60.4.
  - Predicted EF inside the Mild band: 29 % of true Mild, but also 41 % of Moderate and 35 % of Severe.
- **Ceiling simulation:** nearest-echo EF plus Gaussian error, thresholded with the class rules (200 draws):

  | EF error SD | 0 | 3 | 5 | 8 | 10 | 13 |
  |---|---|---|---|---|---|---|
  | Accuracy | 1.00 | 0.90 | 0.82 | 0.73 | 0.68 | 0.63 |
  | Mild recall | 1.00 | 0.81 | 0.67 | 0.50 | 0.42 | 0.34 |

- **Echo-to-echo EF spread** among the 127 ECGs with ≥ 2 linked echoes: median 15.4 points (IQR 4.8–31.1). This is
  an upper bound on measurement noise, because it includes change over time and possible link errors. The Mild band
  is 11.8 points wide.

**Optimism:** the "best" rows above are maxima over 12 candidates and several subsets on the same development CV.
Confirming any of them needs the locked test set, which is the owner's decision and is not used here.
