# Phase 6B exploratory report: cleaned / reduced / echo-relabelled training and the 70 % target (MMCLEAN)

Development set (837 ECGs), patient-grouped 5-fold CV × 3 repeats. The locked test set was not used. Definitions were
declared before running (`AMENDMENT_MMCLEAN.md`); the code is `mmclean.py`. Results are in
`results/mmclean_exploratory.json` and `mmclean_results.csv` / `_comparisons.csv` / `_selective.csv`. The plots are
`plots/mmclean_accuracy.png` and `plots/mmclean_coverage_curve.png`. All figures are aggregates: means over 3 repeats
with corrected repeated-CV 95 % CIs. Status: **EXPLORATORY**, reported separately from the pre-registered Phase 6B result.

## Answer: is 70 % reached?
- **On all 837 ECGs (the real task):** no. The best is the unchanged C3 at **0.562 [0.519, 0.605]**. Every cleaned,
  reduced or relabelled training set is 0.525–0.553, none better (Holm p = 1.00).
- **On cleaner ECG subsets (descriptive):** no. The maximum is **0.677 [0.623, 0.732]**: C3 trained on MN200, scored
  on the 620 MMC-kept ECGs. It is not admissible, because Mild recall is 0.089. Most of that number comes from the
  subset, not from the training: the unchanged C3 already scores 0.644 on the same ECGs (+0.033, Holm p 1.00).
- **With selective prediction:** yes, but only for confident Normal and Severe ECGs. The model answers when its
  confidence passes a nested threshold set on the training folds, and refers the rest to echo:
  - C3 trained on MMC reaches **0.713 [0.637, 0.820]** while answering **22 %** of ECGs;
  - C3 trained on MN200 reaches 0.712 [0.582, 0.804] at 14 %;
  - in both cases about 80 % of the answered ECGs are predicted Normal and Mild recall among them is ≤ 0.03.
  This is a working "rule-in Normal / severe, refer the rest" use. It is not four-class grading, and the unchanged
  C3 does not get there (0.670 at 25 %).
- **Optimism:** the best value in this grid is a maximum over 5 configurations × 3 models × several subsets, chosen on
  the same development CV. It would need confirmation on the locked test set. That is the owner's decision; it is not
  used here.
- **Excluded as circular:** removing or relabelling ECGs because a model misclassifies them gives any number one
  wants. On the border ECGs it gives C3 0.764, but it also lifts random guessing from 0.244 to 0.470
  (`ADDENDUM_LABEL_EXPERIMENTS.md`).

## Configurations
Severe ECGs are never removed or relabelled. Removed ECGs are excluded from every fit: the outer members, the nested
inner models and the prior / rule fitting. Folds, inner-val and evaluation are unchanged.

| Config | What changes in training | ECGs affected |
|---|---|---|
| MMC | suspect Mild / Moderate removed: nearest echo of another class, a linked echo of another class, EF spread > 10, or EF within 3 points of a boundary | 217 (Mild 106, Moderate 111); kept 620 |
| MMNC | the same rule also for Normal | 249 (Normal 32, Mild 106, Moderate 111); kept 588 |
| MN240 / MN200 | MMNC, plus Normal outer-training portion reduced to 240 / 200 (exact in all 15 folds) | 249, plus 43–50 / 83–90 Normal per fold |
| ECHO | Normal / Mild / Moderate relabelled to the nearest-echo class; nothing removed | 37 relabelled (N→Mi 3, Mi→N 10, Mi→Mo 9, Mo→N 14, Mo→Mi 1) |

Evaluation subsets:
- **kept_MMC (620) / kept_MMNC (588):** the ECGs with no suspect-label evidence.
- **HC (333):** kept_MMNC minus the Normal / Mild / Moderate ECGs with a weak echo link (name score < 100 or
  interval > 30 d). Its mix is Normal 174, Mild 33, Moderate 35, Severe 91.

The subsets are descriptive. They are not deployable, because the echo is unknown at prediction time. They also make
the task easier by design: they keep ECGs far from the class boundaries.

## Accuracy on the original labels
All values: mean of 3 repeats, with the 95 % CI in brackets.

| Model | Trained | All 837 | kept_MMC (620) | kept_MMNC (588) | HC (333) | Mild recall (all 837) | Admissible (all 837) |
|---|---|---|---|---|---|---|---|
| MLP fusion | ORIG | 0.544 [0.505, 0.583] | 0.628 | 0.626 | 0.624 | 0.141 | no |
| | MMC | 0.532 [0.492, 0.572] | 0.626 | 0.625 | 0.621 | 0.115 | no |
| | MMNC | 0.530 [0.490, 0.570] | 0.623 | 0.621 | 0.622 | 0.110 | no |
| | MN240 | 0.531 [0.490, 0.572] | 0.620 | 0.618 | 0.612 | 0.112 | no |
| | MN200 | 0.525 [0.481, 0.569] | 0.615 | 0.613 | 0.616 | 0.119 | no |
| | ECHO | 0.552 [0.517, 0.587] | 0.635 | 0.631 | 0.625 | 0.166 | yes |
| C1 vote | ORIG | 0.548 [0.505, 0.591] | 0.620 | 0.618 | 0.615 | 0.222 | yes |
| | MMC | 0.538 [0.496, 0.580] | 0.629 | 0.627 | 0.620 | 0.171 | yes |
| | MMNC | 0.537 [0.495, 0.580] | 0.624 | 0.622 | 0.612 | 0.175 | yes |
| | MN240 | 0.533 [0.488, 0.579] | 0.620 | 0.617 | 0.611 | 0.175 | yes |
| | MN200 | 0.536 [0.484, 0.589] | 0.617 | 0.614 | 0.602 | 0.200 | yes |
| | ECHO | 0.535 [0.505, 0.565] | 0.609 | 0.606 | 0.595 | 0.191 | yes |
| C3 | ORIG | **0.562 [0.519, 0.605]** | 0.644 [0.595, 0.695] | 0.639 | 0.619 | 0.178 | yes |
| | MMC | 0.547 [0.504, 0.589] | 0.672 [0.621, 0.724] | 0.664 | 0.653 [0.580, 0.723] | 0.081 | no |
| | MMNC | 0.542 [0.505, 0.578] | 0.667 | 0.659 [0.620, 0.699] | 0.646 | 0.074 | no |
| | MN240 | 0.547 [0.508, 0.586] | 0.673 | 0.666 | 0.651 | 0.092 | no |
| | MN200 | 0.548 [0.498, 0.598] | **0.677 [0.623, 0.732]** | 0.669 | 0.654 [0.577, 0.730] | 0.094 | no |
| | ECHO | 0.553 [0.515, 0.591] | 0.637 | 0.632 | 0.606 | 0.160 | yes |

- On the kept subsets, C3 trained on cleaned data scores +0.02 to +0.03 over the unchanged C3. It does so by
  predicting Normal more often: Normal recall 0.91–0.93 vs 0.85, Mild recall 0.06–0.09, Moderate recall 0.25–0.29
  vs 0.44.
- On the removed ECGs the cleaned models get worse: C3 drops from 0.382 to 0.26–0.27 on the 249 MMNC-removed ECGs.
- **ECHO on its own labels** (all 837): C3 0.573 [0.538, 0.609], MLP fusion 0.569, C1 0.551. The unchanged models
  score 0.583 / 0.561 / 0.566 on the same labels, so the gain is a definition effect, not a training effect.

**Paired comparisons** cover all 5 configurations × 6 subsets × 3 models × 3 metrics (accuracy, macro-F1, Mild recall),
plus ECHO on its own labels: 324 tests.
- No genuine improvement, and no Holm-adjusted p < 0.05.
- On all 837 ECGs, every accuracy difference lies between −0.021 and +0.008.
- All 17 unadjusted p < 0.05 results are **drops**, all for C3:
  - Mild recall on all 837 ECGs: MMC −0.097 [−0.191, −0.004]; MMNC −0.105; MN200 −0.085;
  - Mild recall on the kept subsets: −0.10 to −0.11;
  - accuracy on the removed ECGs: −0.11 to −0.14, and macro-F1 there: −0.07 to −0.08.

## Selective prediction (nested threshold; all 837 ECGs, original labels)
The model answers only when its confidence (maximum vote probability) passes a threshold. In each outer fold the
threshold is set on that fold's nested inner out-of-fold predictions, never on test labels, so it would apply
unchanged to new patients. A CI upper limit above 1 means few answered ECGs per fold: read it as uninformative.

| Model [trained] | Target | Answered | Accuracy on answered | Answered by true class N / Mi / Mo / S | Recall among answered N / Mi / Mo / S |
|---|---|---|---|---|---|
| C3 [ORIG] | 0.65 | 50 % | 0.634 [0.566, 0.759] | 59 / 43 / 42 / 45 % | 0.93 / 0.08 / 0.38 / 0.53 |
| C3 [ORIG] | 0.70 | 25 % | 0.670 [0.564, 0.839] | 32 / 19 / 17 / 18 % | 0.96 / 0.06 / 0.28 / 0.57 |
| C3 [ORIG] | 0.75 | 6 % | 0.721 [0.184, 1.099] | 8 / 4 / 3 / 2 % | 0.99 / 0.00 / 0.25 / 0.78 |
| C3 [MMC] | 0.70 | 22 % | **0.713 [0.637, 0.820]** | 30 / 17 / 14 / 18 % | 0.98 / 0.01 / 0.24 / 0.75 |
| C1 vote [MMC] | 0.70 | 22 % | 0.709 [0.629, 0.822] | 30 / 17 / 14 / 18 % | 0.97 / 0.01 / 0.24 / 0.75 |
| C3 [MN200] | 0.70 | 14 % | 0.712 [0.582, 0.804] | 20 / 10 / 10 / 8 % | 0.99 / 0.03 / 0.17 / 0.73 |
| MLP fusion [MN240] | 0.70 | 31 % | 0.702 [0.642, 0.785] | 40 / 21 / 24 / 22 % | 0.98 / 0.06 / 0.27 / 0.65 |
| C3 [MMNC] | 0.70 | 26 % | 0.682 [0.507, 0.850] | 36 / 20 / 16 / 20 % | 0.98 / 0.01 / 0.17 / 0.66 |
| C3 [ECHO] | 0.70 | 37 % | 0.673 [0.574, 0.857] | 46 / 30 / 29 / 30 % | 0.96 / 0.11 / 0.37 / 0.53 |
| C3 [ECHO] | 0.75 | 15 % | 0.731 [0.595, 0.906] | 22 / 11 / 10 / 8 % | 0.98 / 0.02 / 0.37 / 0.66 |

Of the about 187 ECGs C3 [MMC] answers at the 0.70 target, it predicts 147 Normal, 4 Mild, 9 Moderate and 27 Severe.
The full table, with all 3 models × 6 trainings × 3 targets, is `results/mmclean_selective.csv`.

**Coverage curve** (`plots/mmclean_coverage_curve.png`; ranked by confidence on the same data, so optimistic): the
unchanged C3 rises from 0.562 at 100 % to 0.658 at 50 % and 0.687 at 30 %. The cleaned trainings reach 0.705–0.716 at
30 %. Of the ECGs the unchanged C3 answers at 30 % coverage, 61 % are truly Normal (Normal is 46 % of all 837 ECGs).

## Interpretation
1. **The suspect labels are not what limits accuracy.** Removing 217–249 suspect ECGs from training, fewer Normal
   ECGs, or relabelling 37 ECGs to the nearest echo changes accuracy on the full task by −0.021 to +0.008. This agrees
   with VER / BRES / CLEAR from the cloud session: label noise explains less than one point.
2. **Higher numbers on cleaner subsets come from choosing easier ECGs.** The unchanged C3 already scores 0.644 on
   kept_MMC against 0.562 overall. Training on clean data adds about 0.03 there, and pays for it by predicting Normal
   more often and almost never Mild.
3. **The ECG signal for Mild is weak at this sample size.** Every route to a higher number lowers Mild recall:
   confident answers are Normal or Severe, and the Mild / Moderate ECGs stay uncertain. A phase-1 claim that holds up
   is: "four-class accuracy 0.56 [0.52, 0.61]; with abstention, 0.71 [0.64, 0.82] on the 22 % of ECGs the model is
   confident about, almost all Normal or Severe, with the rest referred to echo".

## Addendum: noise-free training (NFS / NF; `AMENDMENT_MMCLEAN.md` "Noise-free training")
**How much noise there is** (existing predictions, no training; A1 EF regression vs nearest-echo EF):
- all ECGs: r 0.66, residual SD 13.2;
- exact name match and echo ≤ 30 d: r 0.75, SD 11.9;
- echo > 30 d: r 0.42, SD 15.6;
- name score < 100: r 0.58, SD 14.2.

**NFS / NF training**, removing the 412 / 471 ECGs with noisy links from every fit (results in
`results/mmclean_nf_*`, `plots/mmclean_nf_*`):

| Model | Trained | All 837 | NF_clear (264) | HC (333) | Mild recall (all 837) |
|---|---|---|---|---|---|
| MLP fusion | ORIG / NFS / NF | 0.544 / 0.541 / 0.542 | 0.669 / 0.668 / 0.662 | 0.624 / 0.620 / 0.602 | 0.141 / 0.151 / 0.166 |
| C1 vote | ORIG / NFS / NF | 0.548 / 0.548 / 0.544 | 0.679 / 0.682 / 0.669 | 0.615 / 0.624 / 0.594 | 0.222 / 0.186 / 0.182 |
| C3 | ORIG / NFS / NF | **0.562** / 0.552 / 0.546 | **0.707** / 0.702 / 0.683 | 0.619 / 0.647 / 0.595 | 0.178 / 0.150 / 0.171 |

- C3 on all 837: ORIG 0.562 [0.519, 0.605], NFS 0.552 [0.521, 0.583], NF 0.546 [0.508, 0.585]. No comparison is
  Holm-significant.
- Removing the noise does not raise four-class accuracy. The unchanged C3 already reaches 0.707 [0.617, 0.801] on the
  264 clean, clear-cut ECGs. The fixes built on these runs are in `MILDFIX_REPORT.md`.
