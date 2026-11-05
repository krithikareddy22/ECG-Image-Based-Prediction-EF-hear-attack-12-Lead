# Phase 6B exploratory amendment: MMC / MMNC (train without suspect Mild / Moderate / Normal labels)

**Date:** 2026-11-05. Written before any MMC / MMNC run (laptop session), at the owner's request:
- "few cases in the dataset are wrong ... the difficult Mild ... train by removing them in Mild and Moderate and the
  clear-cut border cases too ... no LSTM this time, our top 3 models";
- "keep Severe as it is ... only change for Mild, Moderate and Normal".

Code: `mmclean.py`.

- **Suspect label:** an ECG that meets at least one of these:
  - the class of the nearest echo differs from the folder label;
  - a linked echo is of another class;
  - the EF spread across linked echoes is > 10 points;
  - the nearest-echo EF is within 3.0 points of a class boundary (52 M / 54 F, 41 %, 31 %).

  The selection uses echo evidence only and never a model prediction. Removing ECGs because a model misclassifies
  them is circular: see the circularity check in `ADDENDUM_LABEL_EXPERIMENTS.md`.
- **Not used:** name-match score < 100 and an ECG–echo interval > 30 days. They flag 391 and 192 of 837 ECGs, evenly
  across classes (Normal 183 and 95). They show a weak link, not a wrong label. Using them would leave 33 Mild and
  35 Moderate ECGs.
- **MMC:** suspect Mild and Moderate ECGs removed. That is 217 ECGs (Mild 106, Moderate 111); kept: Normal 384,
  Mild 79, Moderate 66, Severe 91 (620).
- **MMNC:** suspect Normal, Mild and Moderate ECGs removed. That is 249 ECGs (Normal 32, Mild 106, Moderate 111);
  kept: Normal 352, Mild 79, Moderate 66, Severe 91 (588).
- **Severe ECGs are never removed** (owner request).
- **Training:** removed ECGs are excluded from every fit: the outer members, the nested inner models and the
  prior / rule fitting. This is the same mechanism as `AMENDMENT_CLEAR.md`. Labels are original; folds, inner-val and
  evaluation populations are unchanged.
- **Models:** the top 3 admissible-family recipes on the original labels, with no LSTM (owner request):
  - MLP fusion (3 seeds), 0.544;
  - C1 vote, 0.548;
  - C3 (vote + nested prior rule), 0.561.

  C2 (0.568) is not admissible (Mild recall < 0.15) and is not included.
- **Evaluation:**
  1. All 837 ECGs, original labels. This is primary and comparable with every other result.
  2. The kept ECGs of the configuration. Descriptive only:
     - it is not deployable, because the echo EF is unknown at prediction time;
     - keeping ECGs far from the boundaries makes the task easier by design (spectrum bias).
  3. The removed ECGs.

  On every subset, the same recipe trained on the original labels (Phase 6A B1 runs) is scored on the same ECGs. Each
  comparison uses a paired corrected repeated-CV t-test on accuracy, macro-F1 and Mild recall, with Holm correction
  over all tests (54; 72 with the HC subset of the addition below). A genuine improvement needs Holm p < 0.05, the same sign in all 3 repeats, and admissibility
  (every class recall ≥ 0.15).
- **Status:** EXPLORATORY and reported separately. It is not eligible as an improvement of the original task unless it
  meets the criterion on all 837 ECGs.

## Addition (2026-11-05, written before computing either analysis; owner request "bring accuracy above 70")
Both analyses are descriptive. Neither changes the models.
- **HC subset (evaluation only):** the MMNC-kept ECGs minus the Normal / Mild / Moderate ECGs with a weak echo link
  (name-match score < 100 or ECG–echo interval > 30 days). Severe is untouched. That leaves 333 ECGs: Normal 174,
  Mild 33, Moderate 35, Severe 91. The ORIG, MMC and MMNC models are all scored on it.
  - The class mix differs from the full set (Severe 27 % vs 11 %).
  - Its accuracy is a property of the selected ECGs, not of the model.
- **Coverage curve (all 837 ECGs):**
  - ECGs are ranked by the model's confidence (maximum vote probability); prediction = the model's decision (the C3
    rule for C3).
  - Accuracy is reported at 100 / 90 / 80 / 70 / 60 / 50 / 40 / 30 % coverage, with the unanswered ECGs referred to
    echo.
  - The coverage at which a target is reached is read from the same data, so it is optimistic.
- Neither selection criterion is tuned after seeing a result.

## Grid (2026-11-05, written before any grid run; owner / supervisor request)
The request: "keep Severe the same, take fewer Normal cases, Mild / Moderate / Normal borders may change labels, reach
above 70 %". The configurations are fixed here and are not extended after seeing results. Severe is never removed or
relabelled. The models and the HC / coverage analyses are as above.
- **MN240 / MN200:** MMNC removal, plus the Normal outer-training portion reduced to 240 / 200. Whole Normal patients are
  removed from inner-train with the N-size seeds, which reaches exactly 240 / 200 in every fold.
- **ECHO:** a Normal / Mild / Moderate ECG is relabelled to its nearest-echo class when that class is Normal / Mild /
  Moderate and differs from the folder label. No ECG is removed. This relabels 37 ECGs:
  - Normal→Mild 3;
  - Mild→Normal 10;
  - Mild→Moderate 9;
  - Moderate→Normal 14;
  - Moderate→Mild 1.

  The relabelling uses echo evidence only. ECHO is scored on the original labels and on its own labels.
- **Already covered, not repeated** (cloud session; `ADDENDUM_LABEL_EXPERIMENTS.md`, `results/clear_exploratory.json`):
  - border relabelling by an echo rule (VER, VERB, BRES);
  - boundary shifts (T50, T55);
  - Normal size alone (N300–N200);
  - training without border ECGs (CLEAR).
- **Excluded, because it is circular:** relabelling or removing ECGs because a model misclassifies them. On the border
  ECGs this lifts C3 to 0.764, but it also lifts random guessing from 0.244 to 0.470 (`ADDENDUM_LABEL_EXPERIMENTS.md`).
- **Reporting:**
  - every configuration × model × subset is reported, with corrected CIs and class recalls;
  - Holm correction runs over all paired comparisons;
  - any value ≥ 0.70 is reported together with its subset, label set, class mix and class recalls;
  - the maximum over this grid is optimistic, because it is selected on the same development CV. It would need
    confirmation on the locked test set, which is the owner's decision and is not used here.

## Selective prediction (2026-11-05, written before computing; owner: "remove all which are difficult to predict")
"Difficult to predict" is defined by the model's own confidence, never by the true label. Removing the ECGs a model got
wrong is circular and is not done.

- **Rule:** the model answers only when its confidence (maximum probability; for C3, the vote probability) reaches a
  threshold. Unanswered ECGs are referred to echo.
- **Threshold, per outer fold:** the lowest threshold at which the model's nested inner out-of-fold predictions on that
  fold's training portion reach the target accuracy on the original labels, with ≥ 20 answered ECGs.
  - For C3, the inner prediction is the C3 rule with the fold's own alpha and prior.
  - The threshold is applied unchanged to the outer test fold, so no test label sets it. It would work the same way on
    new patients.
- **Targets:** 0.70 (primary), with 0.65 and 0.75 for context.
- **Scope:** all three models, the ORIG models and every MMCLEAN configuration.
- **Reported:**
  - coverage, overall and per true class;
  - accuracy on the answered ECGs, with a corrected CI over fold cells;
  - per-class recall among the answered ECGs;
  - the predicted-class mix.

## Noise-free training (2026-11-05, written before any NF / NFS run; owner: "let's first eliminate the noise")
**Noise measured first (no training).** The existing predictions were scored by echo-link quality:
- A1 EF regression vs nearest-echo EF:
  - all ECGs: r 0.66, residual SD 13.2;
  - exact name match and echo ≤ 30 days: r 0.75, SD 11.9;
  - echo > 30 days: r 0.42, SD 15.6;
  - name score < 100: r 0.58, SD 14.2.
- Unchanged C3 accuracy: 0.627 on the noise-free-link ECGs below, 0.512 on the rest.

So weak links add label noise. Removing it narrows the EF error, but only from 13.2 to 11.7 points.

- **Noise-free link (NF set):** exact name match (score 100), ECG–echo interval ≤ 30 days (true dates), folder label =
  nearest-echo class, no linked echo of another class, and EF spread ≤ 10. Boundary ECGs are kept: a near-boundary
  label with a clean link is not noise. That gives 366 ECGs: Normal 183, Mild 77, Moderate 74, Severe 32.
- **NFS (primary; supervisor rule "keep Severe the same"):** every non-NF Normal / Mild / Moderate ECG is removed from
  all fits; all 91 Severe are kept. 412 removed; 425 kept (Normal 183, Mild 77, Moderate 74, Severe 91).
- **NF (secondary):** link noise removed in every class, Severe included. 471 removed; 366 kept.
- **Training, models, evaluation and selective prediction:** as above. The top 3 recipes; removed ECGs are excluded
  from every fit; folds, inner-val and evaluation populations are unchanged.
- **Evaluation subsets:**
  - all 837 (primary);
  - NF set (366);
  - NFS set (425);
  - not-NFS (412);
  - NF_clear = NF set and not within 3 points of a boundary (264);
  - HC (333).

  Paired corrected t-tests vs the ORIG models on the same ECGs, with Holm correction over this batch. Outputs:
  `results/mmclean_nf_*`, `plots/mmclean_nf_*`.
