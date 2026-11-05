# Phase-1 result statement: ECG-image triage for LV dysfunction, with the Mild limitation

**Owner framing (2026-11-05):** "triage + Mild limitation". This file gives the numbers to quote and the wording that
the evidence supports.

**Evidence base:**
- development set only: 837 ECGs, 794 patients;
- patient-grouped 5-fold cross-validation × 3 repeats; every threshold set inside the training folds;
- corrected repeated-CV 95 % CIs;
- sections 1–4 are development estimates (optimistic as the best of several designs). The locked 210-ECG test
  set was then scored **once** with the frozen model (section 0, `PHASE7_FINAL_REPORT.md` §8);
- **image only:** the ECG drawing digitised from the PDF (ECGFounder embedding + 121 image-measured clinician
  features). No printed machine measurements, age or sex.

Sources: `PHASE7_FINAL_REPORT.md` (§1, §2b), `results/triage_statement.json`, `results/e6_cascades.csv`,
`results/e7_feasibility.json`.

## 0. Locked-test confirmation (scored once, 2026-11-06; 199 test ECGs after the pre-declared name-pair exclusion)
| Quantity | Locked test (95 % CI) | Development CV |
|---|---|---|
| Four-class accuracy, all patients (image only) | **0.553** [0.485, 0.621] | 0.558 [0.504, 0.611] |
| Mild precision / recall | 0.32 / 0.32 | 0.29 / 0.19 |
| Triage: patients given a confident answer | **50 %** [43, 57] | 56 % [48, 63] |
| Triage: accuracy on those answers | **0.75** [0.66, 0.84] | 0.69 [0.66, 0.74] |
| Confident "Normal" answers with EF ≥ 41 % | 85 % [77, 92] | 90 % [86, 94] |
| EF < 41 % patients falsely given a confident "Normal" | 21 % [11, 31] | 15 % [8, 22] |

Every development value lies inside the locked-test interval. All 210 test ECGs: four-class accuracy 0.538; triage
48 % answered at 0.74.

## 1. Headline: triage
In each case the model first estimates EF from the image, then grades it. It gives an answer only when (a) the grading
agrees with the image classifier (IMG-C3) and (b) the estimated EF is far enough from a class cut-off. Otherwise it
refers the patient to echocardiography.

| Quantity | Value (95 % CI) |
|---|---|
| Patients given a confident answer | **56 %** [48, 63] |
| Accuracy on those answers (four-class) | **0.70** (0.696 [0.655, 0.736]) |
| Patients referred to echo | 44 % |
| Of the confident "Normal" answers: truly Normal / Mild / Moderate / Severe | 74 % / 16 % / 7 % / 3 % |
| Confident "Normal" answers with EF ≥ 41 % (no moderate / severe dysfunction) | **90 %** [86, 94] |
| Of all patients with EF < 41 %: falsely given a confident "Normal" | **15 %** [8, 22] |
| Confident "Moderate / Severe" answers that truly have EF < 41 % | 63 % [41, 85] |

**Wording to use.** From the ECG image alone, the model gives a confident answer for about half of the patients
(56 %), with 70 % accuracy. Most confident answers are "Normal" (about 394 of 466). These exclude moderate or severe
LV dysfunction in 90 % of cases. The remaining 44 % are referred to echocardiography.

**Do not claim:**
- "70 % accuracy" without the word *answered* and the 56 %;
- that the triage detects Mild;
- that it is a safe stand-alone rule-out: 15 % of moderate / severe patients receive a confident "Normal".

## 2. The Mild limitation
| Quantity | Value |
|---|---|
| Four-class accuracy on **all** patients (image only, IMG-C3) | 0.558 [0.504, 0.611] |
| Mild precision / recall / F1 (IMG-C3) | 0.29 / 0.19 / 0.23 |
| Best image-only Mild F1 (IMG-vote) | 0.26 (precision 0.30, recall 0.24; accuracy 0.541) |
| Mild patients given a confident "Normal" by the triage | 34 % |
| Confident Mild answers in the triage | about 4 per 837 |
| Best accuracy on answered patients if ≥ 10 % of Mild patients must be detected | 0.59–0.64 |

**Wording to use.**
- Mild LV dysfunction (EF 41–52 % in men, 41–54 % in women) cannot be reliably identified from the ECG image in this
  dataset. Mild predictions are correct only about 30 % of the time, and Mild ECGs look like Normal or Moderate ECGs.
  The pairwise AUROC for Mild vs Moderate is about 0.6.
- Any rule that detects Mild lowers accuracy below 0.65. Answering only part of the patients does not help: with Mild
  detected, at most 0.59–0.64 is achievable.

## 3. Why, and what phase 2 needs (one paragraph)
- **The size of the gap:** the ECG-derived EF estimate has an error of about 13 EF points (SD), which is wider than the
  12-point Mild band. Removing noisy labels, alternative models and representation learning did not change this
  (Phases 6B–7).
- **What more data alone would do:** the learning curve shows each doubling of training data adding about 1.4
  accuracy points. About 100,000 patients of the current kind would give roughly 0.65.
- **What the curve suggests for verified labels:** EF matched to the ECG by patient identifier within days, Simpson
  biplane EF, and raw digital 10-s 12-lead ECGs could bring the EF error below the ~9 points needed for 70 %
  four-class accuracy with reliable Mild detection. That is a hypothesis to test, not a result.

## 4. One-paragraph summary (for the report or abstract)
> Using only the ECG image (12-lead printouts digitised and encoded with the ECGFounder foundation model), four-class
> LVEF classification reached 0.56 accuracy [0.50, 0.61] in patient-grouped cross-validation on 837 ECGs. Used as a
> triage tool, the model gave a confident answer for 56 % of patients with 0.70 accuracy, mainly ruling out moderate
> or severe dysfunction (90 % of confident "Normal" answers had EF ≥ 41 %). The other patients were referred to
> echocardiography. Mildly reduced EF could not be identified reliably: Mild precision was about 0.30, and no
> operating point combined accuracy above 0.69 with Mild detection. On the pre-specified locked test set, scored once
> with the frozen model (199 ECGs), four-class accuracy was 0.55 [0.49, 0.62], and the triage answered 50 % of
> patients with 0.75 [0.66, 0.84] accuracy, confirming the development estimates.

## 5. Before quoting these numbers externally
- **Locked test set:** done once (section 0). The 11 test ECGs whose names nearly match a development patient were left
  out by a rule fixed beforehand; with them included, the results are 0.538 four-class and 48 % at 0.74. The locked
  test can no longer serve as an unseen test. Any changed model needs new data (`testkit/TESTKIT.md`).
- **Optimism:** the triage was the best of two header-free stage-1 designs. The other design scored 0.697 on 53 %.
