# Why four-class LVEF accuracy from these ECG printouts stops at about 0.57–0.63

Synthesis of Phases 4–7 (development CV, 837 ECGs; the locked test set was not used). Every number comes from a
committed report:
- `MILDFIX_REPORT.md`, `MMCLEAN_REPORT.md`, `ENH_REPORT.md`, `BANDS_REPORT.md`, `HC30_REPORT.md`;
- `../phase_7/PHASE7_FINAL_REPORT.md`.

## 1. The core problem: the ECG estimates EF to about ±13 points, and the middle bands are only 10–12 points wide
- **EF estimation:** the best header-free ECG models estimate EF with r ≈ 0.66, a residual SD of about 13 points and
  an MAE of about 10. The same holds with or without the header, for every encoder tried.
- **Band widths:** Mild (41 to 52 / 54) is about 12 points wide and Moderate (31–41) is 10. An error SD larger than the
  band width means that a Mild ECG's predicted EF falls outside the Mild band most of the time.
  - Predicted EF of true-Mild ECGs: IQR 41.9–60.4.
  - The Mild band also receives 41 % of the Moderate and 35 % of the Severe ECGs.
- **Simulated ceiling:** the true echo EF plus Gaussian error, cut at the class limits.

  | EF error SD | 0 | 3 | 5 | 8 | 10 | 13 |
  |---|---|---|---|---|---|---|
  | Accuracy | 1.00 | 0.90 | 0.82 | 0.73 | 0.68 | **0.63** |
  | Mild recall | 1.00 | 0.81 | 0.67 | 0.50 | 0.42 | 0.34 |

  The models reach 0.57–0.59, close to the 0.63 ceiling for their EF precision. The model is not the bottleneck; the
  precision of the EF information is.
- **Where the errors sit:** 51 % of the ECGs have an echo EF within 6 points of a class limit, and their error rate is
  66–68 %. ECGs ≥ 10 points from every limit have a 16 % error rate.

## 2. What the ECG can and cannot see
- **Strong:** normal vs clearly reduced function. AUROC: Normal vs Moderate 0.88, Normal vs Severe 0.90.
- **Weak between neighbouring grades:** Normal vs Mild 0.75, Mild vs Moderate 0.63, Moderate vs Severe 0.64. The best
  single Mild vs Moderate factor reaches 0.64 (`mild_factor_study.json`).
- **What separates Mild from Normal:** wider QRS, LBBB / incomplete LBBB, and an "abnormal ECG" reading (AUROC up to
  0.73). These are conduction and scar markers. Once function is reduced, how much it is reduced (45 vs 35 %) leaves
  few additional electrical signs.
- **Physiology:** the ECG records electrical activity; EF is a mechanical, load-dependent measurement. Mild dysfunction
  often has a near-normal ECG, and a scarred heart with an abnormal ECG can still have a mildly reduced or
  normal EF.

## 3. Label noise: the target itself is uncertain by several points
- **EF method:** labels come from Teichholz M-mode EF. Teichholz extrapolates from one linear dimension and assumes
  symmetric contraction. With regional wall-motion abnormalities, as after a myocardial infarction, it can be off by
  many points. Guidelines recommend biplane Simpson.
- **Echo-to-echo disagreement:** 127 ECGs have ≥ 2 linked echoes. Their EF differs by a median of 15.4 points (IQR
  4.8–31.1). This is an upper bound on measurement noise, because it also includes real change and link errors.
- **Linking by fuzzy name match:**
  - only 44 % of ECGs (366 / 837) have a clean link (exact name match, echo ≤ 30 days);
  - 391 have a name-match score < 100;
  - 192 have the nearest echo more than 30 days away;
  - 55 folder labels disagree with the nearest echo.
- **Link quality shows directly in the ECG–EF agreement:**

  | Echo link | r | Error SD (EF points) |
  |---|---|---|
  | Clean link | 0.75 | 11.9 |
  | Echo > 30 days away | 0.42 | 15.6 |
  | Name score < 100 | 0.58 | 14.2 |

  Error rate is 0.40 on the 395 high-confidence ECGs vs 0.46 on the rest.
- **But cleaning labels alone does not move accuracy on all ECGs** (VER, BRES, ECHO, MMC, NF: −0.02 to +0.01). The
  remaining label noise still exceeds the band width, and removing noisy ECGs also removes training data.

## 4. The input: printed PDFs, 2.5 s per lead, not simultaneous
- **Signal limits:** each lead covers 2.5 s, in four successive columns. Only lead II covers 10 s. The ECGFounder
  backbone was pre-trained on simultaneous 10-s digital 12-lead ECGs.
- **Reconstruction does not help:** a faithful reconstruction of a 10-s 12-lead ECG (template r 0.98) did not improve
  accuracy (0.528 vs 0.537). Repeating a lead's own beat adds no new information about the heart.
- **Not a scan-quality problem:** digitisation quality is unrelated to error (0.436 vs 0.431).
- **The header adds nothing:** header-free 0.559–0.570 vs 0.561 with the header. The printed machine measurements
  repeat what is in the waveform.

## 5. Too few patients, and too few Mild patients
- **Data size:** 837 ECGs (185 Mild); each training fold has about 535 ECGs.
- **Learning curve** (`phase_7`): accuracy improves +1.4 points per doubling of the data; Mild F1 shows no trend.
- **Extrapolation:**
  - at the current label quality, 70 % would need about 1 million ECGs;
  - with verified labels, an EF error ≤ 9 points could need about 13,000.

## 6. What it is not
| Not the cause | Evidence |
|---|---|
| The model architecture | LR, MLP, LSTM, CNN-LSTM, 1-D ResNet, ECGFounder fine-tuning, ECG-FM, MiniRocket, median-beat models, stacking, two-stage, ordinal and EF-regression grading all land at 0.54–0.57 |
| The decision rule | Prior rules, class weights, cut-point tuning and Mild-aware rules only move errors between neighbouring classes; they do not remove them |
| The band limits alone | The best clinically justified alternative (ESC 2021) reaches 0.626; no limit detects both Mild and Moderate well |

## 7. What would fix it, ranked
1. **Verified echo linkage:** by medical-record number, within the same admission or a few days, instead of name
   matching. This attacks the label noise directly (r 0.42 → 0.75 already, between bad and clean links).
2. **Biplane Simpson EF,** with repeat or core-lab reads near the 41 % and 50–54 % limits, instead of Teichholz.
3. **Digital 10-s simultaneous 12-lead ECGs** (XML / DICOM) instead of PDF printouts.
4. **More patients, enriched for Mild and Moderate:** thousands, ideally from more than one site.
5. **Clinical covariates** where ethically and practically available: NT-proBNP, prior MI or heart failure,
   medications. These carry EF information the ECG lacks.
6. **Meanwhile, the defensible use of the current model:**
   - four-class accuracy 0.57 (0.63 on ESC 2021 limits);
   - confident Normal / Severe triage at about 0.70 on the third to half of ECGs answered;
   - the rest referred to echo.
