# Phase 4A–4C: spreadsheet, continuous-EF and cohort audit

> **Status:** final. Sections 1–3 were written before the owner authorised the ECG_ECHO matching files and are kept
> unchanged as the record of that stage. The UPDATE (sections 4–8) supersedes the hypotheses in section 2.
> Follow-up reports: `PHASE4_LABEL_QUALITY_REPORT.md` (G cohort, label noise), `PHASE4_REGRESSION_REPORT.md`,
> `PHASE4_MULTITASK_REPORT.md`, `PHASE4_MILD_CLASS_ANALYSIS.md`, `PHASE4_MODEL_BENCHMARK.md` and
> `PHASE4_FINAL_REPORT.md`.

Branch `research/ecg-ef-phase4`, created from `39aa39f`.
- **Originals:** no original file was modified. The integrity snapshot covers 4,648 files: originals,
  the 4 spreadsheets, and all phase 1–3 outputs.
- **Privacy:** no patient names, dates of birth or user names were printed or stored. Text values were
  inspected only in masked form (letters → A, digits → 9) and as counts.

## 1. The four authorised spreadsheets (`LV_ECG_Dataset_{2020,2021,2022,2024}/*.xlsx`) [VALID]

| File | Sheets | Rows | Columns (all text) |
|---|---|---|---|
| 2020 | Sheet1 | 232 | Category, Original Filename, Source Path, Destination Path |
| 2021 | Sheet1 | 294 | same |
| 2022 | Sheet1 | 226 | same |
| 2024 | Sheet1 | 295 | same |

**Column meaning, verified on every row:**
- `Category`: Normal / Mild / Moderate / Severe. It equals the label folder in `Destination Path`
  for all 1,047 rows.
  - 2020: 120 / 48 / 36 / 28
  - 2021: 120 / 74 / 69 / 31
  - 2022: 120 / 42 / 43 / 21
  - 2024: 120 / 68 / 73 / 34
- `Original Filename`: the ECG PDF name. It contains the 14-digit acquisition timestamp and the
  patient name. 1,047/1,047 timestamps are unique and match the project manifest one-to-one.
- `Source Path`: the monthly raw-ECG folder the PDF was copied from (e.g. "OCTOBER 2020 ECGS",
  "ecg_2024_RAW\1.ECGS JANUARY 2024").
- `Destination Path`: the label folder it was copied to.

**What the spreadsheets do NOT contain:**
- no continuous EF value (no cell matches an EF-like number);
- no echocardiogram date;
- no separate ECG date column (only the timestamp inside the file name);
- no patient identifier;
- no EF measurement method;
- no multiple-echo information.

**Conclusion [VALID]:** these four spreadsheets are **file-copy logs** of how the label folders were
assembled. They add no label information beyond the folder labels.
- Phase 4B–4G (EF regression, ECG-to-echo intervals, label-disagreement analysis, Mild EF
  distribution) **cannot be carried out from them.**
- Per-row checks:
  - duplicates: none (1,047 unique timestamps);
  - missing values: none in any of the 4 columns.

## 2. Where the EF data most likely is (metadata only; contents NOT opened)

`main`'s git history contains a directory `ECG_ECHO/`, removed by commit `9f2a883` ("Delete ECG_ECHO
directory", 2026-11-04):

| File | Size |
|---|---|
| `2020_echo_ecg_matching_50percent_20261029_162620.xlsx` | 157 KB |
| `2021_echo_first_ecg_matching_20261029_161227.xlsx` | 327 KB |
| `2022_ECG_ECHO_MATCHING_70_PERCENT_20261029_170108.xlsx` | 309 KB |
| `2024_ECG_ECHO_MATCHING_70_PERCENT_20261029_165914.xlsx` | 312 KB |
| `ecg_echo_matching.ipynb` | 972 KB |

- [HYPOTHESIS] These hold the continuous EF values, the echo dates and the original ECG–echo matching.
  `ecg_ef/config.py` says the folders were "labelled from the echo report (ECG-echo matching
  notebook, since removed; EF by Teichholz)".
- [HYPOTHESIS] The "50 percent" / "70 percent" in the file names suggest a fuzzy name-similarity
  threshold. If so, the existing matching would not count as a reliable match under the Phase 4
  rules.
- These files are **outside** the four spreadsheets authorised for Phase 4 and were deliberately
  deleted from `main`, so they were **not opened**. Using them needs the owner's explicit permission.

## 3. Clinical thresholds currently documented

From `ecg_ef/config.py`, documentation only; cannot be verified without the matching files:
- Normal: EF > 52 % (male) / ≥ 54 % (female)
- Mild: 41–52 % (male) / 41–53 % (female)
- Moderate: 31–40 %
- Severe: < 31 %
- Method: Teichholz

Status: **unverified**.

---

# UPDATE: ECG_ECHO matching files audited (owner authorised read-only use)

All numbers are aggregates, from `phase_4/results/audit_matching.json`, `cohort_summary.csv` and
`ef_repeat_variability.json`. Plots: `phase_4/plots/ef_distribution_and_intervals.png`,
`label_disagreement.png`, `ef_spread_multiple_links.png`.

## 4. Structure and matching method [VALID]

**Sheets:** "Matched Records", "Unmatched Echo", "Unmatched ECG" (+ "LV Summary" for 2022 and 2024).

| Year | Matched rows (unique ECGs) | Unmatched echo | Unmatched ECG |
|---|---|---|---|
| 2020 | 968 (738) | 290 | 1,401 |
| 2021 | 1,226 (1,195) | 901 | 4,210 |
| 2022 | 937 (868) | 28 | 3,960 |
| 2024 | 1,596 (1,377) | 43 | 2,658 |

**EF and thresholds:**
- EF method: **Teichholz only**. 89.6 % of values have one decimal, 10.4 % are integers.
- Thresholds, exact rule recovered from the matching notebook (`classify_lv`):
  - Normal: EF > 52 (M) / ≥ 54 (F)
  - Mild: 41 ≤ EF ≤ 52 (M) / < 54 (F)
  - Moderate: 31 ≤ EF < 41
  - Severe: EF < 31
- The sheets' own LV class matches this rule for 98.9 % of rows.

**Matching:**
- There is **no patient identifier**. ECG and echo are linked by **name similarity only**: the maximum of
  several fuzzy ratios, including substring `partial_ratio`.
- Acceptance score ≥ 50 (2020), ≥ 80 (2021), ≥ 70 (2022/2024). The notebook version uses ≥ 85 plus exact age and
  sex agreement.
- There is **no date window**, and one ECG may match several echoes.
- Within these sheets, a "reliable match" in the sense of a verified identifier is therefore **not possible**.
  Links can only be graded.

## 5. The 1,047 dataset ECGs [VALID]
- **Linked echo:** every ECG has ≥ 1 (by acquisition timestamp).
- **Number of links:** 883 have 1; 120 have 2; 31 have 3; 12 have 4; 1 has 7.
- **EF spread across links:** for ECGs with > 1 link the median is **17 points**, the 90th percentile **46**.
  [HYPOTHESIS] Spreads this large indicate wrong-person fuzzy matches.
- **91** ECGs have links implying different EF classes.
- **22** ECGs have no link that reproduces their folder label.
- **Label-consistent link:**
  - name score 100 for 562 ECGs, < 100 for 485 (minimum 50);
  - sex agrees in 519 / 519 where the echo sex is recorded.
- **Folder label vs class from the nearest-in-time echo** (label-agnostic): agreement **93.5 %**, 68 disagree.

| Folder \ nearest-echo class | Normal | Mild | Moderate | Severe |
|---|---|---|---|---|
| Normal (480) | 476 | 4 | 0 | 0 |
| Mild (232) | 12 | 209 | 11 | 0 |
| Moderate (221) | 18 | 0 | 195 | 8 |
| Severe (114) | 14 | 0 | 1 | 99 |

## 6. ECG-to-echo intervals (nearest link; development set / locked test) [VALID]

| Window | Dev ECGs (patients) | Test ECGs | Folder vs EF-class agreement (dev) |
|---|---|---|---|
| ≤ 7 d | 541 (521) | 131 | 0.935 |
| ≤ 14 d | 550 (530) | 138 | 0.936 |
| ≤ 30 d | 587 (565) | 148 | 0.934 |
| ≤ 60 d | 647 (615) | 166 | 0.927 |
| ≤ 90 d | 688 (654) | 176 | 0.927 |
| all | 837 (794) | 210 | 0.934 |
| **high-confidence G** (exact name, ≤ 30 d, sex agrees, no conflicting link within 30 d) | **357 (345)** | 91 | **0.958** |

- **Direction**, label-consistent link, all 1,047: echo before ECG 256, same day 543, echo after ECG 248. Most
  links beyond 90 days are echo-before-ECG (136 of 204).
- [VALID] Most ECGs were recorded on the day of the echo. The window choice therefore trades away mostly the
  ≈ 20 % of ECGs with an echo more than 2 months away.
- **Recommended windows**, chosen on data reliability, not test accuracy:
  - **≤ 14 d** (dev 550), as the best balance of size and temporal reliability;
  - **G** (dev 357) as the high-confidence cohort.

## 7. Label quality (EF of the label-consistent link per folder class) [VALID]

| Folder class | n | Patients | EF range | Median | Within ±2 points of a boundary |
|---|---|---|---|---|---|
| Normal | 480 | 477 | 50.1–85.9 | 69.7 | 12 |
| Mild | 232 | 226 | 35.0–75.9 | 45.5 | 70 (48 near 41; 19 near 52/54) |
| Moderate | 221 | 214 | 30.0–56.4 | 36.1 | 69 |
| Severe | 114 | 108 | 6.1–30.8 | 26.1 | 22 |

- **Repeat-echo variability:** among ECGs with two exact-name echoes, both within 30 days, there are only **18
  pairs** (median |ΔEF| 0.0, 90th percentile 15.3). **Measurement uncertainty cannot be quantified reliably**
  from these data.
- [HYPOTHESIS] Literature test-retest variability for Teichholz EF is several EF points, comparable to the
  11–13-point width of the Mild band.

## 8. Status
- **Done:** 4A, 4B (audit) and 4C (cohorts) here; 4D–4H in the follow-up reports listed at the top.
- **Section 2 hypotheses:**
  - the first ("these hold the continuous EF values") is **confirmed**;
  - the second ("fuzzy name-similarity threshold") is **confirmed** (section 4).
- **EF measurement method:** a single method (Teichholz) is used throughout, so differences between methods cannot
  contribute to label variability in this dataset.
