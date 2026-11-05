# Test kit: score the frozen triage model on a test set you collect

**What it is.** A frozen copy of the header-free triage model (the "56 % answered at 0.69 accuracy" result) plus a
one-shot scorer. You give it ECG PDFs with their echo EF. It reports how the model does on those patients, side by
side with what cross-validation predicted. The rules are fixed in `PHASE7_PLAN.md`, Amendment 5, before any test data
exist.

**What the model looks at.** Only the ECG drawing in the PDF: the ECGFounder embedding of the digitised waveform and
118 measurements from the drawing (mildfeat's 121 columns minus 3 beat-count quality columns). It does not read the
printed machine text, age or other header fields. The
patient's sex is used to place the Normal / Mild cut (52 % for men, 54 % for women), exactly as in the label rule.

**What it outputs for each ECG:**
- **triage:** a confident class (Normal / Mild / Moderate / Severe) or "refer to echo";
- **IMG-C3:** a four-class prediction for every ECG (the "all patients" result);
- **EF estimate:** the model's EF estimate from the image.

**Frozen on 2026-11-06** (file SHA-256 `09808d12…`; all values in `results/testkit_frozen_card.json`):
- **EF estimate:** the average of the ridge and LightGBM EF models (unverified links weighted 0.3).
- **EF grading:** Normal above the sex boundary (52 / 54), Mild 45.5 up to the boundary, Moderate 35–45.5,
  Severe below 35.
- **When it answers:** only when that grade agrees with IMG-C3 (prior rule α = 0.5) **and** the EF estimate is at
  least 3.05 points from every cut. Otherwise it refers.
- **Checks passed:**
  - the kit's code reproduces the cross-validated classifiers (100 % same class) and the EF models (to 10⁻¹³);
  - on 25 development PDFs read from scratch, the image inputs are identical and every decision is the same;
  - all 40 development ECGs in a pipeline test were detected as duplicates (they would be refused in a real run).

**Locked test used once (2026-11-06, owner-authorised, `PHASE7_PLAN.md` Amendment 6):**
- four-class accuracy 0.553 [0.485, 0.621];
- triage: 50 % answered at 0.747 [0.660, 0.837].

The 210 locked ECGs are no longer unseen, so the next test must use new ECGs.

## 1. Rules for a valid test set
| Rule | Why |
|---|---|
| **New ECGs only.** None of the 837 development ECGs. Ideally new patients, recorded after a cut-off date. | A model scores higher on ECGs it trained on. |
| **Same patient = same MRN / UHID.** The hospital checks this; do not match by name. | The development set has 29 near-identical name pairs. |
| **One echo per ECG,** same patient, done within **30 days** (7 is better), with no heart attack, revascularisation or device in between. | The echo must describe the same heart. |
| Record the **numeric EF** and the method (**Simpson biplane** if possible). | The label comes from EF and sex. |
| **Same ECG machine and printout format** as the development PDFs. | The PDF reader is built for this layout. |
| **Score once.** Never change the model, its thresholds or the test set after seeing the result. | Otherwise the test is no longer a test. |

**How many ECGs** (accuracy around 0.70, about 56 % of patients answered):

| Goal | ECGs needed |
|---|---|
| Accuracy on answered patients, ±7 points | about 300 |
| Accuracy on answered patients, ±5 points | about 580 |
| Mild recall, ±10 points | about 100 Mild patients |

## 2. Steps
All commands run from the repository root with the project Python (`C:\krithika_major\p6.ps1` on the laptop). Keep the
PDFs and the manifest **outside the repository** or under a git-ignored `data/` folder: file names can contain patient
names. The kit refuses a manifest anywhere else.

1. **Freeze (once; already done on this branch):**
   ```
   python -m research_experiments.phase_7.testkit.freeze --check
   python -m research_experiments.phase_7.testkit.freeze --freeze
   python -m research_experiments.phase_7.testkit.freeze --selftest 25
   ```
   The frozen model is saved as `phase_7/data/private/testkit/frozen_triage.pt` (git-ignored). Every chosen value and
   the file's SHA-256 are recorded in `results/testkit_frozen_card.json`. The scorer refuses a model whose hash differs
   from the card.
2. **List your PDFs into a manifest:**
   ```
   python -m research_experiments.phase_7.testkit.score --skeleton D:\test_ecgs --out D:\test_ecgs\manifest.csv
   ```
3. **Fill the manifest** (template: `manifest_template.csv`):

   | Column | Content |
   |---|---|
   | `ecg_file` | path to the PDF (absolute, or relative to the manifest's folder) |
   | `patient_id` | pseudonymised MRN / UHID, the same code for the same patient (used for patient-level CIs) |
   | `sex` | M or F |
   | `ef` | echo EF in % (number) |
   | `echo_gap_days` | days between the ECG and the echo |
   | `ef_method` | e.g. Simpson biplane, Teichholz |

4. **Score, once:**
   ```
   python -m research_experiments.phase_7.testkit.score --manifest D:\test_ecgs\manifest.csv --name hospital_2027a
   ```

## 3. What the scorer reports
Results go to `results/testkit/<name>_scores.json` (aggregates only, safe to commit) and are printed. The per-ECG table
stays in `data/private/testkit/runs/<name>/` (manifest row numbers, never file names).

- **Primary analysis:** readable PDFs with a valid EF and sex and an echo gap ≤ 30 days. Every excluded row is counted
  by reason.
- **Sensitivity analysis:** all readable rows, whatever the echo gap.
- **Triage:**
  - share answered and accuracy on answered;
  - the predicted mix, and the recall of each class among answered patients;
  - what the confident "Normal" answers truly are, and the share of EF < 41 % patients wrongly given "Normal";
  - Mild caught, out of all Mild patients.
- **IMG-C3 on every readable ECG:** accuracy, balanced accuracy, macro-F1, precision / recall / F1 per class, and the
  confusion matrix.
- **EF estimate error:** bias, SD, MAE and r.
- **95 % CIs:** Wilson intervals, and a patient-cluster bootstrap (2,000 resamples).
- **Comparison with cross-validation:** each key number next to its CV expectation, and whether the CV value lies inside
  the test CI. The CV expectations are 56 % [48, 63] answered at 0.694 [0.655, 0.736], and IMG-C3 0.558
  [0.504, 0.611] on all patients.

## 4. What the scorer refuses
- **Development ECGs:** the same file-name timestamp or path, or an ECGFounder embedding near-identical to a
  development ECG (cosine ≥ 0.999).
- **Locked-test ECGs (the 210 kept since Phase 2):** recognised by path or file-name timestamp, from metadata only.
  They are scored only if the owner passes `--locked-test-authorised`. Do that only with written authorisation, and
  after the 11 + 3 near-identical name pairs are resolved (`phase_6/PHASE6_HANDOFF.md`, known issue 2).
- **A manifest that was already scored:** a re-run needs `--again "<reason>"`. The reason is logged in the ledger and
  printed in the result.
- **A frozen model whose hash differs from the card,** or one that has not passed its PDF self-test.

## 5. What the kit cannot check
- **The same patient with a different ECG.** Only the hospital can check this, by MRN.
- **Locked-test PDFs that were renamed or copied.** The kit does not open those files, so it cannot recognise them.
- **Whether the echo truly belongs to the patient and the date,** and whether the EF was re-measured honestly.
