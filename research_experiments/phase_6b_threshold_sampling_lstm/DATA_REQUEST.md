# Data request: what is needed to move four-class LVEF accuracy beyond about 0.57–0.63

For the hospital records, echocardiography and ECG departments, and the ethics committee. It follows from
`PROBLEM_ANALYSIS.md`. The current models are close to the ceiling set by the data: the ECG estimates EF within about
13 points, and the label EF is uncertain by several points. These requests target those two limits.

## A. Immediately: re-check the existing development ECGs (no new patients)
**Files:** `data/private/fixes/recheck_worklist.csv` (one row per ECG, sorted by priority) and the fill-in template
`data/private/fixes/verified_echo_template.csv`. Both are git-ignored and stay on the project machine; do not email them
unencrypted.

| Step | Who | What |
|---|---|---|
| A1. Re-link | Medical records | For each worklist row, find the patient's MRN / UHID for the ECG and for the echo. Mark `mrn_verified` yes / no. If the linked echo belongs to someone else, find the correct echo (same patient, closest date, ideally the same admission). |
| A2. Re-measure | Echo lab (cardiologist / sonographer) | Re-measure EF by **biplane Simpson** from the stored echo images. Blind the reader to the current label and to any model output. Fill `ef_simpson`; keep `ef_teichholz`; mark `regional_wma` (regional wall-motion abnormality) yes / no. |
| A3. Return | Owner | Run `python -m research_experiments.phase_6b_threshold_sampling_lstm.fixes --ingest <filled.csv>`. The model is retrained and re-evaluated on the verified labels and compared with the current labels on the same ECGs. |

**Order of work:** high priority first (197 ECGs; Moderate 78, Mild 70), then medium (266). The high tier covers the
ECGs whose class is most likely to change: EF within 3 points of a limit, label vs echo disagreement, weak name match
or a distant echo.

**Rules that keep the result honest:**
- re-read without knowing the model's prediction;
- use every re-read value (never only the ones that "help");
- never relabel an ECG to match the model.

## B. New data (needs ethics approval and a data-sharing agreement)
| Item | Why | Specification |
|---|---|---|
| B1. Patient identifier on both sides | Removes name-matching errors: clean links agree with the ECG at r 0.75 vs 0.42 for distant links | MRN / UHID (pseudonymised by the hospital before export), ECG date-time, echo date-time, admission / encounter ID |
| B2. Biplane Simpson EF | Teichholz is unreliable with regional wall-motion abnormalities (common after a heart attack) | Simpson biplane EF, LVEDV, LVESV; Teichholz EF if measured; regional wall-motion score; image quality; reader ID |
| B3. Digital ECGs | The printouts give 2.5 s per lead, not simultaneous | Native export from the ECG machine or management system: 10-s, 12-lead, simultaneous, ≥ 500 Hz (e.g. HL7 aECG / FDA XML, DICOM-ECG, SCP-ECG, or the vendor XML such as GE MUSE or Philips). The existing PDFs come from the same machines, so exports usually exist for the same patients. |
| B4. Timing | Echo and ECG must describe the same heart state | Echo within ±7 days of the ECG, ideally the same admission; no intervening MI, revascularisation or device implant |
| B5. More patients, enriched for Mild / Moderate | The learning curve gains +1.4 points per doubling; Mild is the scarcest class | Target ≥ 3,000 ECG-echo pairs with ≥ 600 each of Mild (41–52 / 54) and Moderate (31–41); several years and, if possible, a second hospital (external validation) |
| B6. Clinical covariates (optional, separate analysis) | Information the ECG does not carry | NT-proBNP (same admission), troponin, prior MI / heart failure, hypertension, diabetes, current medications (beta-blockers, ACE-i / ARB / ARNI, MRA, SGLT2-i) |

**Privacy:** names are not needed once the MRN / UHID is available. The hospital should pseudonymise identifiers before
export and keep the key.

## C. Public data (optional, for pre-training only)
The EchoNext dataset (PhysioNet, credentialed access; the owner already has access) provides about 100,000 digital
12-lead ECGs paired with echo findings. A model pre-trained there to predict reduced EF could then be fine-tuned on this
hospital's data. This is a separate, larger project:
- the download is about 18 GB of waveforms;
- it needs the GPU laptop;
- it must stay on a separate branch.

## What success would look like
The ceiling simulation (`MILDFIX_REPORT.md`) indicates the targets:

| ECG–echo EF error SD | Expected four-class accuracy |
|---|---|
| 10 points | about 0.68 |
| 8 points | about 0.73 |

A1–A3 alone should narrow the label part of the error; B1–B5 attack both parts.
