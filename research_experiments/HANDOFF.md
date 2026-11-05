# HANDOFF: ECG-based 4-class LVEF prediction research (state at the END of Phase 4; all phases complete)

Read this file first, then `phase_4/reports/PHASE4_FINAL_REPORT.md`, `DATA_ACCESS.md`, `EXPERIMENT_INVENTORY.md`,
`RESUME.md`, `phase_4/reports/PHASE4_CLAIMS_REGISTER.md`, `phase_3/reports/CLAIMS_REGISTER.md` and
`reports/ERRATUM_phase1.md`.

> **Repository visibility:** on 2026-11-05 this GitHub repository was **public**, and it has one public fork.
> `main` tracks the patient-named ECG PDFs, and its history contains the deleted `ECG_ECHO` sheets. Phase 4
> commits contain aggregates only. Changing visibility is the owner's decision.

**Do not re-run completed experiments.** Every result listed in `EXPERIMENT_INVENTORY.md` exists on disk and in
git; the run functions skip existing outputs.

## 1. Objective and architecture
- **Task:** predict the echocardiographic LVEF class (Normal / Mild / Moderate / Severe) from a 12-lead ECG PDF
  printout.
- **Owner's targets** (aspirational): accuracy ≥ 60 %, balanced accuracy ≥ 50 %, macro-F1 ≥ 0.48, Mild F1 ≥ 0.35,
  QWK ≥ 0.60.
- **Inputs** (all derived from the PDF; nothing else):
  - **header numbers:** age, sex and 14 machine measurements (16 features);
  - **ECGFounder embedding:** frozen public foundation model, Li et al. NEJM AI 2025, MIT licence. It takes the
    12-lead waveform recovered exactly from the PDF vector paths, in the time-aligned "aligned" construction,
    and outputs a 1,024-d mean-pooled last-stage embedding;
  - **clinical features:** 28 features from waveform + header;
  - **image CNN:** ResNet-18 on the grid-filtered page image.
- **Best overall model** (Phase 3): the equal-weight **vote ensemble** of
  - (a) MLP fusion of header numbers ‖ ECGFounder (3 seeds),
  - (b) ECGFounder linear head with distance-penalised CE,
  - (c) logistic regression on header numbers + clinical features.

## 2. Branches and commits (never force-push, never merge into the original branch)
| Branch | State |
|---|---|
| `claude/funny-gauss-udlmv0` | original project, `c2487308`; **untouched** |
| `research/ecg-ef-model-improvements` | phases 1–3; last pushed `39aa39f` (Phase-3 report still a draft there) |
| `research/ecg-ef-phase4` | local only, created from `39aa39f` |
| `research/phase4-handoff` | **this branch, pushed**: `39aa39f` + final Phase 3 + complete Phase 4 (4A–4H, reports, aggregates); milestone commits, no force-push |

## 3. Dataset accounting and splits [VALID]
- **Total:** 1,047 ECG PDFs from 993 patients (anonymous hash `data/manifest.csv:patient`). Folder labels:
  Normal 480 / Mild 232 / Moderate 221 / Severe 114. Normal is capped at 120 per year (2020, 2021, 2022, 2024).
- **Original split** (patient-grouped): train 669 / validation 168 / test 210 ECGs (635 / 159 / 199 patients).
  **"669" is the training split, never the total.**
- **Phase 2–4 protocol:**
  - development set **D** = train + validation = **837 ECGs / 794 patients**;
  - **locked test T = 210 ECGs / 199 patients**; 0 patients overlap;
  - T was inspected for ~40 models in Phase 1, so it is **exposed** and has **not** been scored since.
- **Folds:** `phase_2/configs/folds_D.csv`, StratifiedGroupKFold(5) on patients, repeats 0/1/2 (random_state
  2025 + r).
  - Every base model trains on inner-train and selects / early-stops on inner-val (≈ 535 / 134 ECGs per fold).
  - Post-hoc calibration is **fully nested** (an inner CV inside each outer training portion).
- **Integrity:** SHA-256 snapshots in `phase_4/results/integrity_phase4_before.json` (keys are hashed paths, no
  names). Last verification: 0 of 4,648 changed.

## 4. Results (development CV; mean over runs; full tables in `phase_3/reports/PHASE3_REPORT.md`)
| Model | Runs | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|---|
| Header + ResNet-18 (original default) | 1 repeat | 0.495 | 0.444 | 0.389 | 0.193 | 0.491 | 0.442 |
| ECGFounder frozen, linear | 3 | 0.528 | 0.442 | 0.434 | 0.244 | 0.543 | 0.468 |
| 2-way MLP fusion header + ECGFounder | 9 | 0.542 | 0.465 | 0.436 | 0.176 | 0.579 | 0.481 |
| **Vote ensemble (recommended, best overall)** | 3 | **0.547 ± 0.002** | **0.467** | **0.455** | **0.255** | 0.574 | **0.490** |
| MLP fusion + nested protected prior calibration (**best accuracy**) | 3 | **0.568** | 0.463 | 0.442 | 0.159 | 0.591 | – |
| Phase 4: vote + EF-regression member (best Phase 4 accuracy) | 3 | 0.560 | 0.458 | 0.446 | 0.226 | 0.578 | 0.488 |
| Phase 4: training without folder ≠ echo ECGs (best Phase 4 S3) | 9 | 0.554 | 0.471 | 0.449 | 0.197 | 0.586 | 0.491 |
| Phase 4: multitask EF + class, λ selected on inner-val | 9 | 0.546 | 0.464 | 0.444 | 0.211 | 0.583 | 0.484 |
| Phase 4: EF regression → class (best, tuned cut-points) | 9 | 0.546 | 0.435 | 0.432 | 0.248 | 0.580 | 0.471 |

- **Best accuracy: 56.8 %** (calibrated MLP fusion). It sacrifices Mild: F1 0.159, and predicts Normal 56 % vs a
  true 46 %.
- **Best overall:** vote ensemble, **accuracy 54.7 %, balanced accuracy 46.7 %, macro-F1 0.455**. Note: 54.7 % is
  accuracy, not balanced accuracy.
- **Statistics:**
  - nothing survives Holm correction;
  - vote vs original default: +5.4 accuracy (p = 0.013), +6.9 macro-F1 (p = 0.038);
  - vote vs MLP fusion: Mild F1 +0.083 (p = 0.014, nominal).
- S3 = (accuracy + balanced accuracy + macro-F1) / 3, the pre-registered selection score.

### Completed but unsuccessful (do not repeat without new evidence)
- gradient boosting on header / clinical / engineered features;
- attention fusion and vector gates;
- stacking (unstable across repeats);
- per-class threshold calibration (collapses Mild);
- ECGFounder fine-tuning (equal to frozen);
- ordinal heads (CORAL / cumulative link): lower accuracy;
- clinical features inside the fusion: Δ S3 +0.002, p = 0.92;
- image CNN added to the ensemble: no gain;
- masking alternatives (per-window, multiview): none beats "aligned".

## 5. Phase 4: COMPLETE (pre-registered; `phase_4/configs/PREREG_PHASE4.md`, commit `25e1d1f`)

**Audit (4A–4C)** (`PHASE4_DATA_AUDIT.md`, `PHASE4_LABEL_QUALITY_REPORT.md`):
- **Where EF lives:** the 4 authorised sheets are file-copy logs. EF lives in `ECG_ECHO` (git history), linked by
  **fuzzy name only**.
- **Agreement:** 93.5 % folder vs nearest-echo agreement: 98.0 % for single-link ECGs, 67.7 % for multi-link.
- **Measurement uncertainty:** not quantifiable (18 repeat pairs).
- **Cohort G** (357 dev ECGs / 345 patients):
  - its 95.8 % agreement is largely matching consistency, not independent validation;
  - every model is about 7 points more accurate on G, an evaluation-set effect.

**Models (4D–4H)** (`PHASE4_REGRESSION_REPORT.md`, `PHASE4_MULTITASK_REPORT.md`, `PHASE4_MODEL_BENCHMARK.md`):
- **4D EF regression:**
  - r 0.66 / R² 0.43 on all ECGs; r 0.75 / R² 0.55 on G; MAE 10.4 vs 15.6 for the mean predictor;
  - regression → class does not beat direct classification, and clinical thresholds suppress Severe recall
    (0.10–0.12);
  - an error-budget simulation shows the targets need r ≈ 0.80.
- **4E multitask:** ΔS3 +0.001 to +0.007 vs the λ = 0 control (n.s.). The control reproduces the Phase 2 MLP fusion
  exactly.
- **4F:**
  - label-quality training (G only, cleaned labels, EF-derived labels): no gain;
  - vote + regression member: accuracy 0.560 (n.s.), macro AUROC +0.013 (Holm p 0.07).
- **4G Mild** (`PHASE4_MILD_CLASS_ANALYSIS.md`):
  - errors go both ways and concentrate within 5 EF points of a boundary;
  - Mild ranking AUROC 0.61 vs 0.55 is a hypothesis (p 0.043, uncorrected);
  - the Mild F1 target was not reached.
- **4H:** 0 of 26 confirmatory models genuinely improve on the vote ensemble (smallest Holm p 0.25). **The vote
  ensemble remains the recommended model.**

**Deliverables:**
- 8 reports;
- the claims register (P4-1 to P4-23 + C-1);
- `phase_4/results/phase4_results.xlsx` (filterable) and `phase4_tables.md`;
- plots in `phase_4/plots/`.

## 6. Environment
```bash
pip install -r research_experiments/requirements-research.txt
```

- Python 3.13.18, CPU-only, 4 cores, Linux. Versions pinned: torch 2.15.0, timm 1.0.30, numpy 2.5.3,
  pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1, opencv 5.0.0, pymupdf 2.1, matplotlib 3.11.2,
  openpyxl 3.1.5, joblib 1.6.0.
- Use `OMP_NUM_THREADS=1` for parallel single-thread jobs. Oversubscription slowed runs up to 4×.

## 7. Artefacts not in git (regenerate; do NOT retrain models)

**Project cache `data/`** (git-ignored by the original project; needed by every script that reads
`data/manifest.csv`, `data/header_features.csv` or `data/signals.npz`). A fresh clone does not have it. Rebuild it
with the **safe** script below, which writes only into `data/` and verifies the recorded SHA-256 values:

```bash
python -m research_experiments.regenerate_data_cache
```

- **Do NOT use** `python -m ecg_ef.preprocess` (rewrites `results/split.csv`) or `python -m ecg_ef.header`
  (re-trains and overwrites `models/header_logreg.joblib`).
- Add `--images` only if you need the filtered page-image cache (image CNN).

**ECGFounder 12-lead weights** (needed for any new waveform feature) → `research_experiments/models/pretrained/`,
sha256 `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`:

```bash
curl -L -o research_experiments/models/pretrained/12_lead_ECGFounder.pth https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
```

**Frozen features:**
- `phase_2/models/ecgf_aligned_features.npz` is **committed** (14.6 MB). Phase-4 work needs only this plus the
  header features.
- Other feature variants and the activation caches: `python -m research_experiments.phase_2.ecgf_features`
  (≈ 5 min).

**Clinical features** (small CSV, regenerable in ≈ 6 min):
```bash
python -m research_experiments.phase_3.clinical_features
```

**Private Phase-4 tables:** `DATA_ACCESS.md` (≈ 1 min).

**Phase-4 per-ECG predictions** (`phase_4/data/private/runs/`, git-ignored because they are keyed by ECG id):
- A fresh clone has only the committed aggregates (`phase_4/results/runs/*.json`).
- `phase_4.p4common.done()` requires both files, so re-running the scripts regenerates the predictions.
- Regeneration is deterministic (checked: byte-identical OOF), CPU-only, about 1.5 h on 4 cores.
- The commands are listed by `python -m research_experiments.phase_4.status4`.

**Model checkpoints** (`*.pt`, 42–102 MB) are **not** in git. Their predictions (OOF + locked-test
probabilities) **are**. They are needed only for inference on *new* ECGs; the SHA-256 of each is listed in
`EXPERIMENT_INVENTORY.md`.

## 8. Reproduce existing tables (no training; reads saved predictions)
```bash
python -m research_experiments.status
```
```bash
python -m research_experiments.phase_3.p3_robust
```
```bash
python -m research_experiments.phase_3.calibration_summary
```
```bash
python -m research_experiments.phase_3.confusion_collection
```

Phase 4 (needs the private data and per-ECG predictions; see §7):
```bash
python -m research_experiments.phase_4.benchmark4 --finalize
```
```bash
python -m research_experiments.phase_4.make_tables4
```

Experiment → result mapping: `EXPERIMENT_INVENTORY.md`.

## 9. Rules (methodological restrictions and leakage prevention)
1. Same 1,047 ECGs, folder labels and patient-grouped splits. Never modify `LV_ECG_Dataset_*`, `data/`,
   `results/`, `models/` or `ecg_ef/`.
2. Never print or store patient names, dates of birth or identifying values. Mask text when inspecting sheets.
   Committed outputs contain aggregates only.
3. The locked test T is never used for any choice. Score it once, for a pre-registered shortlist, **only with
   the owner's explicit approval**.
4. All preprocessing, feature selection, hyper-parameters, calibration and thresholds are chosen inside the
   training folds. Thresholds come from inner-CV OOF predictions (nested).
5. Statistics:
   - corrected repeated-CV t-test (Bouckaert–Frank) on matched (repeat, fold) cells, plus a patient-cluster
     bootstrap;
   - Holm within metric;
   - report mean ± SD over seeds × repeats, never a single best run.
6. Regression targets use the **label-agnostic nearest echo**, never a label-consistent pick (that would leak
   the folder label).
7. No commit or push without the owner's explicit approval.

## 10. Next steps for the incoming agent (Phase 4 is complete)
1. **Set up and check state.**
   - Install the requirements (§6) and rebuild `data/` with `python -m research_experiments.regenerate_data_cache`
     (§7).
   - Recover the private data (`DATA_ACCESS.md`).
   - Run `python -m research_experiments.status` (phases 2–3), `python -m research_experiments.phase_4.status4`
     (Phase 4) and the integrity check:
   ```bash
   python -m research_experiments.phase_4.integrity4 verify phase4_before
   ```
2. **Do not** start new architecture searches on the same 837 ECGs. Phase 4 shows the limitation is information and
   data (`PHASE4_FINAL_REPORT.md` Q4, Q12, Q13).
3. **Owner decisions needed:**
   - repository visibility (§ top note);
   - whether to score the pre-registered shortlist once on the **exposed** locked test: vote ensemble; vote + MLP
     EF-regression member; calibrated MLP fusion;
   - preferably, obtain an **independent temporal or external cohort**.
4. **If new data arrive** (verified ECG–echo IDs, more abnormal-EF ECGs, digital waveforms), re-use:
   - the Phase 4 pipeline (`phase_4/regression.py`, `multitask.py`, `benchmark4.py`);
   - the pre-registration format.
