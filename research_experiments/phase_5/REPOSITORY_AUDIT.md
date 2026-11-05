# Phase 5 repository audit (completed before any Phase 5 model development)

Audit date: 2026-11-05.

## 1. Branch, remote, commits
- **Branch:** Phase 5 works on **`research/phase5`**, created from `research/phase4-handoff` at `790603f`.
- **Remote:** `origin` = `github.com/krithikareddy22/ECG-Image-Based-Prediction-EF-hear-attack-12-Lead`.
  **Visibility: PUBLIC** (checked 2026-11-05), with 1 public fork.
- **Remote branches (unchanged by Phase 5):**

  | Branch | Commit |
  |---|---|
  | `claude/funny-gauss-udlmv0` (original project) | `c2487308` |
  | `main` | `9f2a8833` |
  | `research/ecg-ef-model-improvements` | `39aa39fd` |
  | `research/phase4-handoff` | `790603fe` |

- **Latest commits (Phase 4):**
  - `790603f` handoff docs;
  - `3024093` reports;
  - `504afbc` benchmark;
  - `11a20df` label quality;
  - `3645cf5` multitask;
  - `294b9fb` regression;
  - `25e1d1f` pre-registration.
- **Working tree at the start:** clean.

## 2. Phase 4 completion status [VERIFIED]
- `phase_4/status4.py`: all 8 steps DONE.
- Reports: 8 of 8 present.
- Claims register: P4-1 to P4-23 plus correction C-1.
- Phase 4 aggregates: 360 evaluation rows in `phase_4/results/runs/*.json`, 68 seed-combined models in
  `phase_4/results/phase4_benchmark.csv`.
- Per-ECG Phase 4 predictions exist locally in `phase_4/data/private/runs/` (git-ignored).

## 3. The four-class task as implemented [VERIFIED]
- **Labels:** the folder label of each PDF (`LV_ECG_Dataset_<year>/<class>/`), mapped Normal = 0, Mild = 1,
  Moderate = 2, Severe = 3 (`ecg_ef/config.py`, `data/manifest.csv:label_idx`).
- **Thresholds:** the folders were produced with the hospital rule recovered in Phase 4 (`phase_4/audit_matching.py`
  `classify`):

  | Class | Men | Women |
  |---|---|---|
  | Normal | EF > 52 | EF ≥ 54 |
  | Mild | 41 ≤ EF ≤ 52 | 41 ≤ EF < 54 |
  | Moderate | 31 ≤ EF < 41 | 31 ≤ EF < 41 |
  | Severe | EF < 31 | EF < 31 |

  The rule is kept unchanged. No conventional thresholds are substituted.
- **Counts** (all 1,047 ECGs): Normal 480 / Mild 232 / Moderate 221 / Severe 114, matching the specification.
- **Development set D** (all modelling): **837 ECGs / 794 patient keys**; Normal 384 / Mild 185 / Moderate 177 /
  Severe 91.
- **Locked test T:** 210 ECGs / 199 patient keys (96 / 47 / 44 / 23).

## 4. Is the 56.8 % benchmark comparable? [VERIFIED]
**Yes:** it is a **four-class** accuracy on the **same folder labels** and the **same 837 development ECGs**.

| Property | Value |
|---|---|
| Source | `phase_2/results/nested_calibration_fusion_mlp_r{0,1,2}.json` |
| Confusion matrices | 4 × 4, n = 837, row sums 384 / 185 / 177 / 91 |
| Model | MLP fusion (header numeric ‖ ECGFounder f6), seed mean of 42 / 43 / 44, fully nested class-prior calibration ("protected" objective) |
| Accuracy per repeat | 0.571 / 0.569 / 0.565 |
| Mean accuracy | **0.568** |

**Caveats:**
- It is **development-CV** (outer-fold) accuracy, not a test-set result.
- Its **Mild recall is 0.08–0.16** (mean 0.11). Mild is nearly undetectable, so under the Phase 5 admissibility rule
  (`PHASE5_PLAN.md` §3) it is a reference, not an admissible winner.
- Its per-ECG predictions were **not stored**. Phase 5A reproduces it with stored (git-ignored) predictions so that
  paired tests become possible.

The other reference point, **"54.7 %"**, is the **accuracy** of the vote ensemble on the same population (balanced
accuracy 46.7 %, macro-F1 0.455). It is also four-class.

## 5. Data and splits [VERIFIED]
| Artefact | Location | Status |
|---|---|---|
| Manifest (ids, labels, patient keys) | `data/manifest.csv` | present, sha256 `2463bd35…` (matches Phase 4) |
| Split (train / val / test) | `results/split.csv` | present, sha256 `7fc76b0e…`, unchanged |
| Development CV folds | `phase_2/configs/folds_D.csv` | StratifiedGroupKFold(5) by patient key, repeats 0–2 (random_state 2025 + r); inner-train / inner-val per fold |
| Waveforms | `data/signals.npz` | 12 leads × 1,250 samples (2.5 s each at 500 Hz) + lead II rhythm 5,000 samples (10 s); digitised from the PDF vector paths |
| Header measurements | `data/header_features.csv` | age, sex, rate, RR, PR, QRSD, QT, QTcB, QTcF, P / QRS / T axes, printed statements |
| ECGFounder embedding | `phase_2/models/ecgf_aligned_features.npz` | committed |
| ECGFounder weights | `models/pretrained/12_lead_ECGFounder.pth` | present (not in git) |
| Phase 3 clinical features | `phase_3/models/clinical_features.csv` | 28 features (QRS voltage, LVH criteria, R-wave progression, rhythm, T-wave) |
| Private EF / cohort table | `phase_4/data/private/cohorts.csv` | git-ignored; used for **analysis only**, never as model input |

**Equivalence check:** the Phase 5 development-only feature construction (`phase_5/p5common.make_X`) equals the
Phase 2 construction for the development rows (max |ΔX| = 2.8 × 10⁻⁷, float32 rounding; outer fold r0 k0).

## 6. Patient grouping, repeats and duplicates (`data_audit/grouping_audit.json`) [VERIFIED counts]
- **Patient key:** `sha1(letters-only upper-case name)[:12]`. **There is no medical-record number.**
- **ECGs per patient:**

  | Set | 1 ECG | 2 ECGs | 3 ECGs |
  |---|---|---|---|
  | Development | 756 | 33 | 5 |
  | Locked test | 190 | 7 | 2 |

  Folds keep all ECGs of a key together (asserted in `phase_2/protocol.py`).
- **Duplicate recordings:** 0 exact and 0 near-duplicate (r ≥ 0.98) 10-s rhythm strips.
- **Possible split identities:** **29 pairs of near-identical names under different keys** (difflib ratio ≥ 0.90 or
  name containment):
  - **15 pairs inside development**, all in different outer folds in at least one repeat. This is possible same-person
    leakage across CV folds for at most about 30 ECGs (3.6 %); it is quantified in the error analysis;
  - **11 pairs straddling development and the locked test**. Relevant to any future locked-test evaluation:
    recommend resolving them before the authorised evaluation;
  - 3 pairs inside the test set.
- **Consequence:** validation is patient-grouped **by name key**. Patient independence cannot be guaranteed without
  an identifier. Phase 5 claims "name-key-grouped" validation and treats these pairs as a documented limitation.
- **ECG–echo matching:** the labels themselves come from fuzzy name matching (Phase 4: 93.5 % agreement with the
  nearest echo). Echo EF is never a model input.

## 7. Existing models and completed experiments (reusable without retraining) [VERIFIED]
| Family | Per-ECG OOF predictions | Runs |
|---|---|---|
| Phase 1 (fixed split, test exposed) | `research_experiments/results/*` | about 40 models; not used for selection |
| Phase 2 (development CV) | `phase_2/results/<tag>/oof.csv` (tracked) | 132 OOF files: MLP fusion (9 runs), ECGF linear / MLP, header LR, imbalance and ordinal losses, HP search incl. unweighted CE, balanced sampler, dropout, hidden size |
| Phase 3 | `phase_3/results/<tag>/oof.csv` (tracked) | 84 files: clinical / 3-way fusion, gated, ensembles (vote / wavg / stack), masking, image CNN (r0) |
| Phase 4 | `phase_4/data/private/runs/<tag>/oof.csv` (git-ignored) | 360 rows: EF regression, multitask, label-quality, fusions |
| Nested calibration | aggregates only | MLP fusion, ECGF linear, header LR (3 repeats); vote ensemble (3 repeats) |

- **Benchmark models:**
  - **vote ensemble:** `phase_3/results/ens_mlp2+ecgflin+hdrclin__vote{,_r1,_r2}`;
  - **MLP fusion:** `phase_2/results/ff_hdrnum+ecgf_mlp{,_seed43,_seed44}{,_r1,_r2}`;
  - **calibrated MLP:** aggregates only, reproduced in 5A;
  - **EF regression and multitask:** Phase 4 private runs.
- **Training-strategy runs already done in Phase 2 (re-evaluated, not repeated, in E7):**
  - class-weighted vs unweighted CE (`ff_mlp_unweighted*`);
  - balanced sampler (`ff_mlp_balsampler*`);
  - input/hidden dropout 0.3 / 0.7 (`ff_mlp_do*`);
  - hidden 32 / 128;
  - distance-penalised loss;
  - focal loss, label smoothing and feature noise on the ECGF-only heads.

## 8. Locked-test protection
- **Earlier phases:** Phase 2 / 3 run functions (`cv_features.run`, `p3common.run3`) **predicted** the locked test and
  stored probabilities, never scored, in `phase_2/locked_test/` (127 files) and `phase_3/locked_test/` (80 files).
  Phase 4 code never predicted it.
- **Phase 5 mechanism:**
  1. `p5common` reads the manifest once and drops all test rows before anything else.
  2. Phase 5 never imports the modules that load all ECGs at import time (`phase_2.cv_features`,
     `phase_2.accuracy_tradeoff`, `phase_3.p3common`, `phase_3.p3_robust`, `phase_4.p4common`); a test asserts this.
  3. No Phase 5 code reads `phase_2/locked_test/` or `phase_3/locked_test/` (test: static scan).
  4. Every saved prediction file is checked to contain exactly the 837 development ids.
- **Exception:** locked-test **names and signals** (never labels or predictions) were read once, for the
  patient-overlap and duplicate counts in §6.

## 9. Privacy and Git risks
1. **Public repository.** `main` tracks the 1,047 PDFs with patient names in their file names, and its history
   contains the deleted `ECG_ECHO` sheets (names, ages, EF).
2. **Per-ECG predictions keyed by ECG id are already tracked and public:**
   - 132 Phase 2 + 84 Phase 3 + 130 Phase 1 result CSVs;
   - **207 locked-test probability files**.

   Phase 5 adds none. Removing them would modify Phases 1–3 (not allowed without the owner's decision) and would
   not remove them from history.
3. **Phase 5 rule:** per-ECG outputs go to `phase_5/data/private/` (git-ignored by the root `data/` rule and by
   `phase_5/.gitignore`). Every commit is scanned for all 1,047 ECG ids, acquisition timestamps and patient hashes
   (`p5common.privacy_scan`).
4. **Integrity:** the new snapshot `integrity5 snapshot phase5_before` covers the originals and all of Phase 1–4
   (including `phase_4/`). It is verified before each push.

## 10. Environment changes
- `PyWavelets 1.8.0` and `lightgbm 4.7.0` were installed for 5C / E2. They are pinned in
  `phase_5/requirements-phase5.txt`.
- Everything else is as pinned in `research_experiments/requirements-research.txt`.

## 11. Conclusion
The four-class implementation, labels, populations and folds are consistent across Phases 2–4, and the 56.8 %
benchmark is a valid four-class development-CV reference. Phase 5 can proceed additively.

**Main threats to validity:**
- name-based patient grouping (29 possible split identities);
- an exposed locked test;
- a public repository with previously published per-ECG predictions.
