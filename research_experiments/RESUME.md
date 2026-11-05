# RESUME HERE: hand-off for any agent or person continuing this research

> **Current state (2026-11-05): phases 1–4 are COMPLETE.**
> - Start with `HANDOFF.md` (§5 Phase 4, §10 next steps) and `phase_4/reports/PHASE4_FINAL_REPORT.md`.
> - Branch: `research/phase4-handoff`, pushed.
> - The sections "FORKING" and "CURRENT STATE (… 19:15)" below are **historical**, kept unchanged as the
>   Phase 3 record.

Run `python -m research_experiments.status` first. It prints DONE/TODO for every step and the next
command. Then run `bash research_experiments/resume_pipeline.sh`. Every step skips finished work:
- `cv_features.run`, `p3common.run3` and `exp_p3` return early when `cv_metrics.json` exists;
- `cv_image_cnn` skips folds whose `foldK.npz` exists;
- `masking_audit` skips feature files that already exist.

## Rules (from the project owner; keep them)
- **Dataset freeze:** the same 1,047 ECGs (837 development / 210 locked test), labels and patient
  groups. Never modify `LV_ECG_Dataset_*`, `data/`, `results/`, `models/` or `ecg_ef/`.
- **Spreadsheets:**
  - *Phase 1–3 rule:* never open `LV_ECG_Dataset_*/*.xlsx`.
  - *Phase 4 (owner authorisation):* the 4 sheets and the `ECG_ECHO` matching files may be read, read-only.
    Extract into the git-ignored `phase_4/data/private/` only and never print names (`DATA_ACCESS.md`).
- **Locked test** (`phase_2/locked_test/`, `phase_3/locked_test/`): probabilities only, never scored.
  It is scored once, for a pre-registered shortlist, **only with the owner's explicit authorisation**.
- **Selection:** on development CV only (`phase_2/configs/folds_D.csv`, repeat 0 primary, repeats
  1–2 for robustness).
- **Calibration:** thresholds fully nested (`phase_2/nested_calibration.py`).
- **Git:**
  - *Phase 1–3:* branch `research/ecg-ef-model-improvements`.
  - *Phase 4:* branch `research/phase4-handoff`.
  - The owner authorised commits and pushes of verified **non-sensitive** research files (aggregates, code, docs).
    Everything per-ECG stays git-ignored.
  - Never force-push, rewrite history, delete branches or change repository access.
- **Patient names:** never print patient names. `data/manifest.csv:source` contains them; use
  `common.load_meta()`, which drops that column.
- **Verify integrity at the end:**
  - `python -m research_experiments.phase_2.integrity2 verify phase2_before`;
  - Phase 4: `python -m research_experiments.phase_4.integrity4 verify phase4_before` (0 changed / 0 missing on
    2026-11-05).

## Where things are
| What | Where |
|---|---|
| Phase-1 report + erratum | `reports/FINAL_REPORT.md`, `reports/ERRATUM_phase1.md` (the phase-1 p < 0.001 CV claim is **retracted**) |
| Claims status | `phase_3/reports/CLAIMS_REGISTER.md` |
| Phase-2 protocol / pre-registration | `phase_2/protocol.py`, `phase_2/configs/preregistration.md` (+ sha256) |
| Phase-2 audits | `phase_2/results/audit_*.json`, `dataset_reconciliation.json`; `phase_2/reports/AUDIT_AND_STATUS.md` |
| Phase-2 CV runs | `phase_2/results/<tag>/{oof.csv, info.json, cv_metrics.json}` |
| Phase-3 code / runs | `phase_3/*.py`; `phase_3/results/<tag>/` |
| Dataset accounting | `phase_3/results/dataset_accounting.json` |

## Key results so far (development CV, 837 ECGs; locked test NOT scored)
| Model | S3 | Accuracy |
|---|---|---|
| Header numeric LR | 0.432 | 0.485 |
| Header numeric + clinical LR | 0.468 | 0.520 |
| ECGFounder frozen, linear | 0.470 | 0.532 |
| 2-way MLP fusion (header + ECGFounder) | 0.479–0.505 over seeds (repeat 0); repeat 1: 0.476 | 0.536–0.559 |
| 3-way MLP fusion (+ clinical) | 0.481 / 0.489 (seeds 42 / 43) | ≈ 0.54 |
| Gated 3-way + distance penalty | 0.482 (seed 42) | 0.536 |
| Gradient boosting / attention fusion / fine-tuning | no gain | – |

## FORKING / CONTINUING ON ANOTHER ACCOUNT
- Fork the repository and check out branch **`research/ecg-ef-model-improvements`**. All research lives in
  `research_experiments/`.
- `python -m research_experiments.checkpoint "<msg>"` pushes to `origin`; in a fork, that is your fork.
- Not in git, so regenerate first (see "If the container was lost"):
  - ECGFounder weights;
  - `phase_2/models/*` features and activation caches;
  - `phase_3/models/*` (clinical and masking features).
- Status at the last push (19:15 session clock), jobs that were STILL RUNNING and must be re-run:
  - image-CNN fold 4: re-run `phase_2.cv_image_cnn`; it resumes at fold 4, ≈ 25 min on 3 CPU threads;
  - vote-ensemble nested calibration, repeat 2: re-run `phase_3.nested_calib_vote --repeat 2`; r0 and r1 are done;
  - then the steps that need the CNN: `phase_2.run_fusion --stage combos`, `phase_3.p3_ensembles`
    (image combos), `phase_3.p3_robust`, `phase_3.calibration_summary`, `phase_3.confusion_collection`;
  - then fill the report placeholders.

## CURRENT STATE (updated 2026-11-04 19:15 session clock)
- Pushed checkpoints on `research/ecg-ef-model-improvements`: 069c2bf, 22aa38f, 4b846cf (+ the latest one).
- **Measured headline (development CV, locked test NOT scored):**
  - best overall = vote ensemble (2-way MLP fusion + ECGFounder linear + distance penalty +
    header + clinical LR): accuracy 0.547 ± 0.002, balanced 0.467, macro-F1 0.455, Mild F1 0.255,
    S3 0.490 ± 0.002;
  - best accuracy = MLP fusion + nested protected prior calibration: 0.568, with balanced / macro-F1
    unchanged and Mild F1 −0.027;
  - 3-way vs 2-way fusion: no incremental information (Δ S3 +0.002, p = 0.92);
  - masking is not limiting (lead coverage matters, observed time does not).
- **Report draft:** `phase_3/reports/PHASE3_REPORT.md`. It contains `*_PLACEHOLDER` markers to fill
  from these files:

  | Placeholder | Fill from |
  |---|---|
  | `BAL_R_PLACEHOLDER`, `MF1_R_PLACEHOLDER`, `BASELINE_TABLE_PLACEHOLDER` | `phase_3/results/p3_robust_tables.md`; re-run `phase_3.p3_robust` after the CNN finishes and the reference R exists |
  | `VOTE_CAL_PLACEHOLDER`, `VOTE_CAL_TABLE_PLACEHOLDER` | `phase_3/results/nested_calibration_vote_r{0,1,2}.json` (mean over repeats: original vs "calibrated: prior, protected") |
  | `ENSEMBLE_EXTRA_PLACEHOLDER` | `phase_3/results/ens_*` `cv_metrics.json` (other combos, incl. `ens_mlp2+image` / `ens_all+image` once the CNN exists) |
  | `COMPLEMENTARITY_PLACEHOLDER` | run `phase_2.complementarity --ecgf ens3__ecgf_mlp_r0 --fusion ens3__fusion_mlp_r0`, then read `phase_2/results/complementarity.json` |
  | `FINAL_MODEL_PLACEHOLDER` | leave as "full-development refit of the vote ensemble not yet trained", unless it was done |

- **Image-CNN CV:** folds 0–3 done (in `phase_2/results/img_resnet18_baseline_recipe/fold*.npz`);
  fold 4 was in progress. Re-running `cv_image_cnn` resumes at the first missing fold.
- **When everything is filled:** run `python -m research_experiments.checkpoint "Checkpoint: phase 3 report final"`
  (the owner authorised milestone checkpoints to the research branch only), then
  `python -m research_experiments.phase_2.integrity2 verify phase2_before`, and report to the owner.

## Remaining steps (in order; all in `resume_pipeline.sh`)
1. Image-CNN CV folds 3–4 (about 25 min per fold on 3 threads).
2. Matched 3-way runs and gated-dist runs over seeds × repeats.
3. Masking dose-response CV (`phase_3.masking_cv`).
4. Header + image fusions, including the reference R (`phase_2.run_fusion --stage combos`), once the
   CNN `oof.csv` exists.
5. `phase_3.p3_robust` (tables and paired tests, including the 3-way vs 2-way question).
6. Nested calibration (`phase_2.nested_calibration`) for `fusion_mlp`, repeats 0–2.
7. Complementarity (`phase_2.complementarity --ecgf ecgf_aligned_f6_mlp --fusion ens3__fusion_mlp_r0`).
   Ensembles: build from the OOF files with `phase_2.fusion.run`.
8. Phase-3 report `phase_3/reports/PHASE3_REPORT.md`. Separate MEASURED results from HYPOTHESES and
   estimated gains; compare 3-way vs 2-way on matched folds, seeds and repeats.
9. Final model on all development data (`phase_2.final_model`), then ask the owner before any
   locked-test scoring.

## If the container was lost (only the GitHub branch survives)
Regenerate the git-ignored artefacts. All are deterministic and need no labels:
1. **ECGFounder 12-lead checkpoint:** download
   https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
   into `research_experiments/models/pretrained/` (sha256 `ee199f37…a7997`).
2. **1-lead checkpoint:** same URL with `1_lead_ECGFounder.pth`, into
   `research_experiments/phase_2/models/pretrained/` (sha256 `f863a388…37603`).
3. **Features:**
   - `python -m research_experiments.phase_2.ecgf_features` (≈ 5 min)
   - `python -m research_experiments.phase_3.clinical_features`
   - `python -m research_experiments.phase_3.masking_audit`
4. **CV runs not on GitHub:** rerun `resume_pipeline.sh`. Runs are deterministic given seeds; the
   CNN folds are the slow part, ≈ 2 h.
