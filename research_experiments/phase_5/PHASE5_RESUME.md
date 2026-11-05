# Phase 5: how to resume (for another Claude session or a person)

Phase 5 never needs to be restarted from scratch. Completed runs are skipped automatically, and interrupted runs are
detected and resumed.

## 1. Set up (fresh clone or container only)
```bash
git fetch origin research/phase5 && git checkout research/phase5
```
```bash
pip install -r research_experiments/requirements-research.txt -r research_experiments/phase_5/requirements-phase5.txt
```
```bash
python -m research_experiments.regenerate_data_cache
```

Then recover the private EF tables, which are needed for the error analysis only:
- follow `research_experiments/DATA_ACCESS.md`;
- run `python -m research_experiments.phase_4.audit_matching`, then `python -m research_experiments.phase_4.cohorts`.

## 2. See where Phase 5 stands
```bash
python -m research_experiments.phase_5.resume5
```

This prints every milestone (NOT STARTED, IN PROGRESS, DONE, BLOCKED, INTERRUPTED), writes `checkpoint.json` and
`PHASE5_STATUS.md`, and prints the **exact next command**.

| State | Meaning | What to do |
|---|---|---|
| `MISSING_PRIVATE` | The committed aggregate exists but the git-ignored per-ECG predictions do not (normal in a fresh clone). | Run the printed command; it regenerates them deterministically and never changes the recorded result. |
| `INTERRUPTED` | A run died. | Re-run the printed command; completed runs are skipped. |
| `CORRUPT` | Prediction hash ≠ aggregate. | Investigate; never overwrite. |

## 3. Continue
```bash
python -m research_experiments.phase_5.resume5 --run-next
```

Or run the printed command yourself. Repeat until every milestone is DONE.

## 4. Before every commit and push
```bash
python -m research_experiments.phase_5.tests.test_phase5
```
```bash
python -m research_experiments.phase_5.integrity5 verify phase5_before
```

The tests must say ALL PASS, and the integrity check must report 0 changed and 0 missing protected files. Only then
commit and push the Phase 5 files to `research/phase5`, never force.

## Rules (summary; full list in `PHASE5_PLAN.md` and `REPOSITORY_AUDIT.md`)
- **Locked test:** never predict, score or inspect the 210-ECG locked test.
- **Imports:** never import `phase_2.cv_features`, `phase_2.accuracy_tradeoff`, `phase_3.p3common`, `phase_3.p3_robust`
  or `phase_4.p4common` from Phase 5 code; they load the locked test.
- **Earlier phases:** do not modify Phase 1–4 files.
- **Privacy:** per-ECG outputs go to `phase_5/data/private/` only.
- **Corrections:** never overwrite a completed run; add `-v2`.
