# Phase 5 handoff

- **Branch:** `research/phase5`, created from `research/phase4-handoff` @ `790603f`. Pushed; no force-push, and no
  existing branch was modified.
- **Read first:** `PHASE5_RESUME.md` (commands), then `reports/PHASE5_FINAL_REPORT.md` (results), `PHASE5_PLAN.md`
  (pre-registration) and `REPOSITORY_AUDIT.md`.

## Completed (all milestones DONE; `python -m research_experiments.phase_5.resume5`)
| Milestone | Outcome |
|---|---|
| M0 audit + foundation | **Audit:** the 56.8 % benchmark is four-class on the same 837 ECGs (comparable); name-key grouping risk documented. **Foundation:** checkpoint and resume architecture, integrity snapshot (5,433 protected files), safeguard tests. |
| 5A baseline | 62 existing runs re-evaluated. Calibrated MLP reproduced exactly (0.568) with stored predictions. C1 vote 0.547; best admissible baseline 0.560. |
| 5B error analysis | **Errors:** Mild confused both ways; missed Severe near EF 31. **Subgroups:** repeat-ECG patients are hard; no leakage signal. **Confidence:** the 50 % most confident ECGs reach 66–68 %. |
| 5C features | 7 new families extracted and validated. Only QRS morphology adds to the header numbers. |
| 5D E1–E7 | 43 candidates in total. Best admissible: vote + nested prior rule, **0.561**. Non-admissible variants reach up to 0.572. **No genuine improvement.** |
| 5E statistics | Corrected t-tests, Holm (family and global), cluster bootstrap, McNemar (`results/phase5_family_comparisons.csv`, `phase5_selection.json`). |
| 5F selection + report | Selected `P5D-E5-vote-prior-protected-v1`. **The 65 % target was not reached**: the CI upper bounds of every model are ≤ 0.612. |

## Remaining work (needs owner decisions or new data, not more modelling on these 837 ECGs)
1. **Owner decision:** score a pre-registered shortlist on the exposed locked test, after resolving the 11
   name-similar development / test pairs. Shortlist: selected model, C1, C2.
2. **Preferably:** an external or temporally independent cohort.
3. **Data that would change the outcome:** verified ECG–echo identifiers, more abnormal-EF ECGs, digital waveforms.
4. **Optional:** a formal selective-prediction (abstention) study, if the clinical owner wants that task.

## Privacy and integrity status at handoff
- **Commits:** every Phase 5 commit was privacy-scanned for all 1,047 ECG ids, timestamps and patient hashes: 0 hits.
- **Per-ECG files:** per-ECG predictions and features live only in `phase_5/data/private/` (git-ignored).
- **Integrity:** `integrity5 verify phase5_before` reports 0 changed / 0 missing protected files (Phases 1–4 and the
  original project).
- **Earlier risks (repository is public):** Phases 1–3 previously published per-ECG prediction CSVs, including
  locked-test probabilities. They are unchanged by Phase 5, and removing them is the owner's decision
  (`REPOSITORY_AUDIT.md` §9).
