# Phase 6 repository audit addendum

`phase_5/REPOSITORY_AUDIT.md` is a Phase 5 file and is not modified. This addendum records the repository state at the
start of Phase 6 and the known issues the owner has said they will handle themselves.

## 1. Branches (all on `origin` = the `chaithr45` fork; the `krithikareddy22` repository showed identical heads)
| Branch | Head | Relation |
|---|---|---|
| `main` | `9f2a883` | original project (ECG_ECHO folder deleted in this commit) |
| `claude/funny-gauss-udlmv0` | `c248730` | original project work; ancestor of all research branches; **never pushed to by Phase 6** |
| `research/ecg-ef-model-improvements` | `39aa39f` | Phases 1–3 |
| `research/phase4-handoff` | `790603f` | + Phase 4 |
| `research/phase5` | `27cbdbd` | + Phase 5 (start point of Phase 6) |
| `research/phase6` | – | this phase, created from `27cbdbd`; additive only |

All research branches form one linear history; no branch diverges.

## 2. Known issue: existing public exposure (owner decision; Phase 6 changes none of it)
The repository is **public** and has a public fork.
1. `main` and every research branch track the 1,047 ECG PDFs, whose file names and contents include patient names.
2. Git history (commit `9f2a883^`) contains the deleted `ECG_ECHO` matching sheets and notebook (names, ages, sex,
   dates, EF values).
3. Phase 1–3 per-ECG prediction CSVs keyed by ECG id are tracked, including the locked-test probability files under
   `phase_2/locked_test/` (and the Phase 1 test predictions).

**Owner decision (2026-11-05):** the owner will handle this. Phase 6 does not modify, delete or rewrite any of it,
and adds nothing similar: every Phase 6 commit contains code, aggregate metrics, aggregate plots and reports only,
and is checked by `privacy_scan` (all 1,047 ECG ids, acquisition timestamps and patient keys) and a staged-diff review.

## 3. Known issue: patient grouping by name
- 29 near-identical name pairs under different patient keys (Phase 5 audit): 15 within D, 11 straddling D and the
  locked test T, 3 within T.
- Phase 6 runs a sensitivity analysis on D with the 15 within-D pairs merged (PHASE6_PLAN.md §4.6). The 11 D/T pairs
  and the 3 T pairs are only counted here; **they must be resolved by the owner before any locked-test evaluation**.
  No ECG is moved between D and T, dropped or relabelled.

## 4. Environment differences in the Phase 6 container (recorded; reproduction verified)
- `torch==2.15.0` was installed from PyPI (CUDA build, run on CPU in FP32) because `download.pytorch.org` is not
  reachable from the container.
- `opencv-python-headless==5.0.0` does not resolve on PyPI; `5.0.0.93` (reports version 5.0.0) was installed.
- `huggingface.co` was initially blocked by the environment's network policy; after the owner changed the policy the
  ECGFounder checkpoint was downloaded and its SHA-256 matched the recorded value.
- Reproduction checks: the Phase 5 feature-validation summary is reproduced exactly; the C1 / C2 / C3 per-ECG
  predictions are checked against the committed Phase 5 aggregates (`results/baseline_regeneration.json`).

## 5. Tooling notes
- `phase_5.resume5` rewrites `PHASE5_STATUS.md` and `checkpoint.json`, and `phase_5.integrity5 verify` writes a report
  into `phase_5/results/`. Phase 6 therefore uses `phase_6.resume6` and `phase_6.integrity6`. During setup `resume5`
  was run once as instructed; its two rewritten Phase 5 files were restored with `git checkout` immediately, before
  any commit (integrity verified afterwards: 0 changed).
- Phase 5 run commands cannot regenerate missing private predictions in a fresh clone without rewriting committed
  aggregates (`save_run` refuses or overwrites). Phase 6 regenerates them with `phase_6.baselines` into its own
  private store instead.
