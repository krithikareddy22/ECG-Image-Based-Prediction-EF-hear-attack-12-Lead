# Phase 6: how to resume (for another Claude session or a person)

Phase 6 never needs to be restarted. Completed runs are skipped automatically; interrupted runs are detected.

## 1. Set up (fresh clone or container only)
```bash
git fetch origin research/phase6 && git checkout research/phase6
```
```bash
pip install -r requirements.txt -r research_experiments/requirements-research.txt -r research_experiments/phase_5/requirements-phase5.txt
```
- If `opencv-python-headless==5.0.0` does not resolve, install `5.0.0.93`. If `download.pytorch.org` is blocked,
  `torch==2.15.0` from PyPI works (FP32 on CPU).

```bash
python -m research_experiments.regenerate_data_cache
```

Recover the private EF tables exactly as `research_experiments/DATA_ACCESS.md` describes (git-ignored folder only),
then:
```bash
python -m research_experiments.phase_4.audit_matching
```
```bash
python -m research_experiments.phase_4.cohorts
```

ECGFounder weights (needs `huggingface.co` and `*.hf.co` in the environment's allowed domains; SHA-256
`ee199f37…7997` is verified on load):
```bash
curl -L -o research_experiments/models/pretrained/12_lead_ECGFounder.pth https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
```

Git-ignored inputs and baselines (development ECGs only; both verify themselves against committed aggregates):
```bash
python -m research_experiments.phase_6.features.restore_inputs --mirror-phase5
```
```bash
python -m research_experiments.phase_6.baselines
```

Integrity snapshot (local; once per container, before any Phase 6 run):
```bash
python -m research_experiments.phase_6.integrity6 snapshot phase6_before
```

## 2. See where Phase 6 stands
```bash
python -m research_experiments.phase_6.resume6
```

Prints every milestone and run state, writes `phase_6/checkpoint.json` and the status table in `PHASE6_STATUS.md`,
and prints the exact next command (`--run-next` executes it).

| State | Meaning | What to do |
|---|---|---|
| `MISSING_PRIVATE` | committed aggregate, git-ignored per-ECG file absent (fresh clone) | re-run the printed command; it regenerates deterministically |
| `INTERRUPTED` | a run died | re-run the printed command; completed runs are skipped |
| `CORRUPT` | prediction hash ≠ aggregate | investigate; never overwrite |

## 3. Before every commit and push
```bash
python -m research_experiments.phase_6.tests.test_phase6
```
```bash
python -m research_experiments.phase_5.tests.test_phase5
```
```bash
python -m research_experiments.phase_6.integrity6 verify phase6_before
```

All must pass (integrity: 0 changed, 0 missing, git check ok). Review the staged diff; then commit and push to
`research/phase6` only, never force. Never push to `claude/funny-gauss-udlmv0`.

## Fresh clone: private files of committed runs (MISSING_PRIVATE)
Re-running a committed run regenerates its git-ignored `oof.csv`; the committed aggregate is never modified. Retrained
models differ in the last digits across CPUs / GPUs, so the file hash may differ: the file is accepted only if accuracy,
balanced accuracy and macro-F1 are within 0.005 of the aggregate (else the script stops), and a record (hashes, metric
differences, platform; no per-ECG data) is written to `results/regeneration/<run_id>.json` (phase_6 and phase_6a).

## Windows environment notes (owner's GPU laptop)
Clone outside OneDrive (`C:\krithika_major\ecg-lvef`) with `git clone -c core.autocrlf=input ...` so tracked files keep
LF line endings (Git for Windows defaults to CRLF, which changes file hashes). Three helpers live **outside the
repository** and are never committed; they change no computation, only how text is written / how Python starts:
- `C:\krithika_major\venv\Lib\site-packages\sitecustomize.py` (venv-local): pandas `to_csv` and `Path.write_text` write
  `\n` instead of the Windows `\r\n`, so `oof.csv` / JSON files are byte-identical to the Linux reference.
- `C:\krithika_major\tools\win_manifest.py`: builds `data/manifest.csv` as on Linux (`/` in the `source` column) and
  writes it only if its SHA-256 equals the recorded value; run it before `regenerate_data_cache`.
- `C:\krithika_major\p6.ps1`: runs the venv Python from the repository root with `PYTHONUTF8=1` (UTF-8 for
  `read_text`, e.g. the plan-hash tests) and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic cuBLAS on CUDA), e.g.
  `& C:\krithika_major\p6.ps1 -m research_experiments.phase_6.baselines`.
- Phase 4 scripts rewrite their committed plots (PNG rendering differs on Windows): restore them with
  `git checkout -- research_experiments/phase_4/plots/` (the CSV / JSON outputs are byte-identical).

## Rules (summary; full list in `PHASE6_PLAN.md`)
- **Locked test:** never load, predict, score or inspect it; never run a test PDF through any model.
- **Imports:** never import `phase_2.cv_features`, `phase_2.accuracy_tradeoff`, `phase_3.p3common`,
  `phase_3.p3_robust`, `phase_4.p4common` or `phase_3.clinical_features` from Phase 6 code.
- **Do not run** `phase_5.resume5` (rewrites Phase 5 files) or `phase_5.integrity5 verify` (writes into
  `phase_5/results/`).
- **Earlier phases:** never modify Phase 1–5 files.
- **Privacy:** per-ECG outputs go to `phase_6/data/private/` only.
- **Corrections:** never overwrite a completed run; add `-v2`.
