# DATA_ACCESS: how to recover the private EF / matching data (procedure only; no clinical content)

**This file contains no patient information, no clinical values and no file contents.** It explains how an
authorised person or agent re-creates the git-ignored private files used by Phase 4.

## Authorisation (recorded in the conversation with the project owner)
- Phase 4 authorised inspection of the four `LV_ECG_Dataset_*/*.xlsx` sheets and experimental use of verified
  continuous EF values.
- The owner then explicitly authorised **read-only** use of the `ECG_ECHO` matching files that commit `9f2a883`
  deleted from `main`:
  - extracted from git history;
  - stored only in a git-ignored folder;
  - never committed;
  - no names printed.

## Sensitivity
| Data | Contains | Rule |
|---|---|---|
| `LV_ECG_Dataset_*/**/*.pdf` (tracked by the original project) | patient names in file names and PDF text | keep the repository **private**; never print names |
| `LV_ECG_Dataset_*/*.xlsx` | file-copy logs (category, file name with patient name, paths) | read-only; never commit derived copies |
| `ECG_ECHO/*.xlsx` (git history only) | patient names, ages, sex, ECG / echo dates, EF values | **never commit**; extract only into `phase_4/data/private/` |
| `ECG_ECHO/ecg_echo_matching.ipynb` (git history only) | code **and outputs with patient-level data** | never commit; read **code cells only** |
| `phase_4/data/private/*.csv` | per-ECG EF, intervals, labels (keyed by `ecg_id` = acquisition timestamp) | git-ignored; never commit |

## Recovery procedure (run from the repository root, on the research branch)

```bash
P=research_experiments/phase_4/data/private/ecg_echo
mkdir -p "$P"
git check-ignore -q "$P" && echo "ignored: OK"      # must print OK before continuing
for f in 2020_echo_ecg_matching_50percent_20261029_162620.xlsx 2021_echo_first_ecg_matching_20261029_161227.xlsx \
         2022_ECG_ECHO_MATCHING_70_PERCENT_20261029_170108.xlsx 2024_ECG_ECHO_MATCHING_70_PERCENT_20261029_165914.xlsx \
         ecg_echo_matching.ipynb; do
  git show "9f2a883^:ECG_ECHO/$f" > "$P/$f"
done
python -m research_experiments.phase_4.audit_matching   # -> private/ecg_echo_links.csv, ecg_label_versions.csv + results/audit_matching.json (aggregates)
python -m research_experiments.phase_4.cohorts          # -> private/cohorts.csv + results/cohort_summary.csv + plots
```

- **Fork caveat:** git history (including `9f2a883`) is preserved by a GitHub fork or `git clone --mirror`. If the
  history is not available, ask the owner for the files and place them in the same folder.
- Both scripts are deterministic and take under 1 minute.

## Rules every script follows
- Names are never read into outputs. Linking to the dataset uses the **14-digit acquisition timestamp** in the
  ECG file name only.
- Committed outputs contain **aggregates only**: no `ecg_id`, no dates, no per-patient EF.
- Pseudonymous patient grouping uses the existing `data/manifest.csv:patient` hash.
- The locked 210-ECG test set is never used for any choice.
