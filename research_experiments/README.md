# research_experiments/ — controlled improvement experiments (isolated)

Everything for the experimental phase lives here. It **only reads** the original project:
`LV_ECG_Dataset_*`, `data/`, `results/` and `models/`. It never writes to them. Every writer
goes through `common._guard`, which refuses any path outside this folder. Integrity is checked
by SHA-256 before and after (`integrity.py`).

The final report is `reports/FINAL_REPORT.md`.

## Layout

| Path | Content |
|---|---|
| `configs/` | one JSON per deep-training run: all arguments + software environment |
| `models/` | trained checkpoints and frozen ECGFounder feature files (git-ignored, large); `models/pretrained/` holds the downloaded ECGFounder checkpoint (git-ignored) |
| `results/<run>/` | `metrics.json` (val + test, 95 % patient-cluster bootstrap CIs, OvR AUROC, error severity), `val_predictions.csv`, `test_predictions.csv`, `history.csv` |
| `results/*.csv/json/md` | comparison tables, paired tests, ablations, cross-validation, integrity checks, timing |
| `plots/` | confusion matrices, training curves, summary figures |
| `logs/` | stdout of every run |
| `reports/` | final report |

## Reproduce (CPU, from the repository root)

```bash
python -m research_experiments.integrity snapshot before
python -m research_experiments.phase0_reproduce              # baseline from saved checkpoints
# ECGFounder checkpoint (MIT), verified by SHA-256 in ecgfounder/model.py
curl -L -o research_experiments/models/pretrained/12_lead_ECGFounder.pth \
  https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
python -m research_experiments.ecgfounder.extract_features   # A1 features (3 input constructions)
python -m research_experiments.ecgfounder.sanity_check       # label-free domain check
python -m research_experiments.exp_feature_heads             # A1 + B heads on fixed features
python -m research_experiments.header_ablation
bash   research_experiments/run_queue.sh                     # A2 fine-tuning, ordinal CNN, baseline retrain
python -m research_experiments.collect_repro
python -m research_experiments.exp_fusion
python -m research_experiments.cv_grouped                    # patient-grouped 5-fold CV (cheap models)
python -m research_experiments.analysis_mild_errors
python -m research_experiments.timing
python -m research_experiments.make_tables
python -m research_experiments.regression_feasibility       # Experiment C data check
python -m research_experiments.integrity verify before
```

Seeds: 42 everywhere (A2: 42, 43, 44). Software versions are in `configs/*.json` and `results/phase0_reproduction.json`.
