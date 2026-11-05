# Equal number of ECGs per category vs class weights (test n = 210)

Mean ± SD over runs (random draws for `undersample`; random seeds for the 1-D CNN).

| Model | Class handling | Training ECGs | Runs | Accuracy | Balanced acc. | Macro-F1 | HF AUC | Recall N / Mi / Mo / S |
|---|---|---:|---:|---|---|---|---|---|
| Header model | none | 669 | 1 | 0.543 | 0.418 | 0.420 | 0.784 | 0.85 / 0.17 / 0.43 / 0.22 |
| Header model | weights | 669 | 1 | 0.514 | 0.442 | 0.421 | 0.795 | 0.75 / 0.13 / 0.45 / 0.43 |
| Header model | undersample | 292 | 20 | 0.457 ± 0.025 | 0.415 ± 0.033 | 0.393 ± 0.032 | 0.776 ± 0.018 | 0.61 / 0.19 / 0.40 / 0.46 |
| Header model | oversample | 1228 | 1 | 0.476 | 0.401 | 0.387 | 0.797 | 0.70 / 0.17 / 0.39 / 0.35 |
