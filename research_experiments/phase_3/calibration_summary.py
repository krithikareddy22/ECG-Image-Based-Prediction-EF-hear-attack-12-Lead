"""Aggregate the nested calibration results (3 CV repeats) into original vs calibrated tables + confusion matrices.
    python -m research_experiments.phase_3.calibration_summary"""
import json
import numpy as np, pandas as pd
from research_experiments.phase_3.p3common import P2RES, RES3
KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "f1_mild", "recall_mild", "qwk", "recall_Normal", "share_pred_normal"]
rows, cms = [], {}
for model in ("fusion_mlp", "ecgf_linear", "header_numeric_lr"):
    for r in (0, 1, 2):
        j = json.loads((P2RES / f"nested_calibration_{model}_r{r}.json").read_text())
        for rule, v in j.items():
            if isinstance(v, dict) and "accuracy" in v:
                rows.append(dict(model=model, repeat=r, rule=rule, **{k: v[k] for k in KEYS}))
                cms[(model, rule)] = cms.get((model, rule), 0) + np.array(v["confusion"])
R = pd.DataFrame(rows)
base = R[R.rule == "original"].set_index(["model", "repeat"])[KEYS]
for k in KEYS:
    R["d_" + k] = R[k].values - base.loc[list(zip(R.model, R.repeat)), k].values
agg = R.groupby(["model", "rule"])[KEYS + ["d_" + k for k in KEYS[:6]]].agg(["mean", "std"])
agg.to_csv(RES3 / "calibration_nested_summary.csv")
lines = ["| Model | Rule | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | Mild recall | QWK | Normal recall | Predicted Normal |", "|---|---|---|---|---|---|---|---|---|---|"]
for (m, rule), g in R.groupby(["model", "rule"], sort=False):
    f = lambda k: f"{g[k].mean():.3f}" + (f" ({g['d_' + k].mean():+.3f})" if rule != "original" and k in KEYS[:6] else "")
    lines.append(f"| {m} | {rule} | {f('accuracy')} | {f('balanced_accuracy')} | {f('macro_f1')} | {f('f1_mild')} | {f('recall_mild')} | {f('qwk')} | {g.recall_Normal.mean():.3f} | {g.share_pred_normal.mean():.1%} |")
cm_md = ["", "Confusion matrices summed over the 3 CV repeats (rows = true N/Mi/Mo/S; 3 x 837 ECGs)"]
for (m, rule), cm in cms.items():
    if m == "fusion_mlp":
        cm_md.append(f"\n**{m} - {rule}**\n```\n" + "\n".join(" ".join(f"{x:5d}" for x in row) for row in cm) + "\n```")
(RES3 / "calibration_nested_tables.md").write_text("\n".join(lines + cm_md) + "\n")
print("\n".join(lines + cm_md))
