"""Label-free compatibility of each ECGFounder input construction: AUROC of the PRE-TRAINED statement
outputs against the statements the ECG machine printed on the same PDF (no EF labels used).
    python -m research_experiments.phase_2.audit_inputs_labelfree
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_experiments.common import ROOT, load_meta
from research_experiments.phase_2.p2common import MOD, RES

tasks = [t.strip().upper() for t in (ROOT / "research_experiments/ecgfounder/tasks_150_labels.txt").read_text().splitlines()]
meta = load_meta()
h = meta[["ecg_id"]].merge(pd.read_csv(ROOT / "data/header_features.csv")[["ecg_id", "statements", "summary"]], on="ecg_id", how="left")
st = h.statements.fillna("").str.upper().str.split("|").map(set); summ = h.summary.fillna("").str.upper().values
out = {}
for mode in ("aligned", "aligned_meas", "leadII12", "onelead_II", "tiled"):
    z = np.load(MOD / f"ecgf_{mode}_features.npz"); assert (z["ecg_id"] == meta.ecg_id.values).all()
    res = {}
    for j, t in enumerate(tasks):
        y = np.array([t in s for s in st]) | (summ == t)
        if 15 <= y.sum() <= len(y) - 15:
            res[t] = float(roc_auc_score(y, z["logits"][:, j]))
    out[mode] = {"n": len(res), "median_auroc": float(np.median(list(res.values()))), "per_statement": res}
    print(f"{mode:13s} n={len(res)} median AUROC {out[mode]['median_auroc']:.3f}  " +
          " ".join(f"{k[:12]}={v:.2f}" for k, v in list(res.items())[:14]))
(RES / "audit_inputs_labelfree.json").write_text(json.dumps(out, indent=1))
