"""Domain-compatibility check of each input construction (uses no EF labels at all):
AUROC of ECGFounder's PRE-TRAINED statement outputs against the statements the ECG machine
printed on the same PDF (header text), for statements present in both vocabularies with >= 15
positives among all 1,047 ECGs. Higher = the digitised input is closer to what ECGFounder saw
in pre-training.   python -m research_experiments.ecgfounder.sanity_check
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_experiments.common import MOD, RES, ROOT, load_meta

tasks = [t.strip().upper() for t in (Path(__file__).parent / "tasks_150_labels.txt").read_text().splitlines()]
meta = load_meta()
h = pd.read_csv(ROOT / "data" / "header_features.csv")[["ecg_id", "statements", "summary"]]
h = meta[["ecg_id"]].merge(h, on="ecg_id", how="left")
st = h.statements.fillna("").str.upper().str.split("|").map(set)
summ = h.summary.fillna("").str.upper()
out = {}
for mode in ("aligned", "tiled", "short"):
    z = np.load(MOD / f"features_ecgfounder_{mode}.npz")
    assert (z["ecg_id"] == meta.ecg_id.values).all()
    lg = z["logits"]
    res = {}
    for j, t in enumerate(tasks):
        y = np.array([t in s for s in st]) | (summ.values == t)
        if 15 <= y.sum() <= len(y) - 15:
            res[t] = (float(roc_auc_score(y, lg[:, j])), int(y.sum()))
    out[mode] = {"n_statements": len(res), "median_auroc": float(np.median([v[0] for v in res.values()])),
                 "per_statement": res}
for t in out["aligned"]["per_statement"]:
    print(f"{t[:45]:45s} n={out['aligned']['per_statement'][t][1]:4d}  " +
          "  ".join(f"{m}={out[m]['per_statement'].get(t, (np.nan,))[0]:.3f}" for m in out))
print({m: (out[m]["n_statements"], round(out[m]["median_auroc"], 3)) for m in out})
(RES / "ecgfounder_input_checks.json").write_text(json.dumps(out, indent=2))
