"""Masking audit, part B-D: development CV (repeat 0, and repeat 1 for the key ones) of each construction with
the SAME linear head as phase 2 (class-weighted CE, L2 on inner-val), label-free statement agreement, and
the dose-response of performance vs observed input fraction.
    python -m research_experiments.phase_3.masking_cv
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_experiments.common import ROOT, load_meta
from research_experiments.phase_3.p3common import MOD3, RES3, run3

LIN = dict(head="linear", loss="ce", weight="balanced")
P2 = ROOT / "research_experiments" / "phase_2"


def labelfree(path):
    tasks = [t.strip().upper() for t in (ROOT / "research_experiments/ecgfounder/tasks_150_labels.txt").read_text().splitlines()]
    meta = load_meta()
    h = meta[["ecg_id"]].merge(pd.read_csv(ROOT / "data/header_features.csv")[["ecg_id", "statements", "summary"]], on="ecg_id", how="left")
    st = h.statements.fillna("").str.upper().str.split("|").map(set); summ = h.summary.fillna("").str.upper().values
    z = np.load(path); lg = z["logits"]; res = []
    for j, t in enumerate(tasks):
        y = np.array([t in s for s in st]) | (summ == t)
        if 15 <= y.sum() <= len(y) - 15:
            res.append(roc_auc_score(y, lg[:, j]))
    return float(np.median(res))


def main():
    info = json.loads((RES3 / "masking_audit_features.json").read_text())
    frac = info["observed_fraction"]
    rows = []
    p2 = {"aligned": ("ecgf_aligned_f6_lin", 0.4166), "aligned_meas": ("ecgf_aligned_meas_f6_lin", 0.3125),
          "leadII12": ("ecgf_leadII12_f6_lin", 0.0833)}
    lf2 = json.loads((P2 / "results/audit_inputs_labelfree.json").read_text())
    for mode, (tag, fr) in p2.items():
        m = json.loads((P2 / "results" / tag / "cv_metrics.json").read_text())["pooled"]
        rows.append(dict(construction=mode, observed_fraction=fr, source="phase 2", labelfree_median_auroc=lf2[mode]["median_auroc"],
                         accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                         mild_f1=m["per_class"]["Mild"]["f1"], qwk=m["qwk"], s3=m["s3"]))
    for mode in ("col_only", "limb_only", "precordial_plus_II", "precordial_only", "leadII_2p5"):
        m = run3(f"mask_{mode}_f6_lin", dict(blocks=[("ecgf3", mode, "f6")]), LIN)
        rows.append(dict(construction=mode, observed_fraction=frac[mode], source="phase 3",
                         labelfree_median_auroc=labelfree(MOD3 / f"ecgf_{mode}_features.npz"),
                         accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                         mild_f1=m["per_class"]["Mild"]["f1"], qwk=m["qwk"], s3=m["s3"]))
    # valid alternatives (no fabrication)
    alts = {"windows_mean (each 2.5-s window alone, recorded leads only)": dict(blocks=[("ecgf3", "windows", "f6")]),
            "windows_concat": dict(blocks=[("ecgf3", "windows", "f6cat")]),
            "multiview: aligned 12-lead + 1-lead model on true 10-s lead II": dict(blocks=[("ecgf", "aligned", "f6"), ("ecgf", "onelead_II", "f6")]),
            "multiview + windows_mean": dict(blocks=[("ecgf", "aligned", "f6"), ("ecgf3", "windows", "f6")])}
    for name, spec in alts.items():
        tag = "mask_alt_" + {"windows_mean": "windows_mean", "windows_concat": "windows_concat", "multiview:": "multiview",
                             "multiview": "multiview_plus_windows"}[name.split(" ")[0]]
        m = run3(tag, spec, LIN)
        rows.append(dict(construction=name, observed_fraction=np.nan, source="phase 3 alternative", labelfree_median_auroc=np.nan,
                         accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                         mild_f1=m["per_class"]["Mild"]["f1"], qwk=m["qwk"], s3=m["s3"]))
    # repeat 1 for the aligned reference and the best alternative (split robustness)
    R = pd.DataFrame(rows)
    dr = R[R.observed_fraction.notna()].sort_values("observed_fraction")
    slope_s3, icpt = np.polyfit(dr.observed_fraction, dr.s3, 1)
    slope_acc, _ = np.polyfit(dr.observed_fraction, dr.accuracy, 1)
    top = dr[dr.observed_fraction >= 0.2]
    slope_top, _ = np.polyfit(top.observed_fraction, top.s3, 1) if len(top) >= 3 else (np.nan, None)
    out = dict(table=rows, dose_response=dict(
        s3_per_10pct_observed_all=float(slope_s3 / 10), accuracy_per_10pct_observed_all=float(slope_acc / 10),
        s3_per_10pct_observed_upper_range=float(slope_top / 10) if np.isfinite(slope_top) else None,
        note="linear fit over constructions that only REMOVE real signal; extrapolating beyond 42 % observed is a HYPOTHESIS"))
    R.to_csv(RES3 / "masking_dose_response.csv", index=False)
    (RES3 / "masking_dose_response.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False)); print(json.dumps(out["dose_response"], indent=1))


if __name__ == "__main__":
    main()
