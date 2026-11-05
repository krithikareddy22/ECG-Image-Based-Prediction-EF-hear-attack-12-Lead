"""5E statistics helpers: paired comparison of two experiments on identical folds.
  corrected repeated-CV t-test on matched (repeat, fold) cells (primary), sign consistency over repeats,
  patient-cluster paired bootstrap + exact McNemar on repeat 0 (secondary), Holm over a planned family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research_experiments.phase_5.baseline.report import combined, groups
from research_experiments.phase_5.p5common import CLASSES, PAT, RECALL_FLOOR, Y, cells, corr_ci, corr_t, holm, metrics, paired_bootstrap

CMP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "f1_Mild")


def model_summary(exp_id, G=None, cache=None):
    G = G or groups()
    rr = {r: (cache or {}).get((exp_id, r)) or combined(G[exp_id][r]) for r in sorted(G[exp_id])}
    ms = [metrics(Y, *rr[r][:2]) for r in rr]
    s = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    s["min_class_recall_mean"] = min(s[f"recall_{c}"] for c in CLASSES); s["admissible"] = s["min_class_recall_mean"] >= RECALL_FLOOR
    s["repeats"] = len(rr); s["runs"] = sum(len(G[exp_id][r]) for r in rr)
    return s, rr


def compare(cand, ref, G=None, boot=True):
    G = G or groups()
    _, a = model_summary(ref, G); _, b = model_summary(cand, G)
    common = sorted(set(a) & set(b))
    out = {}
    for k in CMP_METRICS:
        dl = [y[k] - x[k] for r in common for x, y in zip(cells(*a[r][:2], r), cells(*b[r][:2], r))]
        d, p = corr_t(dl)
        per = [metrics(Y, *b[r][:2])[k] - metrics(Y, *a[r][:2])[k] for r in common]
        out[k] = dict(diff=d, p=p, ci95=corr_ci(dl), per_repeat=per,
                      same_sign=bool(len(per) == 3 and d != 0 and np.all(np.sign(per) == np.sign(d))))
    if boot and 0 in common:
        pb = paired_bootstrap(Y, a[0][0], b[0][0], PAT, a[0][1], b[0][1])
        out["boot_r0_accuracy_ci95"] = pb["accuracy"]["ci95"]; out["mcnemar_r0"] = pb["mcnemar_accuracy"]
    return out


def family_table(cands, ref, G=None, boot=False):
    """Rows: candidate summaries + comparison vs ref, Holm within metric over the candidates."""
    G = G or groups()
    rows = []
    for c in cands:
        s, _ = model_summary(c, G); cp = compare(c, ref, G, boot)
        row = dict(model=c, reference=ref, repeats=s["repeats"], runs=s["runs"],
                   **{k: s[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "recall_Severe", "precision_Severe",
                                        "f1_Mild", "min_class_recall_mean", "admissible")})
        for k in CMP_METRICS:
            row[f"d_{k}"], row[f"p_{k}"], row[f"same_sign_{k}"] = cp[k]["diff"], cp[k]["p"], cp[k]["same_sign"]
            row[f"ci_{k}"] = [round(x, 4) for x in cp[k]["ci95"]]
        if boot and "boot_r0_accuracy_ci95" in cp:
            row["boot_r0_d_accuracy_ci95"] = [round(x, 4) for x in cp["boot_r0_accuracy_ci95"]]
            row["mcnemar_r0_p"] = cp["mcnemar_r0"]["p_exact"]
        rows.append(row)
    T = pd.DataFrame(rows)
    if len(T):
        for k in CMP_METRICS:
            T[f"p_holm_{k}"] = holm(T[f"p_{k}"].values)
        T["genuine_accuracy_gain"] = (T.d_accuracy > 0) & (T.p_holm_accuracy < 0.05) & T.same_sign_accuracy & T.admissible
    return T.sort_values("accuracy", ascending=False)
