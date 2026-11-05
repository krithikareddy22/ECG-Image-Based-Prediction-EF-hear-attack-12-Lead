"""Phase 7 E5-lc (PHASE7_PLAN.md Amendment 1): learning curve of the reference member and the R1 ridge EF regressor
over 25 / 50 / 75 / 100 % of each fold's inner-train patients (nested subsamples). Evidence for the data-needs section.
    python -m research_experiments.phase_7.e5_curve
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

from research_experiments.phase_2 import p2heads
from research_experiments.phase_5.experiments.nested import MLPF
from research_experiments.phase_5.p5common import blockstd, ecgf_f6, feature_family, header_numeric, impute
from research_experiments.phase_7 import p7common as C

FRACS = (0.25, 0.5, 0.75, 1.0)


def subsample(itr, r, k, frac):
    pats = pd.unique(C.PAT[itr]); order = np.random.default_rng(7000 + 10 * r + k).permutation(pats)
    keep = set(order[: max(1, int(round(frac * len(pats))))])
    return itr[np.isin(C.PAT[itr], list(keep))]


def job(r):
    torch.set_num_threads(1)
    ef = C.B.table().ef.values; E = ecgf_f6().astype(float)
    P = {f: np.zeros((C.nD, 4)) for f in FRACS}; EF = {f: np.zeros(C.nD) for f in FRACS}; n = {f: [] for f in FRACS}
    for k in range(5):
        itr0, iva, ote = C.B.split_indices(r, k)
        for f in FRACS:
            itr = subsample(itr0, r, k, f); n[f].append(int(len(itr)))
            fm = np.zeros(C.nD, bool); fm[itr] = True
            X = blockstd([header_numeric(fm), E], fm)
            P[f][ote] = p2heads.fit(X[itr], C.Y[itr], X[iva], C.Y[iva], dict(MLPF, no_std=True), seed=42)["predict"](X[ote])[0]
            fr = itr; fm2 = np.zeros(C.nD, bool); fm2[fr] = True             # same (subsampled) rows as the member
            X2 = blockstd([header_numeric(fm2), E, impute(feature_family("clin3")[0], fm2)], fm2)
            EF[f][ote] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X2[fr], ef[fr]).predict(X2[ote])
    d = C.guard7(C.PRIV7 / "e5_curve"); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"r{r}.npz", **{f"P{int(f * 100)}": P[f] for f in FRACS}, **{f"EF{int(f * 100)}": EF[f] for f in FRACS})
    return r, {f: n[f] for f in FRACS}


def main():
    with Pool(3) as pool:
        ns = dict(pool.map(job, (0, 1, 2)))
    ver, ef = C.verified(); rows = []
    for f in FRACS:
        acc, mf1, mild, sd_all, sd_ver = [], [], [], [], []
        for r in (0, 1, 2):
            z = np.load(C.PRIV7 / "e5_curve" / f"r{r}.npz"); P = z[f"P{int(f * 100)}"]; e = z[f"EF{int(f * 100)}"]
            m = C.B.metrics(C.Y, P, P.argmax(1)); acc.append(m["accuracy"]); mf1.append(m["macro_f1"]); mild.append(m["f1_Mild"])
            sd_all.append(float(np.std(e - ef))); sd_ver.append(float(np.std((e - ef)[ver])))
        rows.append(dict(fraction=f, mean_inner_train_ecgs=float(np.mean([np.mean(ns[r][f]) for r in (0, 1, 2)])),
                         accuracy=float(np.mean(acc)), macro_f1=float(np.mean(mf1)), f1_Mild=float(np.mean(mild)),
                         ef_sd_all=float(np.mean(sd_all)), ef_sd_verified=float(np.mean(sd_ver))))
    D = pd.DataFrame(rows); lx = np.log2(D.mean_inner_train_ecgs.values)
    slopes = {c: float(np.polyfit(lx, D[c].values, 1)[0]) for c in ("accuracy", "macro_f1", "f1_Mild", "ef_sd_all", "ef_sd_verified")}
    n100 = D.mean_inner_train_ecgs.values[-1]

    def need(col, target):
        s = slopes[col]; cur = D[col].values[-1]
        if (target - cur) * s <= 0:
            return None                                   # the trend does not move towards the target
        return float(n100 * 2 ** ((target - cur) / s))
    out = dict(curve=rows, gain_per_doubling=slopes,
               indicative_training_ecgs_needed=dict(accuracy_0p70=need("accuracy", 0.70), ef_sd_9=need("ef_sd_all", 9.0),
                                                    f1_Mild_0p55=need("f1_Mild", 0.55)),
               caveat="log-linear extrapolation from 4 points with the current label quality; indicative only")
    (C.guard7(C.RES7) / "e5_learning_curve.json").write_text(json.dumps(out, indent=1))
    D.to_csv(C.guard7(C.RES7 / "e5_learning_curve.csv"), index=False, float_format="%.4f")
    print(D.to_string(index=False)); print(json.dumps({k: out[k] for k in ("gain_per_doubling", "indicative_training_ecgs_needed")}, indent=1))


if __name__ == "__main__":
    main()
