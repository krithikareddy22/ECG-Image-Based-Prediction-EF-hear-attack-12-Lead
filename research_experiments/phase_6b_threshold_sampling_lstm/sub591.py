"""Exploratory SUB591 (AMENDMENT_SUB591.md): random class-capped subset (N 200, Mi 150, Mo 150, S all), train only on it.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.sub591
"""
import json

import numpy as np
import pandas as pd

from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

TARGET = {0: 200, 1: 150, 2: 150, 3: 10 ** 6}


def subset():
    rng = np.random.default_rng(591); keep = np.zeros(B.nD, bool); cnt = {c: 0 for c in TARGET}
    for p in rng.permutation(pd.unique(B.PAT)):
        rows = np.flatnonzero(B.PAT == p); cls = B.Y[rows]
        add = {c: int((cls == c).sum()) for c in TARGET}
        if all(cnt[c] + add[c] <= TARGET[c] for c in TARGET):
            keep[rows] = True
            for c in TARGET:
                cnt[c] += add[c]
    return keep


SUB = subset()
_orig = B.train_exclude


def excl(cfg, r, k):
    return ~SUB if cfg == "SUB591" else _orig(cfg, r, k)


def main():
    B.train_exclude = excl; E.train_exclude = excl
    print("subset counts N/Mi/Mo/S:", [int((B.Y[SUB] == c).sum()) for c in range(4)], "total", int(SUB.sum()), flush=True)
    E.run_config("SUB591")
    out = dict(subset_counts=[int((B.Y[SUB] == c).sum()) for c in range(4)], models={})
    for m, seeds, rule in (("mlpfusion", B.SEEDS, False), ("vote", (0,), False), ("c3", (0,), True)):
        ref = {"mlpfusion": "P6A-mlpfusion-B1-v1__s{s}_r{r}", "vote": "P6A-vote-B1-v1__s0_r{r}", "c3": "P6A-c3-B1-v1__s0_r{r}"}[m]
        res = {}
        for tag, ids in (("SUB591", [f"P6B-{m}-SUB591-v1__s{s}_r{{r}}" for s in seeds]), ("ORIG", [ref.format(s=s, r="{r}") for s in seeds])):
            for scope, mask in (("all_837", np.ones(B.nD, bool)), ("subset_591", SUB)):
                ms = []
                for r in (0, 1, 2):
                    Ps = [B.load_run(i.format(r=r)) for i in ids]
                    P = np.mean([p[0] for p in Ps], 0); pred = Ps[0][1] if rule else P.argmax(1)
                    ms.append(B.metrics(B.Y[mask], P[mask], pred[mask]))
                res[f"{tag}_on_{scope}"] = {k: float(np.mean([x[k] for x in ms])) for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                                                                         "recall_Normal", "recall_Mild", "recall_Moderate", "recall_Severe", "min_recall")}
        out["models"][m] = res
    (B.guard(B.RES) / "sub591_exploratory.json").write_text(json.dumps(out, indent=1))
    for m, v in out["models"].items():
        for k, x in v.items():
            print(f"{m:10s} {k:22s}", {a: round(b, 3) for a, b in x.items()})


if __name__ == "__main__":
    main()
