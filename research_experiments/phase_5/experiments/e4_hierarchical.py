"""E4: hierarchical classification. Stage 1 Normal vs abnormal; stage 2 Mild / Moderate / Severe trained on abnormal
ECGs only. Both are the phase-2 MLP head (mlp5, header numeric || ECGFounder f6, inner-val selection); combined
P(Normal) = p1, P(c) = (1 - p1) * p2(c); evaluated as the original four-class task. Baseline: flat MLP fusion.
    python -m research_experiments.phase_5.experiments.e4_hierarchical --jobs 3 --part 0
"""
import argparse

import numpy as np
import torch

from research_experiments.phase_5.experiments import mlp5
from research_experiments.phase_5.p5common import REPEATS, SEEDS, Y, done, make_X, mark_running, nD, save_run, split_indices

CMD = "python -m research_experiments.phase_5.experiments.e4_hierarchical"


def run(s, r):
    rid = f"P5D-E4-hier-v1__s{s}_r{r}"
    if done(rid):
        return
    mark_running(rid, CMD)
    P = np.zeros((nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = split_indices(r, k); fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(["hdr", "ecgf"], fm)
        y1 = (Y > 0).astype(int)
        f1, i1 = mlp5.fit(X[itr], y1[itr], X[iva], y1[iva], dict(n_classes=2, no_std=pre), seed=s)
        a_tr, a_va = itr[Y[itr] > 0], iva[Y[iva] > 0]
        f2, i2 = mlp5.fit(X[a_tr], Y[a_tr] - 1, X[a_va], Y[a_va] - 1, dict(n_classes=3, no_std=pre), seed=s)
        p1, p2 = f1(X[ote]), f2(X[ote])
        P[ote, 0] = p1[:, 0]; P[ote, 1:] = p1[:, 1:2] * p2
        info.append(dict(fold=k, stage1=i1, stage2=i2))
    agg = save_run(rid, P, P.argmax(1), r, dict(kind="hierarchical", blocks=["hdr", "ecgf"], seed=s, baseline="P5A-import-mlpfusion-v1"),
                   extra=dict(folds=info))
    m = agg["metrics"]; print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} minR {m['min_recall']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    for s, r in [(s, r) for r in REPEATS for s in SEEDS][a.part::a.jobs]:
        run(s, r)


if __name__ == "__main__":
    main()
