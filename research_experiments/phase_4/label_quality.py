"""Phase 4F label-quality training + benchmark import (pre-registered).

Label-quality variants of the phase-2 MLP fusion (header numeric || ECGFounder f6; p2heads MLPF config, seeds 42/43/44,
repeats 0-2). Only TRAINING and inner-val SELECTION change; evaluation is always on every outer-fold ECG of D:
  lq_trainG        train + select on high-confidence cohort G only
  lq_trainC        train + select on cohort C (nearest echo <= 14 d) only
  lq_dropdisagree  train + select without ECGs whose folder label != class of the nearest-echo EF
  lq_relabel       train + select on the class of the nearest-echo EF instead of the folder label
Benchmarks (read-only phase-2/3 OOF predictions) are re-evaluated with the identical phase-4 evaluation sets:
  bench_vote_r{r}            vote ensemble (phase 3)
  bench_mlpfusion_s{s}_r{r}  2-way MLP fusion (phase 2)
  bench_ecgflin_dist_r{r}, bench_hdrclin_lr_r{r}   the two other vote members

    python -m research_experiments.phase_4.label_quality --bench
    python -m research_experiments.phase_4.label_quality --jobs 2 --part 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.cv_features import MLPF
from research_experiments.phase_3.p3_robust import find
from research_experiments.phase_4.p4common import (COHORTS, PCOLS_, REPEATS, SEEDS, Y, YD, X_of, done, nD, save_run,
                                                     short, split_indices, D)

VARIANTS = ("lq_trainG", "lq_trainC", "lq_dropdisagree", "lq_relabel")


def run(variant, repeat, seed):
    tag = f"{variant}_s{seed}_r{repeat}"
    if done(tag):
        return
    t0 = time.time()
    keep = {"lq_trainG": COHORTS["G"], "lq_trainC": COHORTS["C"], "lq_dropdisagree": Y == YD,
            "lq_relabel": np.ones(nD, bool)}[variant]
    ytrain = YD if variant == "lq_relabel" else Y
    P = np.zeros((nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = split_indices(D, repeat, k)
        itr, iva = itr[keep[itr]], iva[keep[iva]]
        fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = X_of("hdr+ecgf", fm)
        f = p2heads.fit(X[itr], ytrain[itr], X[iva], ytrain[iva], dict(MLPF, no_std=pre), seed=seed)
        P[ote] = f["predict"](X[ote])[0]
        info.append(dict(fold=k, n_train=int(len(itr)), n_val=int(len(iva)),
                         train_class_counts=np.bincount(ytrain[itr], minlength=4).tolist(), l2=f["l2"],
                         val_s3=f["val_s3"], best_epoch=f["best_epoch"]))
    agg = save_run(tag, P, P.argmax(1), repeat, dict(family="4F_label_quality", variant=variant, seed=seed, folds=info,
                                                   minutes=(time.time() - t0) / 60))
    print(f"[{tag}] {short(agg['eval'])} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _oof(t):
    d = find(t)
    return pd.read_csv(d / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS_].values


def bench():
    sfx = lambda r: "" if r == 0 else f"_r{r}"
    for r in REPEATS:
        for s in SEEDS:
            t = ("ff_hdrnum+ecgf_mlp" if s == 42 else f"ff_hdrnum+ecgf_mlp_seed{s}") + sfx(r)
            P = _oof(t); save_run(f"bench_mlpfusion_s{s}_r{r}", P, P.argmax(1), r, dict(family="benchmark", source=t, seed=s))
        for name, t in (("bench_vote", "ens_mlp2+ecgflin+hdrclin__vote"), ("bench_ecgflin_dist", "ecgf_f6_lin_cedist1"),
                        ("bench_hdrclin_lr", "clin_hdrnum+clin_skLR")):
            P = _oof(t + sfx(r))
            agg = save_run(f"{name}_r{r}", P, P.argmax(1), r, dict(family="benchmark", source=t + sfx(r), seed=-1))
            print(f"[{name}_r{r}] {short(agg['eval'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(1)
    if a.bench:
        return bench()
    J = [(v, r, s) for v in VARIANTS for r in REPEATS for s in SEEDS][a.part::a.jobs]
    for j in (J[::-1] if a.part % 2 else J):      # odd workers run backwards (two workers on one list meet in the middle)
        run(*j)


if __name__ == "__main__":
    main()
