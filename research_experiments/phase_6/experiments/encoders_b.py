"""Experiment B pipeline (PHASE6_PLAN.md section 4.2), identical for every frozen embedding:
  (i)  EF: ridge regression on header numeric || embedding (block-standardised on inner-train), alpha in
       {0.1, 1, 10, 100, 1000} chosen on inner-val MAE -> out-of-fold predicted EF (screen metric r)
  (ii) class: Phase 2 MLP fusion (header numeric || embedding; p2heads MLP 64, dropout 0.5, class-balanced CE, L2 and
       epoch on inner-val S3) -> four-class probabilities
Echo EF is a training target of (i) only. Development set, Phase 2 folds; Stage 1 = seed 42 x repeats 0-2.
  B0       ECGFounder f6 (reference)          P6B0-ecgf-f6-v1
  ECGFM    ECG-FM final-layer mean (768-d)    P6B-ecgfm-v1
    python -m research_experiments.phase_6.experiments.encoders_b --emb ECGFM --seeds 42
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import Ridge

from research_experiments.phase_2 import p2heads
from research_experiments.phase_5.experiments.nested import MLPF
from research_experiments.phase_6.p6common import (EMB6, IDS, Y, blockstd, done, ecgf_f6, ef_target, header_numeric, mark_running,
                                                   nD, save_run, split_indices)

EMBS = {"B0": ("P6B0-ecgf-f6-v1", None), "ECGFM": ("P6B-ecgfm-v1", "ecgfm.npz"),
        "F6+ECGFM": ("P6B-f6+ecgfm-v1", "f6+ecgfm")}   # Amendment 2 (exploratory)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def emb(name):
    """-> list of embedding blocks (each block-standardised separately)."""
    if EMBS[name][1] is None:
        return [ecgf_f6().astype(float)]
    if EMBS[name][1] == "f6+ecgfm":
        return emb("B0") + emb("ECGFM")
    z = np.load(EMB6 / EMBS[name][1])
    assert (z["ecg_id"] == np.asarray(IDS, dtype="U32")).all()
    return [z["E"].astype(float)]


def run(name, seed, r):
    rid = f"{EMBS[name][0]}__s{seed}_r{r}"
    if done(rid):
        return
    mark_running(rid, f"python -m research_experiments.phase_6.experiments.encoders_b --emb {name} --seeds {seed}")
    E = emb(name); ef = ef_target()["ef"]; ok = np.isfinite(ef)
    P = np.zeros((nD, 4)); efhat = np.zeros(nD); info = []
    for k in range(5):
        itr, iva, ote = split_indices(r, k)
        fm = np.zeros(nD, bool); fm[itr] = True
        X = blockstd([header_numeric(fm)] + E, fm)
        tr, va = itr[ok[itr]], iva[ok[iva]]
        best = min(((np.mean(np.abs(Ridge(alpha=a).fit(X[tr], ef[tr]).predict(X[va]) - ef[va])), a) for a in ALPHAS))
        efhat[ote] = Ridge(alpha=best[1]).fit(X[tr], ef[tr]).predict(X[ote])
        f = p2heads.fit(X[itr], Y[itr], X[iva], Y[iva], dict(MLPF, no_std=True), seed=seed)
        P[ote] = f["predict"](X[ote])[0]
        info.append(dict(fold=k, ridge_alpha=best[1], val_mae=float(best[0]), mlp_l2=f["l2"], mlp_best_epoch=f["best_epoch"]))
    cfg = dict(embedding=name, ef_model="ridge on header || embedding (alpha on inner-val MAE)", class_model="Phase 2 MLP fusion (p2heads)",
               seed=seed, repeat=r)
    agg = save_run(rid, P, P.argmax(1), r, cfg, efhat=efhat, extra=dict(folds=info, dim=int(sum(b.shape[1] for b in E))))
    m, e = agg["metrics"], agg["ef_regression"]["A"]
    print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} minR {m['min_recall']:.2f} | EF r {e['pearson']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emb", nargs="+", default=["B0", "ECGFM"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); a = ap.parse_args()
    torch.set_num_threads(1)
    for name in a.emb:
        for s in a.seeds:
            for r in (0, 1, 2):
                run(name, s, r)


if __name__ == "__main__":
    main()
