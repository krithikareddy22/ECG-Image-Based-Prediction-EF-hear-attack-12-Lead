"""Phase 6B: the existing best model recipe (MLP fusion 3 seeds, ECGFounder linear distance-CE, header + clinical LR,
equal-weight vote = C1 recipe, + nested class-prior rule, protected = C3 recipe) under the NEW 6B configurations
(T55, N300, N270, N240, N200). ORIG / T50 / XUNC / A270X are re-used from Phase 6A (B1 / B2 / B3 / A270), not re-run.
Training labels (T55) and excluded training rows (N-sizes) apply to every fit incl. the nested inner models and the
rule fitting; evaluation always covers all 837 ECGs.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.existing6b --configs T55 N300 N270 N240 N200 --jobs 3
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES
from research_experiments.phase_6a_normal_mild_lstm.existing import MEMBERS, fit_member
from research_experiments.phase_6b_threshold_sampling_lstm.p6bcommon import (
    INNER, PAT, SEEDS, Y, decide, done, fold_vec, guard, load_run, make_X, mark_running, metrics, nD, save_run, split, table,
    train_exclude, y_def)

CMD = "python -m research_experiments.phase_6b_threshold_sampling_lstm.existing6b"


def rid(model, cfg, seed, r):
    return f"P6B-{model}-{cfg}-v1__s{seed}_r{r}"


def outer_member(base, cfg, seed, r):
    run = rid(base, cfg, seed, r)
    if done(run):
        return load_run(run)[0]
    mark_running(run, CMD)
    yt = y_def(cfg); P = np.zeros((nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = split(cfg, r, k)
        fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(BASES[base][0], fm)
        P[ote] = fit_member(base, X[itr], yt[itr], X[iva], yt[iva], seed, pre)(X[ote])
        info.append(dict(fold=k, n_inner_train=int(len(itr)), normal_inner_train=int((yt[itr] == 0).sum()),
                         normal_outer_training=int((yt[itr] == 0).sum() + (yt[iva] == 0).sum())))
    save_run(run, P, P.argmax(1), r, cfg, dict(model=base, config=cfg, seed=seed, repeat=r), extra=dict(folds=info))
    return P


def fit_predict(base, fit_rows, pred_rows, seed, split_seed, yt, excl):
    blocks, kind, cfg, _ = BASES[base]
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, yt[fit_rows], PAT[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    tr = tr[~excl[tr]]
    fm = np.zeros(nD, bool); fm[tr] = True
    X, pre = make_X(blocks, fm)
    return fit_member(base, X[tr], yt[tr], X[va], yt[va], seed, pre)(X[pred_rows])


def inner_oof(base, cfg, r, k):
    f = guard(INNER / f"{base}_{cfg}" / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    yt, excl = y_def(cfg), train_exclude(cfg, r, k)
    TR = np.flatnonzero(fold_vec(r) != k); P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yt[TR], PAT[TR])):
        P[ib] = np.mean([fit_predict(base, TR[ia], TR[ib], s, 2000 + 10 * k + j, yt, excl) for s in BASES[base][3]], 0)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P)
    return TR, P


def run_config(cfg):
    torch.set_num_threads(1); table()
    yt = y_def(cfg)
    for r in (0, 1, 2):
        Pm = np.mean([outer_member("mlpfusion", cfg, s, r) for s in SEEDS], 0)
        Pv = np.mean([Pm, outer_member("ecgflindist", cfg, 42, r), outer_member("hdrclinlr", cfg, 0, r)], 0)
        if not done(rid("vote", cfg, 0, r)):
            save_run(rid("vote", cfg, 0, r), Pv, Pv.argmax(1), r, cfg, dict(model="vote (C1 recipe)", config=cfg, repeat=r))
        if done(rid("c3", cfg, 0, r)):
            continue
        mark_running(rid("c3", cfg, 0, r), CMD)
        fo = fold_vec(r); pred = np.zeros(nD, int); alphas = []
        for k in range(5):
            inn = [inner_oof(b, cfg, r, k) for b in MEMBERS]
            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
            keep = ~train_exclude(cfg, r, k)[TR]
            prior = np.bincount(yt[TR][keep], minlength=4) / keep.sum()
            a = choose_alpha(Pin[keep], yt[TR][keep], prior, "protected"); alphas.append(a)
            pred[fo == k] = decide(Pv[fo == k], "prior", a, prior)
        save_run(rid("c3", cfg, 0, r), Pv, pred, r, cfg, dict(model="vote + nested prior rule, protected (C3 recipe)", config=cfg, repeat=r),
                 extra=dict(alpha_per_fold=alphas))
        print(f"[{cfg} r{r}] vote acc {metrics(Y, Pv, Pv.argmax(1))['accuracy']:.3f} c3 acc {np.mean(pred == Y):.3f} "
              f"(own labels {np.mean(pred == yt):.3f}) alpha {alphas}", flush=True)
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", nargs="+", default=["T55", "N300", "N270", "N240", "N200"])
    ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    table()
    with Pool(min(a.jobs, len(a.configs))) as pool:
        for c in pool.imap_unordered(run_config, a.configs):
            print("done", c, flush=True)


if __name__ == "__main__":
    main()
