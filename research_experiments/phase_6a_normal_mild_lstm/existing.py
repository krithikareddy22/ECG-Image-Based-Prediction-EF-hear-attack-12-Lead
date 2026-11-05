"""Phase 6A: the existing best model recipe re-trained under each label configuration (B1, B2, B3, A270).

Members (identical recipes to Phases 2-5; phase_2.p2heads / sklearn LR via phase_5 code):
  mlpfusion    header numeric || ECGFounder f6, MLP (64, dropout 0.5), class-balanced CE, seeds 42/43/44
  ecgflindist  ECGFounder f6 linear head, distance-penalised CE (lambda 1), seed 42
  hdrclinlr    header numeric + 28 clinical features, multinomial LR (balanced), C on inner-val S3
  vote (= C1 recipe)  equal-weight mean of [mlpfusion seed-mean, ecgflindist, hdrclinlr]
  c3   (= C3 recipe)  vote + nested class-prior rule (alpha on the training portion's nested inner OOF, protected objective)
Configuration handling: training labels (B2), inner-train rows (B3: uncertain removed; A270: Normal patients moved
from inner-val), applied to every fit, including the nested inner models (B2, B3). Evaluation always on all 837 ECGs.
B1 must reproduce the stored Phase 2/3/5 predictions within 0.005 (checked by `--check`).
    python -m research_experiments.phase_6a_normal_mild_lstm.existing --configs B1 B2 B3 A270 --jobs 4
    python -m research_experiments.phase_6a_normal_mild_lstm.existing --check
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_2 import p2heads
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES
from research_experiments.phase_5.experiments.runner import fit_sklr
from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (
    INNER, PAT, RES, SEEDS, Y, decide, done, fold_vec, guard, label_table, load_run, make_X, mark_running, metrics, nD,
    save_run, split, y_train)

MEMBERS = ("mlpfusion", "ecgflindist", "hdrclinlr")
CMD = "python -m research_experiments.phase_6a_normal_mild_lstm.existing"


def fit_member(base, Xtr, ytr, Xva, yva, seed, pre):
    blocks, kind, cfg, _ = BASES[base]
    if kind == "sklr":
        return fit_sklr(Xtr, ytr, Xva, yva)[0]
    f = p2heads.fit(Xtr, ytr, Xva, yva, dict(cfg, no_std=pre), seed=seed)
    return lambda Z, f=f: f["predict"](Z)[0]


def rid(model, cfg, seed, r):
    return f"P6A-{model}-{cfg}-v1__s{seed}_r{r}"


def outer_member(base, cfg, seed, r):
    """Outer CV predictions of one member under a configuration (all 837 ECGs)."""
    run = rid(base, cfg, seed, r)
    if done(run):
        return load_run(run)[0]
    mark_running(run, CMD)
    yt = y_train(cfg); P = np.zeros((nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = split(cfg, r, k)
        fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(BASES[base][0], fm)
        pf = fit_member(base, X[itr], yt[itr], X[iva], yt[iva], seed, pre)
        P[ote] = pf(X[ote]); info.append(dict(fold=k, n_inner_train=int(len(itr)), n_inner_val=int(len(iva)),
                                              normal_inner_train=int((yt[itr] == 0).sum())))
    save_run(run, P, P.argmax(1), r, cfg, dict(model=base, label_config=cfg, seed=seed, repeat=r), extra=dict(folds=info))
    return P


# ------------------------------------------------------------------ nested inner OOF per configuration
def fit_predict(base, fit_rows, pred_rows, seed, split_seed, yt, excl):
    """phase_5.nested.fit_predict with configurable training labels and training-only exclusion."""
    blocks, kind, cfg, _ = BASES[base]
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, yt[fit_rows], PAT[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    tr = tr[~excl[tr]]
    fm = np.zeros(nD, bool); fm[tr] = True
    X, pre = make_X(blocks, fm)
    return fit_member(base, X[tr], yt[tr], X[va], yt[va], seed, pre)(X[pred_rows])


def inner_oof(base, cfg, r, k):
    if cfg in ("B1", "A270"):                     # nested inner models do not use inner-train / inner-val -> B1 caches
        z = np.load(INNER6 / base / f"r{r}_k{k}.npz"); return z["TR"], z["P"]
    f = guard(INNER / f"{base}_{cfg}" / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    yt = y_train(cfg)
    excl = label_table().b3_uncertain.values if cfg == "B3" else np.zeros(nD, bool)
    TR = np.flatnonzero(fold_vec(r) != k)
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k)
    P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(inner.split(TR, yt[TR], PAT[TR])):
        P[ib] = np.mean([fit_predict(base, TR[ia], TR[ib], s, 2000 + 10 * k + j, yt, excl) for s in BASES[base][3]], 0)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P)
    return TR, P


def run_config(cfg):
    torch.set_num_threads(1)
    yt = y_train(cfg)
    excl = label_table().b3_uncertain.values if cfg == "B3" else np.zeros(nD, bool)
    for r in (0, 1, 2):
        Pm = np.mean([outer_member("mlpfusion", cfg, s, r) for s in SEEDS], 0)
        Pl = outer_member("ecgflindist", cfg, 42, r)
        Ph = outer_member("hdrclinlr", cfg, 0, r)
        Pv = np.mean([Pm, Pl, Ph], 0)
        if not done(rid("vote", cfg, 0, r)):
            save_run(rid("vote", cfg, 0, r), Pv, Pv.argmax(1), r, cfg, dict(model="vote (C1 recipe)", label_config=cfg, repeat=r,
                                                                         members=list(MEMBERS)))
        if done(rid("c3", cfg, 0, r)):
            continue
        mark_running(rid("c3", cfg, 0, r), CMD)
        fo = fold_vec(r); pred = np.zeros(nD, int); alphas = []
        for k in range(5):
            inn = [inner_oof(b, cfg, r, k) for b in MEMBERS]
            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
            keep = ~excl[TR]                                   # B3: uncertain rows are training data -> excluded here too
            prior = np.bincount(yt[TR][keep], minlength=4) / keep.sum()
            a = choose_alpha(Pin[keep], yt[TR][keep], prior, "protected"); alphas.append(a)
            pred[fo == k] = decide(Pv[fo == k], "prior", a, prior)
        save_run(rid("c3", cfg, 0, r), Pv, pred, r, cfg, dict(model="vote + nested prior rule, protected (C3 recipe)", label_config=cfg,
                                                              repeat=r, inner_oof="B1 caches" if cfg in ("B1", "A270") else f"refit under {cfg}"),
                 extra=dict(alpha_per_fold=alphas))
        print(f"[{cfg} r{r}] vote acc {metrics(Y, Pv, Pv.argmax(1))['accuracy']:.3f}  c3 acc {np.mean(pred == Y):.3f} alpha {alphas}", flush=True)
    return cfg


def check():
    """B1 (same code) vs the stored predictions regenerated in phase_6 (C1 members, C1 vote, C3)."""
    from research_experiments.phase_6.p6common import load_run as load6
    from research_experiments.phase_6.p6common import load_agg5
    rows = []
    pairs = [("mlpfusion", s, f"P5A-import-mlpfusion-v1__s{s}") for s in SEEDS] + \
            [("ecgflindist", 42, "P5A-import-ecgflindist-v1__s0"), ("hdrclinlr", 0, "P5A-import-hdrclinlr-v1__s0"),
             ("vote", 0, "P5A-import-vote-v1__s0"), ("c3", 0, "P5D-E5-vote-prior-protected-v1__s0")]
    for model, seed, ref in pairs:
        for r in (0, 1, 2):
            P, pred = load_run(rid(model, "B1", seed, r)); m = metrics(Y, P, pred)
            R = load_agg5(f"{ref}_r{r}")["metrics"]
            d = {k: abs(m[k] - R[k]) for k in ("accuracy", "balanced_accuracy", "macro_f1")}
            Pr, _ = load6(f"{ref}_r{r}")
            rows.append(dict(model=model, seed=seed, repeat=r, **{f"diff_{k}": v for k, v in d.items()},
                             max_abs_prob_diff=float(np.abs(P - Pr).max()), within_0_005=max(d.values()) <= 0.005))
    ok = all(x["within_0_005"] for x in rows)
    (guard(RES) / "b1_reproduction_check.json").write_text(json.dumps(dict(all_within_0_005=ok, rows=rows), indent=1))
    for x in rows:
        print(x)
    print("B1 reproduces the stored models within 0.005:", ok)
    if not ok:
        sys.exit("STOP: same-code B1 does not reproduce the stored baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["B1", "B2", "B3", "A270"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(); return
    label_table()
    with Pool(min(a.jobs, len(a.configs))) as pool:
        for c in pool.imap_unordered(run_config, a.configs):
            print("done", c, flush=True)


if __name__ == "__main__":
    main()
