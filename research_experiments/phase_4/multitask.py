"""Phase 4E: regression-classification multitask learning (pre-registered).

Shared body = the phase-2 MLP fusion body (Dropout 0.5 -> Linear(d, 64) -> GELU -> Dropout 0.5) on header numeric ||
ECGFounder f6 (block-standardised), with two heads:
  class head  4-way, class-balanced CE on the FOLDER label (as in phase 2)
  EF head     Huber(delta 1) on z-scored nearest-echo EF, masked to the EF cohort (A / C / G)
loss = CE + lambda * Huber  (fixed lambda)  or  uncertainty weighting  sum_i exp(-s_i) L_i + s_i  (Kendall et al.).
Optimiser, batches, epochs, early stopping (inner-val S3) and the L2 grid are identical to phase_2/p2heads.py, and the
EF head is initialised from a saved RNG state, so lambda = 0 reproduces the phase-2 MLP fusion exactly (checked).

    python -m research_experiments.phase_4.multitask --jobs 2 --part 0
    python -m research_experiments.phase_4.multitask --select      # builds mt_sel (lambda chosen per fold on inner-val)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_4.p4common import (AGG, COHORTS, D, EF, REPEATS, SEEDS, Y, X_of, done, load_run, nD,
                                                     save_run, short, split_indices)

VARIANTS = {"lam0_A": (0.0, "A"), "lam0.3_A": (0.3, "A"), "lam1_A": (1.0, "A"), "lam3_A": (3.0, "A"),
            "lam1_C": (1.0, "C"), "lam1_G": (1.0, "G"), "uw_A": ("uw", "A")}
# EXPLORATORY (not pre-registered): ordinal-aware class loss (phase-2 ce_dist: CE + E_p[(k - y)^2] / 9) with lambda 1
EXPLORATORY = {"lam1_A_dist": (1.0, "A")}   # its CE-only control exists from phase 2 (ff_mlp_cedist1, repeat 0)


class MT(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.5):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 4)
        st = torch.get_rng_state(); self.ef = nn.Linear(hidden, 1); torch.set_rng_state(st)
        self.s = nn.Parameter(torch.zeros(2))           # log-variances (uncertainty weighting only)

    def forward(self, x):
        h = self.body(x); return self.out(h), self.ef(h)[:, 0]


def fit(Xtr, ytr, eftr, emask, Xva, yva, lam, seed, epochs=200, patience=25, dist=False):
    xtr, xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    yt = torch.tensor(ytr); et = torch.tensor(eftr, dtype=torch.float32); mt = torch.tensor(emask)
    cw = class_w(ytr, "balanced")
    best = None
    for l2 in (1e-4, 1e-3, 1e-2):
        torch.manual_seed(seed); m = MT(Xtr.shape[1])
        main = [p for n, p in m.named_parameters() if n != "s"]
        groups = [dict(params=main, weight_decay=l2 * 100)] + ([dict(params=[m.s], weight_decay=0.0)] if lam == "uw" else [])
        opt = torch.optim.AdamW(groups, lr=1e-3)
        g = torch.Generator().manual_seed(seed)
        best_ep, sc_best, state, bad = 0, -1, None, 0
        for ep in range(epochs):
            m.train(); idx = torch.randperm(len(ytr), generator=g)
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                torch.randn_like(xtr[b])        # p2heads draws feature noise (x 0.0) here; keeps the RNG stream identical
                logits, efp = m(xtr[b])
                w = cw[yt[b]]; li = F.cross_entropy(logits, yt[b], reduction="none")
                if dist:
                    kk = torch.arange(4, dtype=torch.float32)
                    li = li + (F.softmax(logits, 1) * (kk[None] - yt[b][:, None].float()) ** 2).sum(1) / 9
                ce = (li * w).sum() / w.sum()
                mb = mt[b]
                hub = F.huber_loss(efp[mb], et[b][mb], delta=1.0) if mb.any() else torch.zeros(())
                if lam == "uw":
                    lo = torch.exp(-m.s[0]) * ce + m.s[0] + torch.exp(-m.s[1]) * hub + m.s[1]
                elif lam == 0:
                    lo = ce
                else:
                    lo = ce + lam * hub
                opt.zero_grad(); lo.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                p = F.softmax(m(xva)[0], 1).numpy()
            sc = s3(evaluate(yva, p))
            if sc > sc_best + 1e-6:
                sc_best, best_ep, bad = sc, ep, 0; state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        m.load_state_dict(state)
        if best is None or sc_best > best[0] + 1e-9:
            best = (sc_best, l2, m, best_ep)
    sc, l2, m, ep = best; m.eval()

    def predict(X):
        with torch.no_grad():
            lo, e = m(torch.tensor(X, dtype=torch.float32)); return F.softmax(lo, 1).numpy(), e.numpy()
    return predict, dict(l2=l2, val_s3=float(sc), best_epoch=ep, s=[float(v) for v in m.s.detach()])


def run(variant, repeat, seed):
    tag = f"mt_{variant}_s{seed}_r{repeat}"
    if done(tag):
        return
    lam, coh = {**VARIANTS, **EXPLORATORY}[variant]; co = COHORTS[coh]; t0 = time.time(); dist = variant.endswith("_dist")
    P = np.zeros((nD, 4)); ef_hat = np.zeros(nD); info = []
    for k in range(5):
        itr, iva, ote = split_indices(D, repeat, k)
        fm = np.zeros(nD, bool); fm[itr] = True
        X, _ = X_of("hdr+ecgf", fm)
        emu, esd = EF[itr][co[itr]].mean(), EF[itr][co[itr]].std()
        pf, inf = fit(X[itr], Y[itr], (EF[itr] - emu) / esd, co[itr], X[iva], Y[iva], lam, seed, dist=dist)
        P[ote], e = pf(X[ote]); ef_hat[ote] = e * esd + emu
        info.append(dict(fold=k, n_train=int(len(itr)), n_ef_train=int(co[itr].sum()), **inf))
    agg = save_run(tag, P, P.argmax(1), repeat,
                   dict(family="4E_multitask", variant=variant, lam=lam, ef_cohort=coh, seed=seed, folds=info,
                        exploratory=variant in EXPLORATORY, ordinal_aware_loss=dist,
                        minutes=(time.time() - t0) / 60), ef_hat=ef_hat)
    print(f"[{tag}] {short(agg['eval'])} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def select():
    """mt_sel: per fold, the fixed lambda (cohort A) with the best inner-val S3 (no outer-fold information)."""
    from research_experiments.phase_4.p4common import fold_vec
    for r in REPEATS:
        for s in SEEDS:
            tag = f"mt_sel_s{s}_r{r}"
            cands = [f"mt_{v}_s{s}_r{r}" for v in ("lam0_A", "lam0.3_A", "lam1_A", "lam3_A")]
            infos = {c: json.loads((AGG / f"{c}.json").read_text())["info"]["folds"] for c in cands}
            fo = fold_vec(r); P = np.zeros((nD, 4)); ef_hat = np.zeros(nD); chosen = []
            for k in range(5):
                c = max(cands, key=lambda t: infos[t][k]["val_s3"]); chosen.append(c.split("_")[1])
                Pc, _, efc, _ = load_run(c); P[fo == k] = Pc[fo == k]; ef_hat[fo == k] = efc[fo == k]
            agg = save_run(tag, P, P.argmax(1), r, dict(family="4E_multitask", variant="sel", chosen_lambda_per_fold=chosen,
                                                       seed=s), ef_hat=ef_hat)
            print(f"[{tag}] chosen {chosen} {short(agg['eval'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--select", action="store_true"); ap.add_argument("--variants", nargs="*")
    a = ap.parse_args(); torch.set_num_threads(1)
    if a.select:
        return select()
    J = [(v, r, s) for v in (a.variants or VARIANTS) for r in REPEATS for s in SEEDS][a.part::a.jobs]
    for j in J:
        run(*j)


if __name__ == "__main__":
    main()
