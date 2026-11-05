"""Phase 4D: continuous EF regression (pre-registered in phase_4/configs/PREREG_PHASE4.md).

Target: EF of the nearest-in-time linked echo (label-agnostic). Training and inner-val selection are restricted to the
training cohort (A all / C <= 14 d / G high-confidence); predictions are made for EVERY outer-fold ECG and evaluated on
all D (folder labels) and on the secondary sets. One run = one (model, features, cohort, repeat, seed); four EF->class
decoders are derived from it and stored as separate tags  <run>__{tuned,clin,recal,gauss}.

    python -m research_experiments.phase_4.regression --family ridge
    python -m research_experiments.phase_4.regression --family mlp --jobs 3
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from research_experiments.phase_4.p4common import (COHORTS, EF, MALE, REPEATS, SEEDS, Y, X_of, classify_cuts, done,
                                                     ef_class, gauss_probs, nD, save_run, short, split_indices, D,
                                                     tune_cuts)

ALPHAS = np.logspace(-1, 5, 13)
DECODERS = ("tuned", "clin", "recal", "gauss")


def fit_ridge(X, prestd, itr, iva):
    mu, sd = (X[itr].mean(0), X[itr].std(0) + 1e-6) if not prestd else (0.0, 1.0)
    Z = (X - mu) / sd
    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Z[itr], EF[itr]); mae = np.mean(np.abs(m.predict(Z[iva]) - EF[iva]))
        if best is None or mae < best[0]:
            best = (mae, a, m)
    mae, a, m = best
    return m.predict(Z), dict(alpha=float(a), inner_val_mae=float(mae))


class RegMLP(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.5):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.body(x))[:, 0]


def fit_mlp(X, prestd, itr, iva, seed, epochs=300, patience=30):
    mu, sd = (X[itr].mean(0), X[itr].std(0) + 1e-6) if not prestd else (0.0, 1.0)
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    emu, esd = EF[itr].mean(), EF[itr].std()
    yt = torch.tensor((EF - emu) / esd, dtype=torch.float32)
    best = None
    for l2 in (1e-4, 1e-3, 1e-2):
        torch.manual_seed(seed); m = RegMLP(Z.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=l2 * 100)
        g = torch.Generator().manual_seed(seed); tr = torch.tensor(itr)
        bmae, bstate, bad, bep = np.inf, None, 0, 0
        for ep in range(epochs):
            m.train(); idx = tr[torch.randperm(len(tr), generator=g)]
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]; lo = F.huber_loss(m(Z[b]), yt[b], delta=1.0)
                opt.zero_grad(); lo.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                mae = float(np.mean(np.abs(m(Z[iva]).numpy() * esd + emu - EF[iva])))
            if mae < bmae - 1e-6:
                bmae, bad, bep = mae, 0, ep; bstate = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        if best is None or bmae < best[0]:
            best = (bmae, l2, bstate, bep)
    bmae, l2, st, bep = best
    m = RegMLP(Z.shape[1]); m.load_state_dict(st); m.eval()
    with torch.no_grad():
        pred = m(Z).numpy() * esd + emu
    return pred, dict(l2=l2, best_epoch=bep, inner_val_mae=float(bmae))


def run(model, feat, cohort, repeat, seed=42):
    base = f"reg_{model}_{feat}_{cohort}_s{seed}_r{repeat}"
    if all(done(f"{base}__{d}") for d in DECODERS):
        return
    t0 = time.time(); co = COHORTS[cohort]
    ef_hat = np.zeros(nD); P = {d: np.zeros((nD, 4)) for d in DECODERS}; pred = {d: np.zeros(nD, int) for d in DECODERS}
    folds_info = []
    for k in range(5):
        itr, iva, ote = split_indices(D, repeat, k)
        itr_c, iva_c = itr[co[itr]], iva[co[iva]]                      # cohort restricts fitting + selection
        fm = np.zeros(nD, bool); fm[itr_c] = True
        X, prestd = X_of(feat, fm)
        if model == "ridge":
            yhat, inf = fit_ridge(X, prestd, itr_c, iva_c)
        else:
            yhat, inf = fit_mlp(X, prestd, itr_c, iva_c, seed)
        # decoders, all fitted on inner-val only
        b, a = np.polyfit(yhat[iva_c], EF[iva_c], 1)                   # linear recalibration EF ~ a + b * yhat
        rec = a + b * yhat
        sigma = float(np.std(rec[iva_c] - EF[iva_c], ddof=1))
        cuts, sc = tune_cuts(yhat[iva], Y[iva], MALE[iva])             # S3 vs folder labels on the full inner-val
        ef_hat[ote] = yhat[ote]
        gp = gauss_probs(rec[ote], sigma, MALE[ote])
        dec = {"tuned": classify_cuts(yhat[ote], MALE[ote], *cuts), "clin": ef_class(yhat[ote], MALE[ote]),
               "recal": ef_class(rec[ote], MALE[ote]), "gauss": gp.argmax(1)}
        for d in DECODERS:
            P[d][ote] = gp; pred[d][ote] = dec[d]
        folds_info.append(dict(fold=k, n_train=int(len(itr_c)), n_val=int(len(iva_c)), recal_a=float(a), recal_b=float(b),
                               sigma=sigma, cuts=cuts, cuts_val_s3=float(sc), **inf))
    for d in DECODERS:
        info = dict(family="4D_regression", model=model, features=feat, train_cohort=cohort, decoder=d, seed=seed,
                    folds=folds_info, minutes=(time.time() - t0) / 60,
                    n_train_cohort_dev=int(co.sum()), n_patients_train_cohort_dev=int(D.patient[co].nunique()))
        agg = save_run(f"{base}__{d}", P[d], pred[d], repeat, info, ef_hat=ef_hat)
        if d in ("tuned", "clin"):
            print(f"[{base}__{d}] {short(agg['eval'])} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def jobs(family):
    if family == "ridge":
        return [("ridge", f, c, r, 42) for f in ("hdr", "ecgf", "hdr+ecgf") for c in ("A", "C", "G") for r in REPEATS]
    return [("mlp", "hdr+ecgf", c, r, s) for c in ("A", "C", "G") for r in REPEATS for s in SEEDS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", choices=["ridge", "mlp"], required=True)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    J = jobs(a.family)[a.part::a.jobs]
    torch.set_num_threads(1)
    for j in J:
        run(*j)


if __name__ == "__main__":
    main()
