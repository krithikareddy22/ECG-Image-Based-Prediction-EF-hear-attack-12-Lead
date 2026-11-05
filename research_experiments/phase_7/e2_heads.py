"""Phase 7 E2: continuous-EF supervision and Mild separation on [header || ECGFounder f6] (PHASE7_PLAN.md section 3).
Phase 2 MLP architecture (dropout 0.5 -> 64 GELU -> dropout 0.5 -> 4), AdamW lr 1e-3, batch 64, <= 200 epochs, early
stopping on inner-val S3 (patience 25), class-balanced CE; the grid below is searched on inner-val only.
  ce   plain class-balanced CE (control)
  mt   CE + lam * Huber((EF_hat - EF) / 10) on verified-EF rows                       lam   in {0.1, 0.5, 1.0}
  ldl  verified rows: soft targets P(class | EF ~ N(EF_echo, sigma^2)) over the        sigma in {3, 5, 7}
       sex-specific class bands; other rows hard CE
  scl  CE + beta * SupCon on the L2-normalised hidden layer (tau 0.1); positive weight  beta  in {0.1, 0.3}
       exp(-|dEF| / 5) between verified rows, else 1 same class / 0 different class
    python -m research_experiments.phase_7.e2_heads --pilot
    python -m research_experiments.phase_7.e2_heads --full MODE ...
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_5.p5common import blockstd, ecgf_f6, header_numeric
from research_experiments.phase_7 import p7common as C

MODES = ("ce", "mt", "ldl", "scl")
GRID = {"ce": [None], "mt": [0.1, 0.5, 1.0], "ldl": [3.0, 5.0, 7.0], "scl": [0.1, 0.3]}
L2S = (1e-4, 1e-3, 1e-2)


def soft_targets(ef, bnd, sigma):
    a = norm.cdf((31 - ef) / sigma); b = norm.cdf((41 - ef) / sigma); c = norm.cdf((bnd - ef) / sigma)
    Q = np.column_stack([1 - c, c - b, b - a, a])                 # Normal, Mild, Moderate, Severe
    return Q / Q.sum(1, keepdims=True)


class Net(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.5):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 4); self.efh = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x); return h, self.out(h), self.efh(h)[:, 0]


def supcon(h, W, tau=0.1):
    z = F.normalize(h, dim=1); sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool)
    logp = sim.masked_fill(eye, -1e9).log_softmax(1)
    W = W.masked_fill(eye, 0.0); den = W.sum(1)
    ok = den > 0
    return -((W * logp).sum(1)[ok] / den[ok]).mean() if ok.any() else h.sum() * 0


def train(Xtr, ytr, Xva, yva, mode, param, l2, seed, ver_tr, ef_tr, bnd_tr):
    torch.manual_seed(seed)
    xtr, xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    yt = torch.tensor(ytr); cw = class_w(ytr, "balanced")
    vt = torch.tensor(ver_tr); eft = torch.tensor(np.nan_to_num((ef_tr - 50) / 10), dtype=torch.float32)
    # finite placeholder EF for unverified rows (never selected; avoids NaN gradients through torch.where)
    Q = torch.tensor(soft_targets(np.where(ver_tr, ef_tr, 50.0), bnd_tr, param), dtype=torch.float32) if mode == "ldl" else None
    efraw = torch.tensor(np.nan_to_num(ef_tr), dtype=torch.float32)
    m = Net(Xtr.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=l2 * 100)
    g = torch.Generator().manual_seed(seed)
    best, state, bad, best_ep = -1, None, 0, 0
    for ep in range(200):
        m.train(); idx = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; h, lo, efh = m(xtr[b]); w = cw[yt[b]]
            if mode == "ldl":
                logp = F.log_softmax(lo, 1)
                hard = F.cross_entropy(lo, yt[b], reduction="none"); soft = -(Q[b] * logp).sum(1)
                li = torch.where(vt[b], soft, hard)
            else:
                li = F.cross_entropy(lo, yt[b], reduction="none")
            loss = (li * w).sum() / w.sum()
            if mode == "mt" and vt[b].any():
                loss = loss + param * F.huber_loss(efh[vt[b]], eft[b][vt[b]])
            if mode == "scl":
                vb = vt[b]; yb = yt[b]
                both = vb[:, None] & vb[None, :]
                Wef = torch.exp(-(efraw[b][:, None] - efraw[b][None, :]).abs() / 5)
                Wcls = (yb[:, None] == yb[None, :]).float()
                loss = loss + param * supcon(h, torch.where(both, Wef, Wcls))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(xva)[1], 1).numpy()
        sc = s3(evaluate(yva, p))
        if sc > best + 1e-6:
            best, best_ep, bad = sc, ep, 0; state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        if bad >= 25:
            break
    m.load_state_dict(state); m.eval()
    return best, best_ep, m


def rid(mode, seed, r):
    return f"P7-E2-{mode}-v1__s{seed}_r{r}"


def run(args):
    mode, seed, r = args
    torch.set_num_threads(1)
    if C.done(rid(mode, seed, r)):
        return rid(mode, seed, r)
    ver, ef = C.verified(); T = C.B.table(); bnd = T.sex_boundary.values
    E = ecgf_f6().astype(float); P = np.zeros((C.nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = C.B.split_indices(r, k)
        fm = np.zeros(C.nD, bool); fm[itr] = True
        X = blockstd([header_numeric(fm), E], fm)
        best = None
        for prm in GRID[mode]:
            for l2 in L2S:
                sc, ep, m = train(X[itr], C.Y[itr], X[iva], C.Y[iva], mode, prm, l2, seed, ver[itr], np.where(ver[itr], ef[itr], np.nan), bnd[itr])
                if best is None or sc > best[0] + 1e-9:
                    best = (sc, prm, l2, ep, m)
        sc, prm, l2, ep, m = best
        with torch.no_grad():
            P[ote] = F.softmax(m(torch.tensor(X[ote], dtype=torch.float32))[1], 1).numpy()
        info.append(dict(fold=k, param=prm, l2=l2, best_epoch=ep, val_s3=sc))
    C.save(rid(mode, seed, r), P, P.argmax(1), r, dict(experiment="E2", mode=mode, seed=seed, grid=GRID[mode], l2=list(L2S),
                                                        inputs="[header || ECGFounder f6], block-standardised"), dict(folds=info))
    return rid(mode, seed, r)


def pilot_report():
    ref = C.summary([[C.REF_MEMBER.format(r=r)] for r in (0, 1, 2)])
    ctl = C.summary([[rid("ce", 42, r)] for r in (0, 1, 2)])
    keys = ("accuracy", "accuracy_ci95_corrected", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "precision_Mild", "min_recall")
    out = dict(reference={k: ref[k] for k in keys}, pilots={})
    for mode in MODES:
        s = C.summary([[rid(mode, 42, r)] for r in (0, 1, 2)])
        g1, g2 = C.gate(s, ref), C.gate(s, ctl)
        out["pilots"][mode] = dict(**{k: s[k] for k in keys}, gate_vs_reference=g1, gate_vs_ce_control=g2,
                                   passes=bool(g1["passes"] and (g2["passes"] or mode == "ce")))
    (C.guard7(C.RES7) / "e2_pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--full", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=6); a = ap.parse_args()
    jobs = [(m, 42, r) for m in MODES for r in (0, 1, 2)] if a.pilot else []
    jobs += [(m, s, r) for m in a.full for s in C.SEEDS for r in (0, 1, 2)]
    if jobs:
        with Pool(a.jobs) as pool:
            for x in pool.imap_unordered(run, jobs):
                print("done", x, flush=True)
    if a.pilot:
        pilot_report()


if __name__ == "__main__":
    main()
