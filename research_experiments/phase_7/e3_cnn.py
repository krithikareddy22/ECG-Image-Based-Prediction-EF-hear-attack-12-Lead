"""Phase 7 E3: lightweight multi-lead 1D ResNet on the waveforms digitised from the PDFs (no recurrent layers).
Raw digital ECGs are not available; no signal is invented: each lead keeps only its own printed 2.5 s ("short",
12 x 1250 at 500 Hz) plus the 10-s lead-II rhythm strip. Band-pass as ECGFounder (0.67-40 Hz, 50-Hz notch, 0.4-s median
baseline), decimation 500 -> 250 Hz, clip +-5 mV (mV kept). Development rows only.
  lead branch   12 x 625  : stem conv(12->32, k7, s2) + residual blocks 32->64->128->128 (k7, s2) -> global average pool
  rhythm branch 1 x 2500  : stem conv(1->16, k7, s2) + residual blocks 16->32->64->64 (k7, s2) -> global average pool
  head          [192 (|| header 16 for cnnhdr)] -> dropout 0.3 -> 4 classes; EF head (Huber, verified-EF rows, weight 0.5)
Class-balanced CE, AdamW 1e-3 (wd 1e-2), batch 32, <= 60 epochs, early stopping on inner-val S3 (patience 12, min 10).
    python -m research_experiments.phase_7.e3_cnn --pilot            # repeat 0, seed 42, both variants
    python -m research_experiments.phase_7.e3_cnn --full VARIANT ...
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from research_experiments.ecgfounder.inputs import bandpass
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_5.p5common import header_numeric
from research_experiments.phase_7 import p7common as C

CACHE = C.PRIV7 / "cnn_inputs.npz"
VARIANTS = ("cnn", "cnnhdr")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inputs():
    if CACHE.exists():
        z = np.load(CACHE); assert (z["ecg_id"] == np.asarray(C.IDS, dtype="U32")).all(); return z["L"], z["R"]
    z = np.load(C.ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[C.IDS].values      # development rows only
    Ls, Rs = z["leads"][pos], z["rhythm"][pos]
    L = np.stack([np.stack([np.clip(bandpass(np.asarray(Ls[i][j], np.float64)), -5, 5)[::2] for j in range(12)]) for i in range(len(pos))])
    R = np.stack([np.clip(bandpass(np.asarray(Rs[i], np.float64)), -5, 5)[::2] for i in range(len(pos))])[:, None]
    C.guard7(C.PRIV7).mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ecg_id=np.asarray(C.IDS, dtype="U32"), L=L.astype(np.float32), R=R.astype(np.float32))
    return L.astype(np.float32), R.astype(np.float32)


class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv1d(i, o, 7, s, 3, bias=False); self.b1 = nn.BatchNorm1d(o)
        self.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False); self.b2 = nn.BatchNorm1d(o)
        self.sk = nn.Sequential(nn.Conv1d(i, o, 1, s, bias=False), nn.BatchNorm1d(o))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sk(x))


def branch(cin, chans):
    layers = [nn.Conv1d(cin, chans[0], 7, 2, 3, bias=False), nn.BatchNorm1d(chans[0]), nn.ReLU()]
    for a, b in zip(chans[:-1], chans[1:]):
        layers.append(Block(a, b, 2))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())


class Net(nn.Module):
    def __init__(self, n_hdr=0):
        super().__init__()
        self.lead = branch(12, [32, 64, 128, 128]); self.rhy = branch(1, [16, 32, 64, 64])
        d = 192 + n_hdr
        self.drop = nn.Dropout(0.3); self.out = nn.Linear(d, 4); self.efh = nn.Linear(d, 1)

    def forward(self, L, R, H=None):
        h = torch.cat([self.lead(L), self.rhy(R)] + ([H] if H is not None else []), 1)
        h = self.drop(h); return self.out(h), self.efh(h)[:, 0]


def fit_fold(variant, seed, itr, iva, ote, L, R, Hall, ver, ef):
    torch.manual_seed(seed); np.random.seed(seed)
    H = torch.tensor(Hall, dtype=torch.float32) if variant == "cnnhdr" else None
    Lt, Rt = torch.tensor(L), torch.tensor(R)
    y = torch.tensor(C.Y); cw = class_w(C.Y[itr], "balanced").to(DEV)
    vt = torch.tensor(ver); eft = torch.tensor(np.nan_to_num((ef - 50) / 10), dtype=torch.float32)
    m = Net(Hall.shape[1] if variant == "cnnhdr" else 0).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)

    def predict(rows):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                b = rows[i:i + 128]
                lo, _ = m(Lt[b].to(DEV), Rt[b].to(DEV), H[b].to(DEV) if H is not None else None)
                out.append(F.softmax(lo, 1).cpu().numpy())
        return np.concatenate(out)

    best, state, bad, best_ep = -1, None, 0, 0
    for ep in range(60):
        m.train(); perm = itr[torch.randperm(len(itr), generator=g).numpy()]
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            lo, efh = m(Lt[b].to(DEV), Rt[b].to(DEV), H[b].to(DEV) if H is not None else None)
            yb = y[b].to(DEV); w = cw[yb]
            loss = (F.cross_entropy(lo, yb, reduction="none") * w).sum() / w.sum()
            vb = vt[b].to(DEV)
            if vb.any():
                loss = loss + 0.5 * F.huber_loss(efh[vb], eft[b].to(DEV)[vb])
            opt.zero_grad(); loss.backward(); opt.step()
        sc = s3(evaluate(C.Y[iva], predict(iva)))
        if sc > best + 1e-6:
            best, best_ep, bad = sc, ep, 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        if bad >= 12 and ep >= 10:
            break
    m.load_state_dict(state)
    return predict(ote), dict(best_epoch=best_ep, val_s3=best, n_params=sum(p.numel() for p in m.parameters()))


def rid(variant, seed, r):
    return f"P7-E3-{variant}-v1__s{seed}_r{r}"


def run(variant, seed, r, repeat_only=False):
    if C.done(rid(variant, seed, r)):
        return
    L, R = inputs(); ver, ef = C.verified(); P = np.zeros((C.nD, 4)); info = []; t0 = time.time()
    for k in range(5):
        itr, iva, ote = C.B.split_indices(r, k)
        fm = np.zeros(C.nD, bool); fm[itr] = True
        Hall = header_numeric(fm).astype(np.float32)
        mu, sd = Hall[itr].mean(0), Hall[itr].std(0) + 1e-6; Hall = (Hall - mu) / sd
        P[ote], inf = fit_fold(variant, seed, itr, iva, ote, L, R, Hall, ver, np.where(ver, ef, np.nan))
        info.append(dict(fold=k, **inf)); print(f"  {variant} s{seed} r{r} k{k} ep {inf['best_epoch']} acc {np.mean(P[ote].argmax(1) == C.Y[ote]):.3f}", flush=True)
    C.save(rid(variant, seed, r), P, P.argmax(1), r, dict(experiment="E3", variant=variant, seed=seed, device=str(DEV)),
           dict(folds=info, train_minutes=(time.time() - t0) / 60))


def pilot_report():
    out = {}
    ref = C.summary([[C.REF_MEMBER.format(r=0)]])
    for v in VARIANTS:
        s = C.summary([[rid(v, 42, 0)]])
        g = C.gate(s, ref)
        out[v] = dict(**{k: s[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "precision_Mild", "min_recall")},
                      reference_repeat0=dict(accuracy=ref["accuracy"], macro_f1=ref["macro_f1"], f1_Mild=ref["f1_Mild"]),
                      gate=g, passes=bool(g["passes"] and s["accuracy"] >= 0.50))
    (C.guard7(C.RES7) / "e3_pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--full", nargs="*", default=[])
    a = ap.parse_args()
    inputs()
    if a.pilot:
        for v in VARIANTS:
            run(v, 42, 0)
        pilot_report()
    for v in a.full:
        for s in C.SEEDS:
            for r in (0, 1, 2):
                run(v, s, r)


if __name__ == "__main__":
    main()
