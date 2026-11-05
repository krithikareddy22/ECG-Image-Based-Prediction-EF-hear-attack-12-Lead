"""Phase 7 E1: alternative ECGFounder representations in the MLP fusion recipe [header || representation].
Representations come from the existing ECGFounder cache (aligned input; stage 4 400 x 79, stage 5 1024 x 40 time
positions) - no backbone training. E1-dx = the frozen pre-trained 150-label head applied to f6 (sigmoid).
    python -m research_experiments.phase_7.e1_ecgf --pilot           # seed 42, 3 repeats, all representations
    python -m research_experiments.phase_7.e1_ecgf --full NAME ...   # seeds 42-44 for representations passing the gate
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from research_experiments.phase_2 import p2heads
from research_experiments.phase_5.experiments.nested import MLPF
from research_experiments.phase_5.p5common import blockstd, header_numeric
from research_experiments.phase_7 import p7common as C

CACHE = C.ROOT / "research_experiments" / "phase_6" / "data" / "private" / "embeddings" / "ecgf_cache.npz"
REPS = ("f6", "msp", "col", "s45", "dx")
_R = {}


def rep(name):
    if name in _R:
        return _R[name]
    z = np.load(CACHE)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[C.IDS].values
    f6 = z["f6"][pos].astype(np.float64)
    if name == "f6":
        X = f6
    elif name == "msp":
        s5 = z["stage5"][pos].astype(np.float64); X = np.concatenate([s5.mean(2), s5.max(2), s5.std(2)], 1)
    elif name == "col":
        s5 = z["stage5"][pos].astype(np.float64); X = np.concatenate([s5[:, :, 10 * c:10 * c + 10].mean(2) for c in range(4)], 1)
    elif name == "s45":
        X = np.concatenate([z["stage4"][pos].astype(np.float64).mean(2), f6], 1)
    elif name == "dx":
        from research_experiments.ecgfounder.model import load_ecgfounder, verify_checkpoint
        verify_checkpoint(); m = load_ecgfounder()
        with torch.no_grad():
            X = torch.sigmoid(m.dense(torch.tensor(f6, dtype=torch.float32))).numpy().astype(np.float64)
    else:
        raise KeyError(name)
    _R[name] = X
    return X


def rid(name, seed, r):
    return f"P7-E1-{name}-v1__s{seed}_r{r}"


def run(args):
    name, seed, r = args
    torch.set_num_threads(1)
    if C.done(rid(name, seed, r)):
        return rid(name, seed, r)
    R = rep(name); P = np.zeros((C.nD, 4)); info = []
    for k in range(5):
        itr, iva, ote = C.B.split_indices(r, k)
        fm = np.zeros(C.nD, bool); fm[itr] = True
        X = blockstd([header_numeric(fm), R], fm)
        f = p2heads.fit(X[itr], C.Y[itr], X[iva], C.Y[iva], dict(MLPF, no_std=True), seed=seed)
        P[ote] = f["predict"](X[ote])[0]; info.append(dict(fold=k, l2=f["l2"], best_epoch=f["best_epoch"], n_params=f["n_params"]))
    C.save(rid(name, seed, r), P, P.argmax(1), r, dict(experiment="E1", representation=name, dims=int(R.shape[1]), seed=seed,
                                                       recipe="Phase 2 MLP fusion [header || rep], block-standardised"), dict(folds=info))
    return rid(name, seed, r)


def pilot_report():
    ref = C.summary([[C.REF_MEMBER.format(r=r)] for r in (0, 1, 2)])
    out = dict(reference=dict(run=C.REF_MEMBER, **{k: ref[k] for k in ("accuracy", "macro_f1", "f1_Mild", "recall_Mild", "precision_Mild")}),
               pilots={})
    for n in REPS:
        s = C.summary([[rid(n, 42, r)] for r in (0, 1, 2)])
        out["pilots"][n] = dict(dims=int(rep(n).shape[1]), **{k: s[k] for k in ("accuracy", "accuracy_ci95_corrected", "balanced_accuracy", "macro_f1",
                                                                                   "f1_Mild", "recall_Mild", "precision_Mild", "min_recall")},
                                gate=C.gate(s, ref))
    (C.guard7(C.RES7) / "e1_pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--full", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=6); a = ap.parse_args()
    for n in REPS:
        rep(n)
    jobs = [(n, 42, r) for n in REPS for r in (0, 1, 2)] if a.pilot else []
    jobs += [(n, s, r) for n in a.full for s in C.SEEDS for r in (0, 1, 2)]
    if jobs:
        with Pool(a.jobs) as pool:
            for x in pool.imap_unordered(run, jobs):
                print("done", x, flush=True)
    if a.pilot:
        pilot_report()


if __name__ == "__main__":
    main()
