"""Nested inner-CV engine (training-only out-of-fold predictions) for base models.

For outer fold k of repeat r, the training portion TR = all development ECGs outside fold k is split into a
patient-grouped 4-fold inner CV (random_state 1000 + 10 r + k). Every inner model is fitted on TR minus inner fold j,
using its own patient-grouped 80/20 split for model selection (random_state 2000 + 10 k + j), and predicts inner fold j.
This mirrors phase_2/nested_calibration.py exactly. The resulting inner OOF predictions are the ONLY data that stackers
(E3) and decision rules (5A calibration, E5) are fitted on; the outer fold is never involved.
Cache (git-ignored): phase_5/data/private/inner/<base>/r<r>_k<k>.npz  (TR indices + probabilities)
    python -m research_experiments.phase_5.experiments.nested --bases mlpfusion ecgflindist hdrclinlr --repeat 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_ef.metrics import evaluate
from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_5.p5common import PAT, PRIV5, SEEDS, Y, fold_vec, guard5, make_X, nD

LIN = dict(head="linear", loss="ce", weight="balanced")
MLPF = dict(LIN, head="mlp", hidden=64, dropout=0.5)
BASES = {   # name -> (feature blocks, kind, cfg, seeds); identical to the phase-2/3 vote members
    "mlpfusion": (["hdr", "ecgf"], "p2heads", MLPF, SEEDS),
    "ecgflindist": (["ecgf"], "p2heads", dict(LIN, loss="ce_dist", lam=1.0), (42,)),
    "hdrclinlr": (["hdr", "feat:clin3"], "sklr", None, (42,)),
}
INNER = PRIV5 / "inner"


def fit_predict(blocks, kind, cfg, fit_rows, pred_rows, seed, split_seed, y=None, fitter=None):
    """Fit on fit_rows (own grouped 80/20 split for selection), return probabilities for pred_rows."""
    y = Y if y is None else y
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, y[fit_rows], PAT[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    fm = np.zeros(nD, bool); fm[tr] = True
    X, pre = make_X(blocks, fm)
    if fitter is not None:
        return fitter(X[tr], y[tr], X[va], y[va], seed)(X[pred_rows])
    if kind == "sklr":
        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr], y[tr])
            sc = s3(evaluate(y[va], clf.predict_proba(X[va])))
            if best is None or sc > best[0]:
                best = (sc, clf)
        return best[1].predict_proba(X[pred_rows])
    f = p2heads.fit(X[tr], y[tr], X[va], y[va], dict(cfg, no_std=pre), seed=seed)
    return f["predict"](X[pred_rows])[0]


def inner_oof(base, r, k, spec=None):
    """-> (TR indices, inner-OOF probabilities over TR), cached."""
    f = guard5(INNER / base / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    blocks, kind, cfg, seeds = spec or BASES[base]
    TR = np.flatnonzero(fold_vec(r) != k)
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k)
    P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(inner.split(TR, Y[TR], PAT[TR])):
        assert not set(PAT[TR[ia]]) & set(PAT[TR[ib]]), "inner patient leakage"
        P[ib] = np.mean([fit_predict(blocks, kind, cfg, TR[ia], TR[ib], s, 2000 + 10 * k + j) for s in seeds], 0)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, TR=TR, P=P)
    return TR, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bases", nargs="+", default=list(BASES))
    ap.add_argument("--repeat", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); torch.set_num_threads(1)
    for r in a.repeat:
        for base in a.bases:
            for k in range(5):
                TR, P = inner_oof(base, r, k)
                print(f"[inner {base} r{r} k{k}] n={len(TR)} inner-OOF acc {np.mean(P.argmax(1) == Y[TR]):.3f}", flush=True)


if __name__ == "__main__":
    main()
