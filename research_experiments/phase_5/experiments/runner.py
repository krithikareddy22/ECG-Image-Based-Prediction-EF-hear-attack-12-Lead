"""Generic Phase-5 CV runner: one run = (experiment id, seed, repeat). Phase-2 protocol: in outer fold k the model is
fitted on inner-train, hyper-parameters / early stopping chosen on inner-val (both inside the outer training portion),
and the outer fold is predicted once. Only development ECGs exist here; nothing is predicted for the locked test.

kinds:
  sklr     StandardScaler + multinomial LR, class_weight balanced, C grid chosen on inner-val (S3 by default)
  p2heads  phase-2 heads (linear / MLP) with a cfg dict (exactly as phases 2-4)
  fitter   a callable fitter(Xtr, ytr, Xva, yva, seed, **kw) -> predict_proba(X)   (GBM, extended MLP, ...)
Optional: train_filter(itr_mask_rows, repeat, k) -> boolean mask over itr rows (E6), select='s3'|'accuracy'.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_ef.metrics import evaluate
from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_5.p5common import Y, done, make_X, mark_running, nD, save_run, split_indices

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def sel_score(y, P, select):
    return float(np.mean(P.argmax(1) == y)) if select == "accuracy" else s3(evaluate(y, P))


def fit_sklr(Xtr, ytr, Xva, yva, select="s3"):
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(Xtr, ytr)
        sc = sel_score(yva, clf.predict_proba(Xva), select)
        if best is None or sc > best[0] + 1e-12:
            best = (sc, C, clf)
    return best[2].predict_proba, dict(C=best[1], val_score=best[0])


def cv_run(exp_id, blocks, kind, seed, repeat, cfg=None, fitter=None, select="s3", train_filter=None, command="", extra_cfg=None,
           y_train=None):
    rid = f"{exp_id}__s{seed}_r{repeat}"
    if done(rid):
        return None
    mark_running(rid, command)
    t0 = time.time(); P = np.zeros((nD, 4)); info = []
    yt = Y if y_train is None else y_train
    for k in range(5):
        itr, iva, ote = split_indices(repeat, k)
        if train_filter is not None:
            itr = itr[train_filter(itr, repeat, k)]
        fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(blocks, fm)
        if kind == "sklr":
            pf, inf = fit_sklr(X[itr], yt[itr], X[iva], yt[iva], select)
        elif kind == "p2heads":
            f = p2heads.fit(X[itr], yt[itr], X[iva], yt[iva], dict(cfg, no_std=pre), seed=seed)
            pf = lambda Z, f=f: f["predict"](Z)[0]; inf = dict(l2=f["l2"], best_epoch=f["best_epoch"], val_s3=f["val_s3"])
        else:
            pf, inf = fitter(X[itr], yt[itr], X[iva], yt[iva], seed)
        P[ote] = pf(X[ote]); info.append(dict(fold=k, n_train=int(len(itr)), n_features=int(X.shape[1]), **inf))
    config = dict(exp_id=exp_id, blocks=blocks, kind=kind, cfg=cfg, select=select, seed=seed, repeat=repeat,
                  train_filter=getattr(train_filter, "__name__", None), **(extra_cfg or {}))
    agg = save_run(rid, P, P.argmax(1), repeat, config, extra=dict(folds=info, minutes=(time.time() - t0) / 60))
    m = agg["metrics"]
    print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} "
          f"Sev R/P {m['recall_Severe']:.2f}/{m['precision_Severe']:.2f} minR {m['min_recall']:.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return agg
