"""Fully NESTED decision-rule calibration (class-prior correction and per-class thresholds).

Two separate ingredients, from different data:
  (1) class prior pi      : label frequencies of the TRAINING PORTION of outer fold k (folds != k)
  (2) thresholds / alpha  : selected on GENUINELY out-of-fold predictions INSIDE that training portion:
                            an inner 4-fold patient-grouped CV is run on the training portion only; every
                            inner model predicts the inner fold it did not see (and did not use for model
                            selection either: each inner model selects its own L2 / epoch on a separate
                            patient-grouped 20 % split of its own training data)
The chosen (pi, alpha / b) are then applied to the predictions of the outer fold-k model on fold k.
No model involved in choosing fold k's calibration has seen fold k; no prediction used to fit a threshold
was made by a model trained on that sample.

For the LOCKED TEST the parameters are chosen on the development-set 5-fold OOF predictions (all repeats
pooled), pi from development labels; they are stored here and applied later to the final model trained on
the full development set (final_model.py). Test labels are never read.

rules: prior (p * pi^alpha) and bias (log p + b, b_Normal = 0, grid 0.25); objectives: 'accuracy' and
'protected' (max accuracy s.t. balanced accuracy and macro-F1 >= argmax - 0.01 on the fitting data).

    python -m research_experiments.phase_2.nested_calibration --model fusion_mlp --repeat 0
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.accuracy_tradeoff import ALPHAS, BIASES, decide, mets
from research_experiments.phase_2.cv_features import ALL, D, F6, FUS, H_NUM, LIN, MLPF, make_X, nD
from research_experiments.phase_2.p2common import PCOLS, RES, guard, s3
from research_experiments.phase_2.protocol import folds
from ecg_ef.metrics import evaluate

Y = D.label_idx.values
G = D.patient.values
MODELS = {   # name -> (spec, cfg, sklearn-LR?, OOF tag pattern {seed, repeat})
    "fusion_mlp": (FUS, MLPF, False),
    "ecgf_linear": (dict(blocks=[F6]), LIN, False),
    "header_numeric_lr": (dict(blocks=[H_NUM]), {}, True),
}


def oof_tag(model, s, r):
    base = {"fusion_mlp": "ff_hdrnum+ecgf_mlp", "ecgf_linear": "ecgf_aligned_f6_lin", "header_numeric_lr": "hdr_numeric_skLR"}[model]
    t = base if (s == 42 or model != "fusion_mlp") else f"{base}_seed{s}"
    return t + (f"_r{r}" if r else "")


def fit_predict(spec, cfg, sk, fit_rows, pred_rows, seed, split_seed):
    """Fit on fit_rows (own patient-grouped 80/20 split for model selection), predict pred_rows."""
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, Y[fit_rows], G[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    fm = np.zeros(len(ALL), bool); fm[tr] = True
    X, pre = make_X(spec, fm)
    if sk:
        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr], Y[tr])
            sc = s3(evaluate(Y[va], clf.predict_proba(X[va])))
            if best is None or sc > best[0]:
                best = (sc, clf)
        return best[1].predict_proba(X[pred_rows])
    f = p2heads.fit(X[tr], Y[tr], X[va], Y[va], dict(cfg, no_std=pre), seed=seed)
    return f["predict"](X[pred_rows])[0]


def choose(P, y, method, objective, prior):
    base = mets(y, P.argmax(1)); best = None
    for c in (ALPHAS if method == "prior" else BIASES):
        m = mets(y, decide(P, method, c, prior))
        if objective == "protected" and (m["balanced_accuracy"] < base["balanced_accuracy"] - 0.01 or m["macro_f1"] < base["macro_f1"] - 0.01):
            continue
        if best is None or m["accuracy"] > best[1] + 1e-12:
            best = (c, m["accuracy"])
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    spec, cfg, sk = MODELS[a.model]
    seeds = a.seeds if a.model == "fusion_mlp" else [42]
    r = a.repeat; t0 = time.time()
    f = folds(r); fo = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
    # outer predictions = existing OOF of the base runs (seed ensemble)
    P_out = np.mean([pd.read_csv(RES / oof_tag(a.model, s, r) / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values
                     for s in seeds], 0)
    rules = [(m, o) for m in ("prior", "bias") for o in ("accuracy", "protected")]
    pred = {ru: np.zeros(nD, int) for ru in rules}; params = {ru: [] for ru in rules}; inner_info = []
    for k in range(5):
        TR = np.flatnonzero(fo != k)
        prior_k = np.bincount(Y[TR], minlength=4) / len(TR)                       # (1) prior: training-portion labels
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k)
        P_in = np.zeros((len(TR), 4))
        for j, (ia, ib) in enumerate(inner.split(TR, Y[TR], G[TR])):
            assert not set(G[TR[ia]]) & set(G[TR[ib]])
            P_in[ib] = np.mean([fit_predict(spec, cfg, sk, TR[ia], TR[ib], s, 2000 + 10 * k + j) for s in seeds], 0)
        inner_info.append(dict(fold=k, inner_oof_accuracy=float(np.mean(P_in.argmax(1) == Y[TR]))))
        for ru in rules:                                                          # (2) thresholds: inner OOF only
            c = choose(P_in, Y[TR], ru[0], ru[1], prior_k)
            pred[ru][fo == k] = decide(P_out[fo == k], ru[0], c, prior_k)
            params[ru].append(float(c) if ru[0] == "prior" else np.asarray(c).round(2).tolist())
        print(f"fold {k} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = {"model": a.model, "repeat": r, "seeds": seeds, "inner": inner_info,
           "original": dict(**mets(Y, P_out.argmax(1)), confusion=confusion_matrix(Y, P_out.argmax(1), labels=range(4)).tolist())}
    for ru in rules:
        res[f"calibrated: {ru[0]}, {ru[1]}"] = dict(**mets(Y, pred[ru]), params_per_fold=params[ru],
                                                   confusion=confusion_matrix(Y, pred[ru], labels=range(4)).tolist())
    guard(RES).mkdir(exist_ok=True)
    (RES / f"nested_calibration_{a.model}_r{r}.json").write_text(json.dumps(res, indent=1))
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "f1_mild", "recall_mild", "qwk", "recall_Normal", "share_pred_normal"]
    print(pd.DataFrame({k: {m: v[m] for m in keys} for k, v in res.items() if isinstance(v, dict) and "accuracy" in v}).T.round(3).to_string())


if __name__ == "__main__":
    main()
