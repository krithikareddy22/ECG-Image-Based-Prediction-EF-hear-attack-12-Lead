"""Late fusion, stacking and decision-rule calibration on OUT-OF-FOLD predictions (phase-2 protocol).

Every post-hoc parameter used for outer fold k is fitted on the OOF predictions of the other four folds
only (cross-fitting); stackers are trained only on OOF predictions (never on predictions made for
samples a base model was trained on). Locked-test predictions use parameters fitted on all D OOF.

methods:  vote (equal weights) | wavg (simplex grid, step 0.1, S3) | stack_lr | stack_lr_hdr |
          stack_xgb_hdr | stack_mlp_hdr
decision rules (post-hoc, cross-fitted):  argmax | prior (p * pi^alpha) | bias (per-class log offsets)
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_2.cv_features import ALL, D, T, header_block, nD
from research_experiments.phase_2.p2common import LOCK, PCOLS, RES, cv_report, fold_vector, s3, save_cv, short
from ecg_ef.metrics import evaluate

Y = D.label_idx.values
FOLD = fold_vector(D)
PRIOR = np.bincount(Y, minlength=4) / len(Y)


def _load(tag):
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values
    t = pd.read_csv(LOCK / f"{tag}.csv").set_index("ecg_id").loc[T.ecg_id][PCOLS].values
    return o, t


def _metrics(y, p, pred=None):
    if pred is None:
        return evaluate(y, p)
    return evaluate(y, np.eye(4)[pred] * 0.97 + 0.01)


def _wgrid(m, step=0.1):
    g = np.round(np.arange(0, 1 + 1e-9, step), 2)
    return [np.array(list(w) + [round(1 - sum(w), 2)]) for w in itertools.product(g, repeat=m - 1) if sum(w) <= 1 + 1e-9]


def logit(P):
    return np.log(np.clip(P, 1e-6, 1))


def _stack_features(Ps, idx_rows, hdr_fit_mask=None, use_hdr=False):
    X = np.concatenate([logit(P) for P in Ps], 1)
    if use_hdr:
        H = header_block(hdr_fit_mask, "numeric")
        X = np.concatenate([X, H[idx_rows]], 1)
    return X


def _fit_stacker(kind, Xtr, ytr, groups, seed=0):
    """Returns predict_proba fn. Hyper-parameter chosen by an inner 4-fold patient-grouped CV on Xtr."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(Xtr, ytr, groups))
    if kind == "lr":
        grid = [dict(C=C) for C in (0.01, 0.03, 0.1, 0.3, 1.0)]
        make = lambda h: make_pipeline(StandardScaler(), LogisticRegression(C=h["C"], class_weight="balanced", max_iter=5000))
    elif kind == "xgb":
        import xgboost as xgb
        grid = [dict(max_depth=d, n_estimators=n) for d in (2, 3) for n in (100, 300)]
        sw_fn = lambda y: (1.0 / np.bincount(y, minlength=4))[y] * len(y) / 4
        make = lambda h: xgb.XGBClassifier(max_depth=h["max_depth"], n_estimators=h["n_estimators"], learning_rate=0.03,
                                           subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0,
                                           tree_method="hist", n_jobs=1, random_state=seed, eval_metric="mlogloss")
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        grid = [dict(alpha=a) for a in (1e-2, 1e-1, 1.0)]
        make = lambda h: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=h["alpha"],
                                                                       max_iter=2000, early_stopping=False, random_state=seed))
    best = None
    for h in grid:
        P = np.zeros((len(ytr), 4))
        for a, b in inner:
            mdl = make(h)
            if kind == "xgb":
                mdl.fit(Xtr[a], ytr[a], sample_weight=sw_fn(ytr[a]))
            else:
                mdl.fit(Xtr[a], ytr[a])
            P[b] = mdl.predict_proba(Xtr[b])
        sc = s3(evaluate(ytr, P))
        if best is None or sc > best[0]:
            best = (sc, h)
    mdl = make(best[1])
    if kind == "xgb":
        mdl.fit(Xtr, ytr, sample_weight=sw_fn(ytr))
    else:
        mdl.fit(Xtr, ytr)
    return mdl.predict_proba, best[1]


def combine(method, tags, tag_out=None):
    loaded = [_load(t) for t in tags]
    O = [l[0] for l in loaded]; TT = [l[1] for l in loaded]
    P = np.zeros((nD, 4)); params = {}
    if method == "vote":
        P = np.mean(O, 0); PT = np.mean(TT, 0); params = {"weights": [1 / len(O)] * len(O)}
    elif method == "wavg":
        W = _wgrid(len(O))
        def best_w(mask):
            return max(W, key=lambda w: s3(evaluate(Y[mask], sum(wi * o[mask] for wi, o in zip(w, O)))))
        for k in range(5):
            w = best_w(FOLD != k); P[FOLD == k] = sum(wi * o[FOLD == k] for wi, o in zip(w, O)); params[f"fold{k}"] = w.tolist()
        w = best_w(np.ones(nD, bool)); PT = sum(wi * t for wi, t in zip(w, TT)); params["final"] = w.tolist()
    else:
        kind = {"stack_lr": "lr", "stack_lr_hdr": "lr", "stack_xgb_hdr": "xgb", "stack_mlp_hdr": "mlp"}[method]
        use_hdr = method.endswith("_hdr")
        for k in range(5):
            tr = np.flatnonzero(FOLD != k); te = np.flatnonzero(FOLD == k)
            fm = np.zeros(len(ALL), bool); fm[tr] = True
            Xtr = _stack_features([o[tr] for o in O], tr, fm, use_hdr)
            Xte = _stack_features([o[te] for o in O], te, fm, use_hdr)
            f, h = _fit_stacker(kind, Xtr, Y[tr], D.patient.values[tr], seed=k); P[te] = f(Xte); params[f"fold{k}"] = h
        fm = np.zeros(len(ALL), bool); fm[:nD] = True
        rowsT = np.arange(nD, len(ALL))
        f, h = _fit_stacker(kind, _stack_features(O, np.arange(nD), fm, use_hdr), Y, D.patient.values, seed=99)
        PT = f(_stack_features(TT, rowsT, fm, use_hdr)); params["final"] = h
    tag_out = tag_out or f"{method}__" + "+".join(tags)
    return tag_out, P, PT, params


def decision(P, PT, rule, objective="s3"):
    """Cross-fitted post-hoc decision rule. Returns (pred_oof, pred_T, params)."""
    if rule == "argmax":
        return P.argmax(1), PT.argmax(1), {}
    if rule == "prior":
        cands = [dict(alpha=a) for a in np.round(np.arange(0, 1.51, 0.1), 2)]
        # the class prior is estimated from the fitting rows only (other folds / development set), see below
        apply = lambda p, c: (p * c["prior"][None] ** c["alpha"]).argmax(1)
    else:
        offs = np.arange(-1.0, 1.01, 0.25)
        cands = [dict(b=[0.0, a, b, c]) for a in offs for b in offs for c in offs]
        apply = lambda p, c: (np.log(np.clip(p, 1e-9, 1)) + np.array(c["b"])[None]).argmax(1)
    def score(y, pred):
        m = _metrics(y, None, pred) if False else evaluate(y, np.eye(4)[pred] * 0.97 + 0.01)
        return s3(m) if objective == "s3" else m["accuracy"]
    pred = np.zeros(nD, int); params = {}
    for k in range(5):
        tr = FOLD != k
        cs = [dict(c, prior=np.bincount(Y[tr], minlength=4) / tr.sum()) for c in cands] if rule == "prior" else cands
        c = max(cs, key=lambda c: score(Y[tr], apply(P[tr], c)))
        pred[FOLD == k] = apply(P[FOLD == k], c); params[f"fold{k}"] = {k_: (v.tolist() if hasattr(v, "tolist") else v) for k_, v in c.items()}
    cs = [dict(c, prior=PRIOR) for c in cands] if rule == "prior" else cands     # development-set prior only
    c = max(cs, key=lambda c: score(Y, apply(P, c))); params["final"] = {k_: (v.tolist() if hasattr(v, "tolist") else v) for k_, v in c.items()}
    return pred, apply(PT, c), params


def run(method, tags, rule="argmax", objective="s3", name=None):
    tag, P, PT, params = combine(method, tags)
    name = name or tag
    if rule != "argmax":
        name = f"{name}__{rule}_{objective}"
    pred, predT, dparams = decision(P, PT, rule, objective)
    oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", FOLD); oof["pred"] = pred
    locked = pd.DataFrame(PT, columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values); locked["pred"] = predT
    save_cv(name, oof, locked, dict(method=method, members=tags, rule=rule, objective=objective, params=params,
                                    decision_params=dparams), overwrite=True)
    m = cv_report(name, D, P, pred)
    print(f"[{name}] {short(m)}", flush=True)
    return m
