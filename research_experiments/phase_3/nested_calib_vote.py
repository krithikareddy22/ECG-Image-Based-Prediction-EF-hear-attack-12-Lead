"""Fully nested calibration for the equal-weight VOTE ensemble (2-way MLP fusion [3 seeds] + ECGFounder linear with
distance penalty + header-numeric+clinical LR). Same rules / objectives / leakage control as
phase_2/nested_calibration.py: class prior from the training portion of each outer fold; prior-exponent and
per-class thresholds chosen on inner out-of-fold predictions of members re-fitted inside that training portion.
    python -m research_experiments.phase_3.nested_calib_vote --repeat 0
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.accuracy_tradeoff import decide, mets
from research_experiments.phase_2.nested_calibration import choose
from research_experiments.phase_2.p2common import PCOLS, s3
from research_experiments.phase_2.protocol import folds
from research_experiments.phase_3 import exp_p3  # registers the 'clin' block
from research_experiments.phase_3.p3common import ALL, D, RES3, make_X, nD
from ecg_ef.metrics import evaluate

Y, G = D.label_idx.values, D.patient.values
MEMBERS = [  # (spec, cfg or None for sklearn LR, seeds, phase-2/3 OOF tag pattern)
    (dict(blocks=[exp_p3.H_NUM, exp_p3.F6]), exp_p3.MLPF, (42, 43, 44)),
    (dict(blocks=[exp_p3.F6]), dict(exp_p3.LIN, loss="ce_dist", lam=1.0), (42,)),
    (dict(blocks=[exp_p3.H_NUM, exp_p3.CLIN]), None, (42,)),
]
OOF_TAGS = lambda r: [[t + (f"_r{r}" if r else "") for t in ("ff_hdrnum+ecgf_mlp", "ff_hdrnum+ecgf_mlp_seed43", "ff_hdrnum+ecgf_mlp_seed44")],
                      ["ecgf_f6_lin_cedist1" + (f"_r{r}" if r else "")], ["clin_hdrnum+clin_skLR" + (f"_r{r}" if r else "")]]


def load_oof(tag):
    from research_experiments.phase_3.p3_robust import find
    return pd.read_csv(find(tag) / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values


def fit_member(spec, cfg, seed, fit_rows, pred_rows, split_seed):
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, Y[fit_rows], G[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    fm = np.zeros(len(ALL), bool); fm[tr] = True
    X, pre = make_X(spec, fm)
    if cfg is None:
        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr], Y[tr])
            sc = s3(evaluate(Y[va], clf.predict_proba(X[va])))
            if best is None or sc > best[0]:
                best = (sc, clf)
        return best[1].predict_proba(X[pred_rows])
    return p2heads.fit(X[tr], Y[tr], X[va], Y[va], dict(cfg, no_std=pre), seed=seed)["predict"](X[pred_rows])[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeat", type=int, default=0); a = ap.parse_args(); r = a.repeat
    f = folds(r); fo = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
    P_out = np.mean([np.mean([load_oof(t) for t in ts], 0) for ts in OOF_TAGS(r)], 0)   # equal-weight vote of members
    rules = [(m, o) for m in ("prior", "bias") for o in ("accuracy", "protected")]
    pred = {ru: np.zeros(nD, int) for ru in rules}; params = {ru: [] for ru in rules}
    for k in range(5):
        TR = np.flatnonzero(fo != k); prior_k = np.bincount(Y[TR], minlength=4) / len(TR)
        P_in = np.zeros((len(TR), 4))
        for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], G[TR])):
            mem = [np.mean([fit_member(sp, cf, s, TR[ia], TR[ib], 2000 + 10 * k + j) for s in seeds], 0) for sp, cf, seeds in MEMBERS]
            P_in[ib] = np.mean(mem, 0)
        for ru in rules:
            c = choose(P_in, Y[TR], ru[0], ru[1], prior_k)
            pred[ru][fo == k] = decide(P_out[fo == k], ru[0], c, prior_k)
            params[ru].append(float(c) if ru[0] == "prior" else np.asarray(c).round(2).tolist())
        print(f"fold {k} done", flush=True)
    res = {"model": "vote(mlp2 + ecgf_lin_dist + hdr_clin_lr)", "repeat": r,
           "original": dict(**mets(Y, P_out.argmax(1)), confusion=confusion_matrix(Y, P_out.argmax(1), labels=range(4)).tolist())}
    for ru in rules:
        res[f"calibrated: {ru[0]}, {ru[1]}"] = dict(**mets(Y, pred[ru]), params_per_fold=params[ru],
                                                   confusion=confusion_matrix(Y, pred[ru], labels=range(4)).tolist())
    (RES3 / f"nested_calibration_vote_r{r}.json").write_text(json.dumps(res, indent=1))
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "f1_mild", "recall_mild", "qwk", "recall_Normal", "share_pred_normal"]
    print(pd.DataFrame({k: {m: v[m] for m in keys} for k, v in res.items() if isinstance(v, dict) and "accuracy" in v}).T.round(3).to_string())


if __name__ == "__main__":
    main()
