"""Complementary ensembles on development-set OOF predictions, per CV repeat (cross-fitted; locked test unscored).

Members are seed-ensembled first (probability mean over seeds within a repeat). Methods:
  vote   equal-weight probability mean
  wavg   simplex weights (step 0.1), chosen for outer fold k on the OOF of the other 4 folds (S3)
  stack  multinomial LR on member log-probabilities, trained for fold k on the other 4 folds' OOF only,
         C chosen by an inner 4-fold patient-grouped CV on those rows
Saved as phase-3 runs (oof.csv / info.json / cv_metrics.json, locked-test probabilities stored).
    python -m research_experiments.phase_3.p3_ensembles
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.phase_2.p2common import PCOLS, s3
from research_experiments.phase_3.p3_robust import find, fold_of, seeds_repeats
from research_experiments.phase_3.p3common import LOCK3, cv_report3, save_cv3
from research_experiments.phase_2.protocol import dev_and_test
from ecg_ef.metrics import evaluate

META, D, T = dev_and_test()
Y, G = D.label_idx.values, D.patient.values
P2LOCK = LOCK3.parent.parent / "phase_2" / "locked_test"

MEMBERS = {   # name -> tag list over seeds x repeats
    "mlp2": seeds_repeats("ff_hdrnum+ecgf_mlp"),
    "mlp3": seeds_repeats("clin_mlpfusion_hdr+clin+ecgf"),
    "gated": seeds_repeats("gate_scalar_hdr+clin+ecgf_dist"),
    "ecgf_lin_dist": ["ecgf_f6_lin_cedist1", "ecgf_f6_lin_cedist1_r1", "ecgf_f6_lin_cedist1_r2"],
    "hdr_clin_lr": ["clin_hdrnum+clin_skLR", "clin_hdrnum+clin_skLR_r1", "clin_hdrnum+clin_skLR_r2"],
    "image_cnn": ["img_resnet18_baseline_recipe"],
    "hdr_all_lr": ["hdr_all_skLR", "hdr_all_skLR_r1", "hdr_all_skLR_r2"],
}
COMBOS = {
    "ens_mlp2+ecgflin+hdrclin": ["mlp2", "ecgf_lin_dist", "hdr_clin_lr"],
    "ens_mlp2+mlp3+gated": ["mlp2", "mlp3", "gated"],
    "ens_all_feature_models": ["mlp2", "mlp3", "gated", "ecgf_lin_dist", "hdr_clin_lr"],
    "ens_ecgflin+hdrclin": ["ecgf_lin_dist", "hdr_clin_lr"],
    "ens_mlp2+image": ["mlp2", "image_cnn"],
    "ens_all+image": ["mlp2", "mlp3", "gated", "ecgf_lin_dist", "hdr_clin_lr", "image_cnn"],
}


def member_probs(name, r):
    """Seed-mean OOF and locked-test probabilities of a member for repeat r (None if missing)."""
    Ps, PTs = [], []
    for t in MEMBERS[name]:
        d = find(t)
        if d is None:
            continue
        if json.loads((d / "info.json").read_text()).get("repeat", 0) != r:
            continue
        Ps.append(pd.read_csv(d / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values)
        lk = (LOCK3 / f"{t}.csv") if (LOCK3 / f"{t}.csv").exists() else (P2LOCK / f"{t}.csv")
        PTs.append(pd.read_csv(lk).set_index("ecg_id").loc[T.ecg_id][PCOLS].values)
    return (np.mean(Ps, 0), np.mean(PTs, 0)) if Ps else (None, None)


def wgrid(m):
    g = np.round(np.arange(0, 1.0001, 0.1), 1)
    return [np.array(list(w) + [round(1 - sum(w), 1)]) for w in itertools.product(g, repeat=m - 1) if sum(w) <= 1 + 1e-9]


def stack_fit(X, y, groups, seed):
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, y, groups))
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        P = np.zeros((len(y), 4))
        for a, b in inner:
            P[b] = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[a], y[a]).predict_proba(X[b])
        sc = s3(evaluate(y, P))
        if best is None or sc > best[0]:
            best = (sc, C)
    return make_pipeline(StandardScaler(), LogisticRegression(C=best[1], class_weight="balanced", max_iter=5000)).fit(X, y)


def main():
    import sys
    only = sys.argv[sys.argv.index('--only') + 1:] if '--only' in sys.argv else None
    for cname, mem in COMBOS.items():
        if only and cname not in only:
            continue
        for r in (0, 1, 2):
            got = [member_probs(m, r) for m in mem]
            if any(p is None for p, _ in got):
                continue
            O = [p for p, _ in got]; TT = [t for _, t in got]; fo = fold_of(r)
            for method in ("vote", "wavg", "stack"):
                P = np.zeros((len(Y), 4)); params = {}
                if method == "vote":
                    P = np.mean(O, 0); PT = np.mean(TT, 0)
                elif method == "wavg":
                    W = wgrid(len(O))
                    best = lambda mask: max(W, key=lambda w: s3(evaluate(Y[mask], sum(wi * o[mask] for wi, o in zip(w, O)))))
                    for k in range(5):
                        w = best(fo != k); P[fo == k] = sum(wi * o[fo == k] for wi, o in zip(w, O)); params[f"fold{k}"] = w.tolist()
                    w = best(np.ones(len(Y), bool)); PT = sum(wi * t for wi, t in zip(w, TT)); params["final"] = w.tolist()
                else:
                    L = np.concatenate([np.log(np.clip(o, 1e-6, 1)) for o in O], 1)
                    LT = np.concatenate([np.log(np.clip(t, 1e-6, 1)) for t in TT], 1)
                    for k in range(5):
                        tr, te = fo != k, fo == k
                        P[te] = stack_fit(L[tr], Y[tr], G[tr], seed=k).predict_proba(L[te])
                    PT = stack_fit(L, Y, G, seed=99).predict_proba(LT)
                tag = f"{cname}__{method}" + (f"_r{r}" if r else "")
                save_cv3(tag, P, P.argmax(1), PT, dict(members=mem, method=method, params=params, seed=-1), r)
                cv_report3(tag, P, P.argmax(1), r)


if __name__ == "__main__":
    main()
