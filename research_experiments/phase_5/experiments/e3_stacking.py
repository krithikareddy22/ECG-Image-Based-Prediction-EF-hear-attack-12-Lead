"""E3: regularised stacking trained ONLY on training-portion out-of-fold predictions (nested; never in-sample).
Members = the vote-ensemble members (MLP fusion seed-mean, ECGF linear + distance penalty, header + clinical LR).
For outer fold k the meta-learner is fitted on the nested inner-OOF predictions of the training portion
(experiments/nested.py) and applied to the members' outer-fold predictions. Hypothesis: a learned combination beats the
equal-weight vote (C1).  Variants: lr-balanced (class-balanced multinomial LR on member log-probabilities, C by grouped
4-fold CV on the inner OOF, S3), lr-unweighted (no class weights, C by accuracy), wavg-acc (simplex weights, step 0.1,
maximising inner-OOF accuracy).
    python -m research_experiments.phase_5.experiments.e3_stacking
"""
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_5.experiments.nested import inner_oof
from research_experiments.phase_5.p5common import PAT, SEEDS, Y, done, fold_vec, load_run, mark_running, nD, save_run

MEMBERS = {"mlpfusion": [f"P5A-import-mlpfusion-v1__s{s}" for s in SEEDS], "ecgflindist": ["P5A-import-ecgflindist-v1__s0"],
           "hdrclinlr": ["P5A-import-hdrclinlr-v1__s0"]}
CMD = "python -m research_experiments.phase_5.experiments.e3_stacking"


def outer_members(r):
    return [np.mean([load_run(f"{p}_r{r}")[0] for p in pre], 0) for pre in MEMBERS.values()]


def feats(Ps):
    return np.concatenate([np.log(np.clip(P, 1e-6, 1)) for P in Ps], 1)


def fit_meta(Z, y, groups, variant, seed=0):
    cw = "balanced" if variant == "lr-balanced" else None
    crit = (lambda yy, P: s3(evaluate(yy, P))) if variant == "lr-balanced" else (lambda yy, P: float(np.mean(P.argmax(1) == yy)))
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(Z, y, groups))
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        P = np.zeros((len(y), 4))
        for a, b in folds:
            P[b] = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=cw, max_iter=5000)).fit(Z[a], y[a]).predict_proba(Z[b])
        sc = crit(y, P)
        if best is None or sc > best[0] + 1e-12:
            best = (sc, C)
    return make_pipeline(StandardScaler(), LogisticRegression(C=best[1], class_weight=cw, max_iter=5000)).fit(Z, y), best[1]


def main():
    W = [np.array(w) for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3) if abs(sum(w) - 1) < 1e-9]
    for r in (0, 1, 2):
        out = outer_members(r); fo = fold_vec(r)
        for v in ("lr-balanced", "lr-unweighted", "wavg-acc"):
            rid = f"P5D-E3-stack-{v}-v1__s0_r{r}"
            if done(rid):
                continue
            mark_running(rid, CMD)
            P = np.zeros((nD, 4)); params = []
            for k in range(5):
                inn = [inner_oof(b, r, k) for b in MEMBERS]
                TR = inn[0][0]; Pin = [x[1] for x in inn]
                te = fo == k
                if v == "wavg-acc":
                    acc = [np.mean(sum(w_ * p for w_, p in zip(w, Pin)).argmax(1) == Y[TR]) for w in W]
                    w = W[int(np.argmax(acc))]; params.append(w.tolist())
                    P[te] = sum(w_ * p[te] for w_, p in zip(w, out))
                else:
                    meta, C = fit_meta(feats(Pin), Y[TR], PAT[TR], v, seed=3000 + 10 * r + k); params.append(C)
                    P[te] = meta.predict_proba(feats([p[te] for p in out]))
            agg = save_run(rid, P, P.argmax(1), r, dict(kind="nested stacking", variant=v, members=list(MEMBERS), baseline="P5A-import-vote-v1"),
                           extra=dict(params_per_fold=params))
            m = agg["metrics"]; print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} minR {m['min_recall']:.2f} {params}", flush=True)


if __name__ == "__main__":
    main()
