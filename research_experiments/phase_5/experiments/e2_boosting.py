"""E2: gradient-boosted trees on engineered features. Hypothesis: XGBoost / LightGBM on header + engineered ECG features
(optionally + 32 in-fold PCA components of ECGFounder) beats the linear / MLP models.
Depth / leaves chosen on inner-val S3; early stopping on inner-val log-loss; class-balanced sample weights.
Baselines: header + clinical LR (P5A-import-hdrclinlr-v1), MLP fusion (P5A-import-mlpfusion-v1).
    python -m research_experiments.phase_5.experiments.e2_boosting
"""
import numpy as np

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_5.experiments.runner import cv_run
from research_experiments.phase_5.registry_ext import E2_SETS


def bal_w(y):
    c = np.bincount(y, minlength=4).astype(float); return (c.sum() / (4 * c))[y]


def xgb_fit(Xtr, ytr, Xva, yva, seed):
    import xgboost as xgb
    best = None
    for depth in (2, 3, 4):
        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=depth, subsample=0.8, colsample_bytree=0.5,
                              min_child_weight=2, reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss",
                              early_stopping_rounds=50, random_state=seed, n_jobs=1, tree_method="hist")
        m.fit(Xtr, ytr, sample_weight=bal_w(ytr), eval_set=[(Xva, yva)], verbose=False)
        sc = s3(evaluate(yva, m.predict_proba(Xva)))
        if best is None or sc > best[0]:
            best = (sc, depth, m)
    return best[2].predict_proba, dict(depth=best[1], best_iteration=int(best[2].best_iteration), val_s3=best[0])


def lgbm_fit(Xtr, ytr, Xva, yva, seed):
    import lightgbm as lgb
    best = None
    for leaves in (4, 8, 16):
        m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=leaves, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, min_child_samples=10, reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=bal_w(ytr), eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        sc = s3(evaluate(yva, m.predict_proba(Xva)))
        if best is None or sc > best[0]:
            best = (sc, leaves, m)
    return best[2].predict_proba, dict(num_leaves=best[1], best_iteration=int(best[2].best_iteration_ or 0), val_s3=best[0])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    J = [(mdl, f, st, blocks, r) for mdl, f in (("xgb", xgb_fit), ("lgbm", lgbm_fit)) for st, blocks in E2_SETS.items() for r in (0, 1, 2)]
    for mdl, f, st, blocks, r in J[::-1][a.part::a.jobs] if a.jobs > 1 else J:      # extra workers run from the end
        if True:
            if True:
                cv_run(f"P5D-E2-{mdl}-{st}-v1", blocks, "fitter", 42, r, fitter=f, command="python -m research_experiments.phase_5.experiments.e2_boosting",
                       extra_cfg=dict(model=mdl, feature_set=st))


if __name__ == "__main__":
    main()
