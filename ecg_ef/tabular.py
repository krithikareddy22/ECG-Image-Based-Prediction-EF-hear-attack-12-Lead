"""Gradient boosting (XGBoost) and logistic regression on engineered features, following
Ning et al. (arXiv 2604.25942): ECG waveform features (ecg_ef/ecg_features.py) and
"EHR-like" variables - here the PDF header's age, sex, machine measurements and
statements (ecg_ef/header.py) - alone and combined.

Same patient-grouped split as everything else. Hyper-parameters, the number of boosting
rounds (early stopping) and the screening threshold are all chosen on the VALIDATION set;
the test set is used once.

    python -m ecg_ef.ecg_features      # once
    python -m ecg_ef.tabular           # -> results/{xgb,logreg}_{waveform,header,waveform_header}/
"""
from __future__ import annotations

import itertools
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, HF_TARGET_SENSITIVITY, RESULTS_DIR
from .ecg_features import FEATURES_CSV
from .header import load_header_data
from .metrics import evaluate, pick_hf_threshold
from .train import plot_confusion, save_predictions, selection_score


def feature_sets():
    h, Xh, _, _, _, _ = load_header_data()            # header design matrix, train-median imputed
    w = pd.read_csv(FEATURES_CSV).set_index("ecg_id").loc[h.ecg_id].reset_index(drop=True)
    w = w.fillna(w[h.split.values == "train"].median())
    Xh = Xh.reset_index(drop=True)
    sets = {"waveform": w, "header": Xh,
            "waveform_header": pd.concat([w.add_prefix("ecg:"), Xh.add_prefix("hdr:")], axis=1)}
    return h.reset_index(drop=True), sets


def balanced_weights(y):
    counts = np.bincount(y, minlength=4)
    return (len(y) / (4 * counts))[y]


def fit_xgb(Xtr, ytr, Xva, yva, seed=0):
    """Small grid; rounds by early stopping on validation log-loss; config by val score."""
    best = None
    for depth, colsample, mcw in itertools.product((2, 3, 4), (0.3, 0.6), (1, 5)):
        clf = xgb.XGBClassifier(
            objective="multi:softprob", n_estimators=2000, learning_rate=0.03, max_depth=depth,
            subsample=0.8, colsample_bytree=colsample, min_child_weight=mcw, reg_lambda=1.0,
            eval_metric="mlogloss", early_stopping_rounds=100, tree_method="hist",
            n_jobs=4, random_state=seed)
        clf.fit(Xtr, ytr, sample_weight=balanced_weights(ytr), eval_set=[(Xva, yva)],
                sample_weight_eval_set=[balanced_weights(yva)], verbose=False)
        sc = selection_score(evaluate(yva, clf.predict_proba(Xva)))
        if best is None or sc > best[0]:
            best = (sc, clf, dict(max_depth=depth, colsample_bytree=colsample,
                                  min_child_weight=mcw, n_rounds=int(clf.best_iteration) + 1))
    return best


def fit_logreg(Xtr, ytr, Xva, yva):
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                                 max_iter=5000))
        clf.fit(Xtr, ytr)
        sc = selection_score(evaluate(yva, clf.predict_proba(Xva)))
        if best is None or sc > best[0]:
            best = (sc, clf, dict(C=C))
    return best


def shap_plot(clf, X: pd.DataFrame, path, top=20):
    """Mean |SHAP| over the test set and all classes (XGBoost's exact tree SHAP)."""
    contrib = clf.get_booster().predict(xgb.DMatrix(X.values, feature_names=list(X.columns)),
                                        pred_contribs=True)        # [n, classes, features+1]
    imp = pd.Series(np.abs(contrib[:, :, :-1]).mean((0, 1)), index=X.columns).nlargest(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(imp.index[::-1], imp.values[::-1], color="#2a78d6", height=0.62)
    ax.set_xlabel("mean |SHAP value| (test ECGs, all 4 classes)", fontsize=9)
    ax.set_title("Most influential features", loc="left", fontsize=10.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=0); ax.xaxis.grid(True, color="#e4e3df"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor="#fcfcfb"); plt.close(fig)
    return imp


def main():
    h, sets = feature_sets()
    split = h.split.values
    y = h.label_idx.values
    for name, X in sets.items():
        Xs = {s: X[split == s] for s in ("train", "val", "test")}
        ys = {s: y[split == s] for s in ("train", "val", "test")}
        for kind in ("xgb", "logreg"):
            t0 = time.time()
            fit = fit_xgb if kind == "xgb" else fit_logreg
            _, clf, params = fit(Xs["train"], ys["train"], Xs["val"], ys["val"])
            pva, pte = clf.predict_proba(Xs["val"]), clf.predict_proba(Xs["test"])
            thr = pick_hf_threshold(ys["val"], pva, HF_TARGET_SENSITIVITY)
            val_m, test_m = evaluate(ys["val"], pva, thr), evaluate(ys["test"], pte, thr)
            tag = f"{kind}_{name}"
            out = RESULTS_DIR / tag; out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(dict(
                tag=tag, model=kind, variant=name, pretrained=False, img_hw=[0, 0],
                params=None, n_features=X.shape[1], hyperparameters=params,
                train_minutes=round((time.time() - t0) / 60, 2), hf_threshold=thr,
                val=val_m, test=test_m), indent=2))
            plot_confusion(test_m["confusion_matrix"], f"{tag} - test (acc {test_m['accuracy']:.1%})",
                           out / "confusion_matrix.png")
            save_predictions(pva, h[split == "val"], out / "val_predictions.csv")
            save_predictions(pte, h[split == "test"], out / "test_predictions.csv")
            if kind == "xgb":
                imp = shap_plot(clf, Xs["test"], out / "shap_importance.png")
                imp.round(4).to_csv(out / "shap_importance.csv")
            print(f"[{tag:24s}] {X.shape[1]:3d} feats  val score {selection_score(val_m):.3f} | TEST "
                  f"acc {test_m['accuracy']:.3f}  bal {test_m['balanced_accuracy']:.3f}  "
                  f"F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}  "
                  f"HF-AUC {test_m['hf_screen']['auc']:.3f}  {params}", flush=True)


if __name__ == "__main__":
    main()
