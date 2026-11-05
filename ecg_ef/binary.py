"""The "further treatment" (YES/NO) decision: reduced EF (Mild/Moderate/Severe) vs Normal.

Compares (all choices made on the VALIDATION set, test used once):
  * how the YES/NO cut-off is chosen:
      sens80   highest cut-off that still catches >= 80 % of reduced-EF ECGs (current)
      youden   maximises sensitivity + specificity
      max_acc  maximises accuracy
  * models trained directly for YES/NO (instead of deriving it from the 4-class model):
      header logistic regression, waveform-feature logistic regression / XGBoost, and
      their fusion with the image CNN's score.

    python -m ecg_ef.binary      # -> results/binary_comparison.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RESULTS_DIR
from .ensemble import load_predictions
from .tabular import feature_sets


def thresholds(y, s):
    """Cut-offs chosen on validation by the three rules."""
    cand = np.unique(s)
    sens = np.array([(s[y] >= c).mean() for c in cand])
    spec = np.array([(s[~y] < c).mean() for c in cand])
    acc = np.array([((s >= c) == y).mean() for c in cand])
    ok = np.flatnonzero(sens >= 0.80)
    return {"sens80": cand[ok.max()], "youden": cand[np.argmax(sens + spec)],
            "max_acc": cand[np.argmax(acc)]}


def at(y, s, c):
    yes = s >= c
    return dict(acc=(yes == y).mean(), sens=yes[y].mean(), spec=(~yes[~y]).mean())


def fit_lr(X, y, split):
    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                                 max_iter=5000))
        clf.fit(X[split == "train"], y[split == "train"])
        auc = roc_auc_score(y[split == "val"], clf.predict_proba(X[split == "val"])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, clf)
    return {s: best[1].predict_proba(X[split == s])[:, 1] for s in ("val", "test")}


def fit_xgb(X, y, split):
    best = None
    w = np.where(y[split == "train"], 1.0, y[split == "train"].mean() / (1 - y[split == "train"].mean()))
    for depth in (2, 3):
        clf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=depth, subsample=0.8,
                                colsample_bytree=0.3, eval_metric="auc", early_stopping_rounds=100,
                                tree_method="hist", n_jobs=4, random_state=0)
        clf.fit(X[split == "train"], y[split == "train"], sample_weight=w,
                eval_set=[(X[split == "val"], y[split == "val"])], verbose=False)
        auc = roc_auc_score(y[split == "val"], clf.predict_proba(X[split == "val"])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, clf)
    return {s: best[1].predict_proba(X[split == s])[:, 1] for s in ("val", "test")}


def main():
    h, sets = feature_sets()
    split = h.split.values
    y = (h.label_idx.values > 0)
    yv, yt = y[split == "val"], y[split == "test"]
    ids = {s: h.ecg_id.values[split == s] for s in ("val", "test")}

    def from_4class(tag):           # P(reduced EF) = 1 - P(Normal), aligned to h's order
        out = {}
        for s in ("val", "test"):
            _, p, e = load_predictions(tag, s)
            out[s] = pd.Series(1 - p[:, 0], index=e).loc[ids[s]].values
        return out

    scores = {
        "Default: 4-class image + header fusion": from_4class("fusion_header_resnet18"),
        "4-class image CNN (ResNet-18)": from_4class("resnet18_filtered"),
        "4-class header model": from_4class("header_logreg"),
        "YES/NO header logistic regression": fit_lr(sets["header"].values, y, split),
        "YES/NO waveform-feature logistic regression": fit_lr(sets["waveform"].values, y, split),
        "YES/NO waveform-feature XGBoost": fit_xgb(sets["waveform"].values, y, split),
    }
    img = scores["4-class image CNN (ResNet-18)"]
    for name in ("YES/NO header logistic regression", "YES/NO waveform-feature XGBoost"):
        base = scores[name]
        w = max(np.linspace(0, 1, 11), key=lambda a: roc_auc_score(yv, a * base["val"] + (1 - a) * img["val"]))
        scores[f"{name} + image CNN (weight {w:.1f})"] = {
            s: w * base[s] + (1 - w) * img[s] for s in ("val", "test")}
    hdr, wav = scores["YES/NO header logistic regression"], scores["YES/NO waveform-feature XGBoost"]
    best3 = max(((a, b) for a in np.linspace(0, 1, 11) for b in np.linspace(0, 1, 11) if a + b <= 1),
                key=lambda ab: roc_auc_score(yv, ab[0] * hdr["val"] + ab[1] * wav["val"]
                                             + (1 - sum(ab)) * img["val"]))
    a, b = best3
    scores[f"YES/NO header {a:.1f} + waveform XGBoost {b:.1f} + image CNN {1 - a - b:.1f}"] = {
        s: a * hdr[s] + b * wav[s] + (1 - a - b) * img[s] for s in ("val", "test")}

    rows = []
    for name, sc in scores.items():
        thr = thresholds(yv, sc["val"])
        r = {"Model": name, "Val AUC": roc_auc_score(yv, sc["val"]), "Test AUC": roc_auc_score(yt, sc["test"])}
        for rule, c in thr.items():
            m = at(yt, sc["test"], c)
            r[f"{rule} acc"], r[f"{rule} sens"], r[f"{rule} spec"] = m["acc"], m["sens"], m["spec"]
        rows.append(r)
    df = pd.DataFrame(rows)
    md = df.copy()
    for c in md.columns[1:]:
        md[c] = md[c].map(lambda x: f"{x:.3f}")
    text = ("# Further-treatment (YES/NO) decision: test set, n = 210\n\n"
            "Cut-off rules chosen on validation: `sens80` = catch >= 80 % of reduced-EF ECGs, "
            "`youden` = max sensitivity + specificity, `max_acc` = max accuracy.\n\n"
            + md.to_markdown(index=False) + "\n")
    (RESULTS_DIR / "binary_comparison.md").write_text(text)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
