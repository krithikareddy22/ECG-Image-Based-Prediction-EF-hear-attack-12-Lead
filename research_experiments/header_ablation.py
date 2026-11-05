"""Ablation of the printed ECG header: which part carries the signal?
Same recipe as the project's header model (ecg_ef/header.py): standardise + multinomial
logistic regression, class_weight='balanced', C in {0.01..1} chosen on VALIDATION by
mean(macro-F1, QWK, HF AUC); original split; missing values imputed with TRAIN medians.

    python -m research_experiments.header_ablation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.common import RES, load_meta, save_run, seed_all
from research_experiments.exp_feature_heads import header_matrix
from ecg_ef.metrics import evaluate
from ecg_ef.train import selection_score

DEMO = ["age", "male"]
MEAS = ["Rate", "RR", "PR", "QRSD", "QT", "QTcB", "QTcF", "PR_missing", "P_axis_sin", "P_axis_cos",
        "QRS_axis_sin", "QRS_axis_cos", "T_axis_sin", "T_axis_cos"]


def groups(cols):
    verdict = [c for c in cols if c.startswith("summary=")]
    stmts = [c for c in cols if c.startswith("stmt=")]
    return {
        "H1_demographics": DEMO,
        "H2_measurements": MEAS,
        "H3_demographics+measurements (no machine text)": DEMO + MEAS,
        "H4_verdict_only": verdict,
        "H5_statements_only": stmts,
        "H6_verdict+statements (machine text only)": verdict + stmts,
        "H7_all_minus_statements": DEMO + MEAS + verdict,
        "H8_all (= baseline header model)": DEMO + MEAS + verdict + stmts,
    }


def fit_lr(X, y, sp):
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        clf.fit(X[sp == "train"], y[sp == "train"])
        sc = selection_score(evaluate(y[sp == "val"], clf.predict_proba(X[sp == "val"])))
        if best is None or sc > best[0]:
            best = (sc, C, clf)
    return best


def main():
    seed_all()
    meta = load_meta()
    Xh, _ = header_matrix(meta)
    sp, y = meta.split.values, meta.label_idx.values
    mv = meta[sp == "val"].reset_index(drop=True); mt = meta[sp == "test"].reset_index(drop=True)
    rows = []
    for name, cols in groups(list(Xh.columns)).items():
        X = Xh[cols].values
        sc, C, clf = fit_lr(X, y, sp)
        tag = "HA_" + name.split(" ")[0]
        s = save_run(tag, mv, clf.predict_proba(X[sp == "val"]), mt, clf.predict_proba(X[sp == "test"]),
                     dict(input=f"header subset: {name}", n_features=len(cols), C=C, val_selection_score=sc))
        t = s["test"]
        rows.append(dict(subset=name, n_features=len(cols), C=C, val_score=sc, acc=t["accuracy"],
                         bal_acc=t["balanced_accuracy"], macro_f1=t["macro_f1"], qwk=t["qwk"],
                         mild_f1=t["per_class"]["Mild"]["f1"], macro_auroc=t["auroc_ovr"]["macro"],
                         hf_auc=t["hf_screen"]["auc"],
                         bal_acc_ci=t["ci95"]["balanced_accuracy"]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "header_ablation.csv", index=False)
    print(df.drop(columns="bal_acc_ci").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
