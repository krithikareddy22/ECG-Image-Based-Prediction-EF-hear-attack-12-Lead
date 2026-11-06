"""Secondary analysis: patient-grouped, stratified 5-fold cross-validation over all 1,047 ECGs
for the models that are cheap enough on CPU (header models, frozen-ECGFounder heads, their
fusions). The CNN / fine-tuned models are NOT cross-validated (CPU cost) - documented.

Leakage control per outer fold: patients never cross folds (StratifiedGroupKFold on the
anonymous patient hash); an inner patient-grouped split of the outer-training part gives the
validation set for C / L2 / fusion-weight selection; the header statement vocabulary and the
imputation medians are rebuilt from the outer-training part only; ECGFounder features are
label-free (pre-trained), standardised with training statistics.

    python -m research_experiments.cv_grouped
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.common import MOD, RES, ROOT, _stat_fns, full_metrics, load_meta, paired_bootstrap, seed_all
from research_experiments.header_ablation import DEMO, MEAS
from research_experiments.heads import fit_head
from ecg_ef.header import design_matrix
from ecg_ef.metrics import evaluate
from ecg_ef.train import selection_score


def header_fold(h, tr_mask, numbers_only=False):
    tr = h[tr_mask]
    counts = pd.Series([s for L in tr.statements.fillna("").str.split("|") for s in L if s]).value_counts()
    vocab = sorted(counts[counts >= 5].index)
    X = design_matrix(h, vocab)
    if numbers_only:
        X = X[DEMO + MEAS]
    return X.fillna(X[tr_mask].median()).values


def lr_fit(X, y, itr, iva):
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        clf.fit(X[itr], y[itr])
        sc = selection_score(evaluate(y[iva], clf.predict_proba(X[iva])))
        if best is None or sc > best[0]:
            best = (sc, clf)
    return best[1].predict_proba


def fuse_w(y_va, pa, pb):
    grid = np.round(np.linspace(0, 1, 11), 2)
    return max(grid, key=lambda w: selection_score(evaluate(y_va, w * pa + (1 - w) * pb)))


def main():
    seed_all()
    meta = load_meta()
    h = pd.read_csv(ROOT / "data" / "header_features.csv")
    h = meta[["ecg_id"]].merge(h, on="ecg_id", how="left")
    z = np.load(MOD / "features_ecgfounder_aligned.npz")
    assert (z["ecg_id"] == meta.ecg_id.values).all()
    F = z["feat"]
    y, g = meta.label_idx.values, meta.patient.values
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    names = ["header_LR", "headerNumbersOnly_LR", "header_cumlink", "ecgfA1_multinomial", "ecgfA1_cumlink",
             "FU_header+ecgfA1", "FU_headerNumbersOnly+ecgfA1", "FU_ordinal_header+ecgfA1"]
    oof = {n: np.zeros((len(y), 4)) for n in names}
    oof_pred = {n: np.zeros(len(y), int) for n in names}
    fold_rows, fold_of = [], np.zeros(len(y), int)
    for k, (trv, te) in enumerate(outer.split(np.zeros(len(y)), y, g)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=k)
        a, b = next(inner.split(np.zeros(len(trv)), y[trv], g[trv]))
        itr, iva = trv[a], trv[b]
        assert not set(g[itr]) & set(g[iva]) and not set(g[trv]) & set(g[te])
        fold_of[te] = k
        tr_mask = np.zeros(len(y), bool); tr_mask[itr] = True
        Xh, Xn = header_fold(h, tr_mask), header_fold(h, tr_mask, numbers_only=True)
        P = {}
        P["header_LR"] = lr_fit(Xh, y, itr, iva)
        P["headerNumbersOnly_LR"] = lr_fit(Xn, y, itr, iva)
        for nm, X, head in (("header_cumlink", Xh, "cumlink"), ("ecgfA1_multinomial", F, "multinomial"),
                            ("ecgfA1_cumlink", F, "cumlink")):
            fit = fit_head(X[itr], y[itr], X[iva], y[iva], head)
            P[nm] = (lambda f: (lambda Z: f["predict"](Z)["probs"]))(fit)
        Xof = {"header_LR": Xh, "headerNumbersOnly_LR": Xn, "header_cumlink": Xh, "ecgfA1_multinomial": F,
               "ecgfA1_cumlink": F}
        pv = {n: P[n](Xof[n][iva]) for n in P}; pt = {n: P[n](Xof[n][te]) for n in P}
        for fu, (m1, m2) in {"FU_header+ecgfA1": ("header_LR", "ecgfA1_multinomial"),
                             "FU_headerNumbersOnly+ecgfA1": ("headerNumbersOnly_LR", "ecgfA1_multinomial"),
                             "FU_ordinal_header+ecgfA1": ("header_cumlink", "ecgfA1_cumlink")}.items():
            w = fuse_w(y[iva], pv[m1], pv[m2])
            pt[fu] = w * pt[m1] + (1 - w) * pt[m2]
        for n in names:
            oof[n][te] = pt[n]; oof_pred[n][te] = pt[n].argmax(1)
            m = full_metrics(y[te], pt[n])
            fold_rows.append(dict(fold=k, model=n, n_test=len(te), acc=m["accuracy"], bal_acc=m["balanced_accuracy"],
                                  macro_f1=m["macro_f1"], qwk=m["qwk"], mild_f1=m["per_class"]["Mild"]["f1"],
                                  mild_recall=m["per_class"]["Mild"]["recall"], macro_auroc=m["auroc_ovr"]["macro"],
                                  hf_auc=m["hf_screen"]["auc"]))
        print(f"fold {k} done", flush=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(RES / "cv5_per_fold.csv", index=False)
    summ = df.drop(columns=["fold", "n_test"]).groupby("model").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.loc[names]
    pooled = {n: full_metrics(y, oof[n]) for n in names}
    for n in names:
        summ.loc[n, "pooled_oof_bal_acc"] = pooled[n]["balanced_accuracy"]
        summ.loc[n, "pooled_oof_acc"] = pooled[n]["accuracy"]
    summ.to_csv(RES / "cv5_summary.csv")
    tests = {}
    for a, b in (("header_LR", "FU_header+ecgfA1"), ("headerNumbersOnly_LR", "FU_headerNumbersOnly+ecgfA1"),
                 ("header_LR", "header_cumlink"), ("ecgfA1_multinomial", "ecgfA1_cumlink"),
                 ("FU_header+ecgfA1", "FU_ordinal_header+ecgfA1"), ("header_LR", "headerNumbersOnly_LR")):
        tests[f"{b} vs {a}"] = paired_bootstrap(y, oof[a], oof[b], g)
        d = df[df.model == b].set_index("fold").bal_acc - df[df.model == a].set_index("fold").bal_acc
        tests[f"{b} vs {a}"]["folds_bal_acc_improved"] = int((d > 0).sum())
    json.dump(dict(pooled={n: {k: pooled[n][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk")}
                           | {"mild_f1": pooled[n]["per_class"]["Mild"]["f1"], "macro_auroc": pooled[n]["auroc_ovr"]["macro"]}
                           for n in names}, paired_tests_pooled_oof=tests),
              open(RES / "cv5_tests.json", "w"), indent=2)
    pd.DataFrame({"ecg_id": meta.ecg_id, "fold": fold_of, "true": y,
                  **{f"pred_{n}": oof_pred[n] for n in names}}).to_csv(RES / "cv5_oof_predictions.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    for k, v in tests.items():
        print(k, {m: (round(v[m]["diff"], 3), [round(c, 3) for c in v[m]["ci95"]], round(v[m]["p_two_sided"], 3))
                  for m in ("accuracy", "balanced_accuracy", "macro_f1", "mild_f1")}, "folds+", v["folds_bal_acc_improved"])


if __name__ == "__main__":
    main()
