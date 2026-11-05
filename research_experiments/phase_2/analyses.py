"""Additional research questions on the development-set OOF predictions (no locked-test data).

    python -m research_experiments.phase_2.analyses --best TAG --compare TAG1 TAG2 ...
Q1/Q3  does ECGFounder encode the printed measurements? (CV ridge R^2 of each measurement from the embedding)
Q4     per-class F1 changes between models
Q5     measurements associated with Mild misclassification (Kruskal-Wallis + BH-FDR)
Q6     subgroup performance (sex, age band, year) with patient-bootstrap CIs
Q7     confidence vs correctness (AUROC of max-probability, ECE, accuracy at coverage)
Q8/9   learning curves (fraction of inner-train data) for a simple and a fused linear model
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from research_experiments.common import CLASSES, ROOT  # noqa: E402
from research_experiments.header_ablation import MEAS  # noqa: E402
from research_experiments.phase_2 import p2heads  # noqa: E402
from research_experiments.phase_2.cv_features import ALL, D, ecgf, make_X, nD  # noqa: E402
from research_experiments.phase_2.p2common import PLOTS, RES, fold_vector, guard, load_oof, s3  # noqa: E402
from research_experiments.phase_2.protocol import split_indices  # noqa: E402
from ecg_ef.metrics import evaluate  # noqa: E402

Y = D.label_idx.values


def oof_pred(tag):
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    return o[[f"p_{c}" for c in CLASSES]].values, (o.pred.values if "pred" in o else o[[f"p_{c}" for c in CLASSES]].values.argmax(1))


def q1_measurements_from_embedding():
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[D.ecg_id]
    F = ecgf("aligned", "f6")[:nD]
    out = {}
    for c in ["Rate", "PR", "QRSD", "QT", "QTcB", "age"]:
        y = H[c].values.astype(float); ok = np.isfinite(y)
        pred = np.full(ok.sum(), np.nan)
        for tr, te in GroupKFold(5).split(F[ok], y[ok], D.patient.values[ok]):
            mu, sd = F[ok][tr].mean(0), F[ok][tr].std(0) + 1e-6
            r = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit((F[ok][tr] - mu) / sd, y[ok][tr])
            pred[te] = r.predict((F[ok][te] - mu) / sd)
        out[c] = float(1 - np.sum((y[ok] - pred) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    return out


def q5_mild(tag):
    P, pred = oof_pred(tag)
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[D.ecg_id]
    mild = Y == 1
    grp = np.where(pred[mild] == 1, "correct", np.where(pred[mild] == 0, "called_Normal", "called_Mod/Sev"))
    rows = []
    for c in ["age", "male", "Rate", "PR", "QRSD", "QT", "QTcB", "QRS_axis", "T_axis"]:
        v = H[c].values[mild].astype(float)
        groups = [v[(grp == g) & np.isfinite(v)] for g in ("called_Normal", "correct", "called_Mod/Sev")]
        p = stats.kruskal(*groups).pvalue if all(len(x) > 2 for x in groups) else np.nan
        rows.append(dict(measure=c, **{f"median_{g}": float(np.median(x)) if len(x) else np.nan
                                       for g, x in zip(("called_Normal", "correct", "called_Mod/Sev"), groups)}, p=p))
    df = pd.DataFrame(rows)
    ok = df.p.notna()
    p = df.p[ok].values; n = len(p); order = np.argsort(p); q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    df.loc[ok, "q_bh"] = np.minimum(q, 1)
    df["n"] = [f"{(grp == 'called_Normal').sum()}/{(grp == 'correct').sum()}/{(grp == 'called_Mod/Sev').sum()}"] * len(df)
    return df


def q6_subgroups(tag):
    P, pred = oof_pred(tag)
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[D.ecg_id]
    age = H.age.values
    groups = {"male": H.male.values == 1, "female": H.male.values == 0, "age<50": age < 50,
              "age 50-65": (age >= 50) & (age <= 65), "age>65": age > 65,
              **{f"year {y}": D.year.values == y for y in sorted(D.year.unique())}}
    rng = np.random.default_rng(0); rows = []
    for name, mask in groups.items():
        idx = np.flatnonzero(mask); pats = D.patient.values[idx]; up = np.unique(pats)
        accs, bals = [], []
        for _ in range(1000):
            pick = rng.choice(up, len(up)); ii = np.concatenate([idx[pats == p] for p in pick])
            accs.append(np.mean(pred[ii] == Y[ii]))
            if len(np.unique(Y[ii])) == 4:
                bals.append(balanced_accuracy_score(Y[ii], pred[ii]))
        rows.append(dict(subgroup=name, n=len(idx), accuracy=float(np.mean(pred[idx] == Y[idx])),
                         acc_ci=[float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                         balanced_accuracy=float(balanced_accuracy_score(Y[idx], pred[idx])),
                         bal_ci=[float(np.percentile(bals, 2.5)), float(np.percentile(bals, 97.5))] if bals else None))
    return pd.DataFrame(rows)


def q7_confidence(tag):
    P, pred = oof_pred(tag)
    conf = P.max(1); correct = pred == Y
    bins = np.linspace(0, 1, 11); ece = 0; rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean()); rel.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    order = np.argsort(-conf)
    cov = {f"acc_at_{int(c * 100)}pct_coverage": float(correct[order[:int(c * len(conf))]].mean()) for c in (0.5, 0.7, 0.9, 1.0)}
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ":", c="gray"); ax.plot([r[0] for r in rel], [r[1] for r in rel], "o-")
    ax.set_xlabel("max predicted probability"); ax.set_ylabel("observed accuracy"); ax.set_title(f"reliability\n{tag[:50]}", fontsize=8)
    fig.tight_layout(); fig.savefig(guard(PLOTS / f"reliability_{tag[:80]}.png"), dpi=120); plt.close(fig)
    return dict(auroc_conf_vs_correct=float(roc_auc_score(correct, conf)), ece=float(ece), reliability=rel, **cov)


def q8_learning_curve(spec, cfg, name):
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        P = np.zeros((nD, 4))
        for k in range(5):
            itr, iva, ote = split_indices(D, 0, k)
            rng = np.random.default_rng(k)
            pats = np.unique(D.patient.values[itr]); keep = set(rng.choice(pats, int(round(frac * len(pats))), replace=False))
            sub = np.array([i for i in itr if D.patient.values[i] in keep])
            fm = np.zeros(len(ALL), bool); fm[sub] = True
            X, pre = make_X(spec, fm)
            f = p2heads.fit(X[sub], Y[sub], X[iva], Y[iva], dict(cfg, no_std=pre))
            P[ote] = f["predict"](X[ote])[0]
        m = evaluate(Y, P)
        rows.append(dict(model=name, train_fraction=frac, n_train_approx=int(frac * 535), accuracy=m["accuracy"],
                         balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"], s3=s3(m)))
        print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--best", required=True); ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--skip-lc", action="store_true")
    a = ap.parse_args()
    out = {"q1_r2_measurement_from_ecgfounder_embedding": q1_measurements_from_embedding()}
    print(out, flush=True)
    q5_mild(a.best).to_csv(guard(RES / "analysis_mild_measurements.csv"), index=False)
    q6_subgroups(a.best).to_csv(RES / "analysis_subgroups.csv", index=False)
    out["q7_confidence"] = {t: q7_confidence(t) for t in [a.best] + a.others}
    if not a.skip_lc:
        LIN = dict(head="linear", loss="ce", weight="balanced")
        lc = q8_learning_curve(dict(blocks=[("hdr", "numeric")]), LIN, "header numeric (linear)")
        lc += q8_learning_curve(dict(blocks=[("ecgf", "aligned", "f6")]), LIN, "ECGFounder frozen (linear)")
        lc += q8_learning_curve(dict(blocks=[("hdr", "numeric"), ("ecgf", "aligned", "f6")]), LIN, "header numeric + ECGFounder (linear)")
        pd.DataFrame(lc).to_csv(RES / "analysis_learning_curves.csv", index=False)
    (RES / "analyses.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
