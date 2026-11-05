"""Decision-rule calibration: ORIGINAL (argmax) vs CALIBRATED predictions, development-set OOF only.

Why: models trained with inverse-frequency class weights output probabilities calibrated to a balanced
prior, so argmax favours balanced accuracy. Two post-hoc calibrations move the decision toward plain
accuracy without retraining:
  prior : p_k * pi_k^alpha           (one parameter; alpha = 1 = prior-corrected expected-accuracy rule)
  bias  : log p_k + b_k, b_Normal = 0 (per-class decision thresholds; 3 parameters on a 0.25 grid)
Leakage control:
  * every parameter used for outer fold k (alpha, b, AND the class prior pi) is estimated only from the OOF
    predictions / labels of the other four folds (cross-fitting); fold k's labels are never used for it
  * the parameters for the locked test set are estimated from the development set only; test labels are
    never read
Objectives for choosing the parameter on the other folds:
  'accuracy'  : maximise accuracy
  'protected' : maximise accuracy subject to balanced accuracy and macro-F1 >= their argmax value - 0.01
Reported for original vs calibrated: accuracy, balanced accuracy, macro-F1, Mild F1, Mild recall, QWK,
per-class recall, share of ECGs predicted Normal (true share 46 %), confusion matrices.
Discrimination (OvR AUROC) is unchanged by construction: calibration only re-allocates decisions.

    python -m research_experiments.phase_2.accuracy_tradeoff --tags ens3__fusion_mlp_r0 ... --name fusion
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from research_experiments.common import CLASSES
from research_experiments.phase_2.p2common import LOCK, PCOLS, RES
from research_experiments.phase_2.protocol import dev_and_test, folds

META, D, T = dev_and_test()
Y = D.label_idx.values
ALPHAS = np.round(np.arange(0, 1.51, 0.1), 2)
OFFS = np.arange(-1.0, 1.01, 0.25)
BIASES = [np.array([0.0, a, b, c]) for a in OFFS for b in OFFS for c in OFFS]


def mets(y, pred):
    rec = [float(np.mean(pred[y == k] == k)) for k in range(4)]
    f1 = f1_score(y, pred, labels=range(4), average=None, zero_division=0)
    return dict(accuracy=float(np.mean(pred == y)), balanced_accuracy=float(np.mean(rec)), macro_f1=float(f1.mean()),
                f1_mild=float(f1[1]), recall_mild=rec[1], qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
                **{f"recall_{c}": rec[k] for k, c in enumerate(CLASSES)},
                share_pred_normal=float(np.mean(pred == 0)))


def decide(P, method, param, prior):
    if method == "prior":
        return (P * prior[None] ** param).argmax(1)
    return (np.log(np.clip(P, 1e-9, 1)) + param[None]).argmax(1)


def choose(P, y, method, objective, prior):
    base = mets(y, P.argmax(1)); best = None
    for c in (ALPHAS if method == "prior" else BIASES):
        m = mets(y, decide(P, method, c, prior))
        if objective == "protected" and (m["balanced_accuracy"] < base["balanced_accuracy"] - 0.01
                                         or m["macro_f1"] < base["macro_f1"] - 0.01):
            continue
        if best is None or m["accuracy"] > best[1] + 1e-12:
            best = (c, m["accuracy"])
    return best[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--name", default="fusion")
    a = ap.parse_args()
    rows, cms, curve, locked_params = [], {}, [], {}
    for tag in a.tags:
        info = json.loads((RES / tag / "info.json").read_text()); r = int(info.get("repeat", 0))
        P = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values
        f = folds(r); fo = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
        pred0 = P.argmax(1)
        rows.append(dict(tag=tag, repeat=r, rule="original (argmax)", params=None, **mets(Y, pred0)))
        cms[(tag, "original (argmax)")] = confusion_matrix(Y, pred0, labels=range(4))
        prior_all = np.bincount(Y, minlength=4) / len(Y)
        for al in ALPHAS:                                                     # descriptive curve
            curve.append(dict(tag=tag, repeat=r, alpha=al, **mets(Y, decide(P, "prior", al, prior_all))))
        for method in ("prior", "bias"):
            for obj in ("accuracy", "protected"):
                pred = np.zeros(len(Y), int); chosen = []
                for k in range(5):
                    tr = fo != k
                    prior_k = np.bincount(Y[tr], minlength=4) / tr.sum()      # prior from the OTHER folds only
                    c = choose(P[tr], Y[tr], method, obj, prior_k)
                    pred[fo == k] = decide(P[fo == k], method, c, prior_k)
                    chosen.append(float(c) if method == "prior" else np.asarray(c).round(2).tolist())
                rule = f"calibrated: {method}, {obj}"
                rows.append(dict(tag=tag, repeat=r, rule=rule, params=chosen, **mets(Y, pred)))
                cms[(tag, rule)] = confusion_matrix(Y, pred, labels=range(4))
                # parameter for the locked test set: estimated on the whole development set (no test labels)
                locked_params[f"{tag} | {rule}"] = (lambda c: float(c) if method == "prior" else np.asarray(c).tolist())(
                    choose(P, Y, method, obj, prior_all))
    R = pd.DataFrame(rows)
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "f1_mild", "recall_mild", "qwk",
            "recall_Normal", "recall_Moderate", "recall_Severe", "share_pred_normal"]
    base = R[R.rule == "original (argmax)"].set_index("tag")[keys]
    for k in keys:
        R[f"delta_{k}"] = R[k].values - base.loc[R.tag, k].values
    R.to_csv(RES / f"calibration_{a.name}_per_run.csv", index=False)
    agg = R.groupby("rule")[keys + [f"delta_{k}" for k in keys[:6]]].agg(["mean", "std"])
    agg.to_csv(RES / f"calibration_{a.name}_summary.csv")
    pd.DataFrame(curve).to_csv(RES / f"calibration_{a.name}_alpha_curve.csv", index=False)
    cm_sum = {}
    for (tag, rule), cm in cms.items():
        cm_sum[rule] = cm_sum.get(rule, 0) + cm
    md = [f"Confusion matrices summed over {len(a.tags)} runs (rows = true N/Mi/Mo/S, cols = predicted)\n"]
    for rule, cm in cm_sum.items():
        md.append(f"**{rule}**\n```\n" + "\n".join(" ".join(f"{x:5d}" for x in row) for row in cm) + "\n```")
    (RES / f"calibration_{a.name}_confusion.md").write_text("\n".join(md))
    (RES / f"calibration_{a.name}_locked_params.json").write_text(json.dumps(locked_params, indent=1))
    pd.set_option("display.width", 250)
    print(R.groupby("rule")[keys].mean().round(3).to_string())
    print(R.groupby("rule")[[f"delta_{k}" for k in keys[:6]]].agg(["mean", "std"]).round(3).to_string())
    print("\n".join(md))


if __name__ == "__main__":
    main()
