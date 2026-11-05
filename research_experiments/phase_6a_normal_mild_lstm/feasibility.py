"""Feasibility of an 80 % accuracy target (owner request 2026-11-05). Analysis of EXISTING development-CV predictions
only (no new model, no locked test). Three questions:
  1. Error budget: which EF-prediction error (SD) / correlation would four-class 80 % need? (Phase 4 simulation method:
     predicted EF = nearest-echo EF + Gaussian error, clinical rule, 200 draws.)
  2. Clinically standard binary tasks derived from the four-class probabilities of the C1 vote (nested threshold: chosen on
     the training portion's inner OOF only, maximising accuracy):  T1 Normal vs reduced EF (Mild+Moderate+Severe = the
     original project's "further evaluation" YES/NO);  T2 EF < 41 % (Moderate+Severe) vs >= 41 %.
  3. Selective four-class prediction: abstain on low-confidence ECGs; the confidence threshold that gives >= 80 % accuracy
     is chosen on the inner OOF of the training portion only and applied to the outer fold -> achieved accuracy and
     coverage (fraction of ECGs that receive a prediction).
Output (aggregates): results/feasibility_80.json, plots/feasibility_80.png
    python -m research_experiments.phase_6a_normal_mild_lstm.feasibility
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research_experiments.phase_6.p6common import INNER6, load_run as load6  # noqa: E402
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import PLOTS, RES, Y, fold_vec, guard, label_table  # noqa: E402

MEMBERS = ("mlpfusion", "ecgflindist", "hdrclinlr")


def rule(ef, male):
    return np.where(male & (ef > 52) | ~male & (ef >= 54), 0, np.where(ef >= 41, 1, np.where(ef >= 31, 2, 3)))


def error_budget():
    T = label_table(); ok = np.isfinite(T.ef.values); ef, male, y = T.ef.values[ok], T.male.values[ok], Y[ok]
    sd = ef.std(); rng = np.random.default_rng(0); out = []
    for s in np.arange(0, 21, 1.0):
        acc = np.mean([np.mean(rule(ef + rng.normal(0, s, len(ef)), male) == y) for _ in range(200)])
        out.append(dict(sigma=float(s), r=float(sd / np.sqrt(sd ** 2 + s ** 2)), accuracy=float(acc)))
    return out


def inner_vote(r, k):
    Ps = [np.load(INNER6 / b / f"r{r}_k{k}.npz") for b in MEMBERS]
    return Ps[0]["TR"], np.mean([z["P"] for z in Ps], 0)


def best_threshold(score, yb):
    cand = np.unique(np.round(score, 3))
    accs = [np.mean((score >= t) == yb) for t in cand]
    return float(cand[int(np.argmax(accs))])


def binary_tasks():
    res = {}
    for name, pos in (("T1 Normal vs reduced EF (Mild/Moderate/Severe)", (1, 2, 3)), ("T2 EF < 41 % (Moderate/Severe) vs >= 41 %", (2, 3))):
        reps = []
        for r in (0, 1, 2):
            P = load6(f"P5A-import-vote-v1__s0_r{r}")[0]; fv = fold_vec(r); yb = np.isin(Y, pos)
            score = P[:, list(pos)].sum(1); pred = np.zeros(len(Y), bool)
            for k in range(5):
                TR, Pin = inner_vote(r, k)
                t = best_threshold(Pin[:, list(pos)].sum(1), np.isin(Y[TR], pos))
                pred[fv == k] = score[fv == k] >= t
            reps.append(dict(accuracy=float(np.mean(pred == yb)), sensitivity=float(np.mean(pred[yb])), specificity=float(np.mean(~pred[~yb])),
                             balanced_accuracy=float((np.mean(pred[yb]) + np.mean(~pred[~yb])) / 2), auroc=float(roc_auc_score(yb, score)),
                             prevalence=float(yb.mean()), majority_class_accuracy=float(max(yb.mean(), 1 - yb.mean()))))
        res[name] = {k: float(np.mean([x[k] for x in reps])) for k in reps[0]} | {"accuracy_per_repeat": [round(x["accuracy"], 4) for x in reps]}
    return res


def selective(target=0.80):
    reps, curve = [], []
    for r in (0, 1, 2):
        P = load6(f"P5A-import-vote-v1__s0_r{r}")[0]; fv = fold_vec(r); conf = P.max(1); pred = P.argmax(1)
        keep = np.zeros(len(Y), bool)
        for k in range(5):
            TR, Pin = inner_vote(r, k); ci, yi = Pin.max(1), Y[TR] == Pin.argmax(1)
            ts = np.sort(np.unique(np.round(ci, 3)))
            ok = [t for t in ts if (ci >= t).sum() >= 20 and yi[ci >= t].mean() >= target]
            t = ok[0] if ok else 1.01                      # lowest threshold reaching the target on the inner OOF
            keep[fv == k] = conf[fv == k] >= t
        reps.append(dict(coverage=float(keep.mean()), accuracy_on_covered=float(np.mean(pred[keep] == Y[keep])) if keep.any() else None,
                         covered_by_class={c: float(keep[Y == i].mean()) for i, c in enumerate(("Normal", "Mild", "Moderate", "Severe"))},
                         covered_predicted_class_mix={c: int((pred[keep] == i).sum()) for i, c in enumerate(("Normal", "Mild", "Moderate", "Severe"))}))
        o = np.argsort(-conf)
        for c in np.arange(0.1, 1.01, 0.05):
            n = int(round(c * len(Y))); curve.append((r, float(c), float(np.mean(pred[o[:n]] == Y[o[:n]]))))
    cv = {round(c, 2): float(np.mean([a for rr, cc, a in curve if abs(cc - c) < 1e-9])) for c in np.arange(0.1, 1.01, 0.05)}
    return dict(nested_target=target, per_repeat=reps, mean_coverage=float(np.mean([x["coverage"] for x in reps])),
                mean_accuracy_on_covered=(float(np.mean([x["accuracy_on_covered"] for x in reps if x["accuracy_on_covered"] is not None])) if any(x["accuracy_on_covered"] is not None for x in reps) else None), accuracy_by_coverage_descriptive=cv)


def main():
    eb = error_budget(); bt = binary_tasks(); sel = selective()
    s80 = next((x for x in eb[::-1] if x["accuracy"] >= 0.80), None)
    out = dict(error_budget=eb, four_class_80_needs=s80, binary_tasks=bt, selective_four_class=sel,
               note="development-CV only; C1 vote probabilities; thresholds nested (training portion inner OOF only)")
    (guard(RES) / "feasibility_80.json").write_text(json.dumps(out, indent=1, default=float))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot([x["r"] for x in eb], [x["accuracy"] for x in eb], "o-"); ax[0].axhline(0.8, ls=":", c="k"); ax[0].axvline(0.66, ls="--", c="r")
    ax[0].set_xlabel("correlation of predicted EF with echo EF (r)"); ax[0].set_ylabel("four-class accuracy (simulated)")
    ax[0].set_title(f"Error budget: 80 % needs r >= {s80['r'] if s80 else float('nan'):.2f}; ECG models reach 0.66", fontsize=9)
    cv = sel["accuracy_by_coverage_descriptive"]
    ax[1].plot([100 * c for c in cv], list(cv.values()), "o-"); ax[1].axhline(0.8, ls=":", c="k")
    ax[1].set_xlabel("coverage: % of ECGs given a prediction (most confident first)"); ax[1].set_ylabel("four-class accuracy on covered ECGs")
    ax[1].set_title("Selective four-class prediction (C1 vote, development CV)", fontsize=9)
    fig.tight_layout(); guard(PLOTS).mkdir(parents=True, exist_ok=True); fig.savefig(PLOTS / "feasibility_80.png", dpi=150); plt.close(fig)
    print(json.dumps(dict(four_class_80_needs=s80, binary_tasks=bt, selective={k: v for k, v in sel.items() if k != "accuracy_by_coverage_descriptive"},
                          curve=sel["accuracy_by_coverage_descriptive"]), indent=1, default=float))


if __name__ == "__main__":
    main()
