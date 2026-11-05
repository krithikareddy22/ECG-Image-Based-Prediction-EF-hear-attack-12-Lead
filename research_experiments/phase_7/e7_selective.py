"""Phase 7 E7 (PHASE7_PLAN.md Amendment 3): header-free selective four-class prediction that must still detect Mild.
The model answers only the ECGs it is confident about (per-predicted-class confidence thresholds); the rest are referred
to echo. Per outer fold, thresholds are chosen on the fold's training portion only (nested inner OOF): maximise the
answered share subject to accuracy on answered >= 0.70 AND overall Mild recall >= m (true-Mild ECGs that are answered
AND predicted Mild, over ALL true-Mild ECGs; referred Mild counts as missed). m in {0.10, 0.15, 0.20}.
Base: IMG-vote (ECGFounder MLP + linear, image only) argmax; variant AG also requires agreement with header-free EF
grading (E6). Confidence = IMG-vote probability of the predicted class.
    python -m research_experiments.phase_7.e7_selective
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.e6_cascade import OUTD

B = C.B
MREQ = (0.10, 0.15, 0.20)
TARGET = 0.70
QS = np.linspace(0, 0.9, 10)                    # per-class thresholds = quantiles of that class's confidence (inner)


def choose(conf, pred, y, agree, m):
    """Per-class thresholds (one per predicted class) maximising coverage s.t. accuracy >= TARGET, Mild recall >= m."""
    cand = []
    for c in range(4):
        v = conf[(pred == c) & agree]
        cand.append(np.r_[np.quantile(v, QS) if len(v) else [np.inf], np.inf])      # inf = never answer this class
    n_mild = (y == 1).sum(); best = None
    for t in product(*cand):
        keep = agree & (conf >= np.asarray(t)[pred])
        n = keep.sum()
        if n < 20:
            continue
        acc = np.mean(pred[keep] == y[keep]); mr = np.sum(keep & (pred == 1) & (y == 1)) / n_mild
        if acc >= TARGET and mr >= m and (best is None or n > best[0]):
            best = (n, t)
    return None if best is None else np.asarray(best[1])


def run():
    out_rows = []
    for variant in ("vote", "AG"):
        for m in MREQ:
            reps, cells = [], []
            for r in (0, 1, 2):
                Pv, _ = B.load_run(f"P6B-imgvote-IMGF-v1__s0_r{r}"); pred = Pv.argmax(1); conf = Pv.max(1)
                fv = B.fold_vec(r); keep = np.zeros(C.nD, bool); infeasible = 0
                for k in range(5):
                    z1 = np.load(INNER6 / "ecgflindist" / f"r{r}_k{k}.npz"); z2 = np.load(B.INNER / "ecgfmlp_IMGF" / f"r{r}_k{k}.npz")
                    TR = z1["TR"]; Pin = (z1["P"] + z2["P"]) / 2; pin = Pin.argmax(1); cin = Pin.max(1)
                    zi = np.load(OUTD / f"inner_r{r}_k{k}.npz"); assert (zi["TR"] == TR).all()
                    ote = np.flatnonzero(fv == k); assert (zi["ote"] == ote).all()
                    ag_in = (pin == zi["g_in"]) if variant == "AG" else np.ones(len(TR), bool)
                    ag_out = (pred[ote] == zi["g_out"]) if variant == "AG" else np.ones(len(ote), bool)
                    t = choose(cin, pin, C.Y[TR], ag_in, m)
                    if t is None:
                        infeasible += 1; continue                                   # nothing answered in this fold
                    keep[ote] = ag_out & (conf[ote] >= t[pred[ote]])
                    s = keep & (fv == k)
                    if s.any():
                        cells.append(float(np.mean(pred[s] == C.Y[s])))
                y = C.Y
                reps.append(dict(coverage=keep.mean(), accuracy=np.mean(pred[keep] == y[keep]) if keep.any() else np.nan,
                                 mild_recall_overall=np.sum(keep & (pred == 1) & (y == 1)) / (y == 1).sum(),
                                 mild_precision_answered=(np.mean(y[keep & (pred == 1)] == 1) if (keep & (pred == 1)).any() else np.nan),
                                 n_mild_predicted=int((keep & (pred == 1)).sum()), infeasible_folds=infeasible,
                                 cov_by_class=[float(keep[y == i].mean()) for i in range(4)],
                                 recall_answered=[float(np.mean(pred[keep & (y == i)] == i)) if (keep & (y == i)).any() else np.nan for i in range(4)],
                                 mix=np.bincount(pred[keep], minlength=4).tolist()))
            ci = C.corr_ci(cells) if len(cells) > 2 else [np.nan, np.nan]
            row = dict(variant=variant, mild_requirement=m, coverage=float(np.mean([x["coverage"] for x in reps])),
                       accuracy_answered=float(np.nanmean([x["accuracy"] for x in reps])), accuracy_ci95_corrected=f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                       mild_recall_overall=float(np.mean([x["mild_recall_overall"] for x in reps])),
                       mild_precision_answered=float(np.nanmean([x["mild_precision_answered"] for x in reps])),
                       n_mild_predicted=float(np.mean([x["n_mild_predicted"] for x in reps])),
                       infeasible_folds=int(sum(x["infeasible_folds"] for x in reps)),
                       **{f"answered_{c}": float(np.mean([x["cov_by_class"][i] for x in reps])) for i, c in enumerate(C.CLASSES)},
                       **{f"recall_answered_{c}": float(np.nanmean([x["recall_answered"][i] for x in reps])) for i, c in enumerate(C.CLASSES)},
                       **{f"predicted_{c}": float(np.mean([x["mix"][i] for x in reps])) for i, c in enumerate(C.CLASSES)})
            out_rows.append(row)
    D = pd.DataFrame(out_rows)
    D.to_csv(C.guard7(C.RES7 / "e7_selective_mild.csv"), index=False, float_format="%.4f")
    ok = D[(D.accuracy_answered >= 0.69) & (D.mild_recall_overall >= 0.10)]
    summ = dict(meets_owner_rule=ok.to_dict("records"),
                rule="answered share chosen by the model; accuracy on answered > 0.69 and Mild detected (overall Mild recall >= 0.10)")
    (C.guard7(C.RES7) / "e7_selective_mild.json").write_text(json.dumps(dict(rows=out_rows, **summ), indent=1, default=float))
    pd.set_option("display.width", 250)
    print(D.to_string(index=False, float_format="%.3f"))
    return D


if __name__ == "__main__":
    run()
