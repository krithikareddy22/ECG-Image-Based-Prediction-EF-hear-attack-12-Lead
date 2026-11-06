"""Phase 7 E7 feasibility check (descriptive): on the INNER (training-fold) IMG-vote predictions of three outer folds,
how precise are the most confident Mild calls, and what is the best accuracy on answered ECGs once >= 10 % of all Mild
ECGs must be correctly called Mild? Explains why E7 found no feasible threshold. Aggregates only.
    python -m research_experiments.phase_7.e7_feasibility
"""
import json
import numpy as np

from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_7 import p7common as C

B = C.B
OUT = {}
for r, k in ((0, 0), (1, 2), (2, 4)):
    z1 = np.load(INNER6 / "ecgflindist" / f"r{r}_k{k}.npz"); z2 = np.load(B.INNER / "ecgfmlp_IMGF" / f"r{r}_k{k}.npz")
    TR = z1["TR"]; P = (z1["P"] + z2["P"]) / 2; pred = P.argmax(1); conf = P.max(1); y = C.Y[TR]
    nm = (y == 1).sum()
    print(f"r{r} k{k}: inner rows {len(TR)}, true Mild {nm}, predicted Mild {(pred == 1).sum()}, Mild precision {np.mean(y[pred == 1] == 1):.3f}")
    # precision of the most confident Mild predictions
    mi = np.flatnonzero(pred == 1); o = mi[np.argsort(-conf[mi])]
    for n in (10, 20, 30, 40, len(o)):
        print(f"   top {n:3d} Mild calls: precision {np.mean(y[o[:n]] == 1):.3f}, correct {np.sum(y[o[:n]] == 1)} = {np.sum(y[o[:n]] == 1) / nm:.3f} of all Mild")
    # best accuracy of the non-Mild answers at any confidence level
    nn = np.flatnonzero(pred != 1); o2 = nn[np.argsort(-conf[nn])]
    for n in (50, 100, 200, 300, 400, len(o2)):
        print(f"   top {n:3d} non-Mild calls: accuracy {np.mean(pred[o2[:n]] == y[o2[:n]]):.3f}")
    # the needed combination: >= 10 % of Mild correct (keep top-k Mild) + top-x non-Mild, overall >= 0.70?
    k10 = int(np.ceil(0.10 * nm)); cum = np.cumsum(y[o] == 1); kk = int(np.argmax(cum >= k10)) + 1 if (cum >= k10).any() else None
    if kk:
        best = max(((np.sum(pred[o2[:x]] == y[o2[:x]]) + k10) / (x + kk), x) for x in range(20, len(o2) + 1))
        print(f"   need {kk} Mild calls for {k10} correct; best overall accuracy with any number of top non-Mild calls: {best[0]:.3f} (answering {best[1] + kk} of {len(TR)})")
        OUT[f"r{r}_k{k}"] = dict(inner_rows=int(len(TR)), true_mild=int(nm), mild_calls=int((pred == 1).sum()),
                                 mild_precision=float(np.mean(y[pred == 1] == 1)), mild_calls_needed_for_10pct=kk,
                                 best_accuracy_with_10pct_mild=float(best[0]), answered=int(best[1] + kk))
(C.guard7(C.RES7) / "e7_feasibility.json").write_text(json.dumps(OUT, indent=1))
