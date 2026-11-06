"""Triage (header-free AGT-img, 56 % answered) scores on the ANSWERED ECGs, and a descriptive Mild trade-off curve on the
same answered set. On answered ECGs the triage's class equals the IMG-C3 decision (stage 1 answers only where they
agree), i.e. argmax(log P_vote + alpha_k * log prior_k). The curve adds a FIXED Mild offset b to that decision (same b
in every fold; no tuning), keeping the answered set unchanged. Aggregates only.
    python -m research_experiments.phase_7.triage_mild_curve
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.e6_cascade import imgc3_alphas, stage1

B = C.B
OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def decisions(r, b):
    """IMG-C3 decision with a Mild offset b, per outer fold (prior and alpha from that fold's training rows)."""
    Pv, _ = B.load_run(f"P6B-imgvote-IMGF-v1__s0_r{r}"); al = imgc3_alphas(r); fv = B.fold_vec(r)
    pred = np.zeros(C.nD, int)
    for k in range(5):
        TR = np.load(INNER6 / "ecgflindist" / f"r{r}_k{k}.npz")["TR"]
        prior = np.bincount(C.Y[TR], minlength=4) / len(TR); s = fv == k
        pred[s] = (np.log(Pv[s].clip(1e-9)) + al[k] * np.log(prior)[None] + np.array([0, b, 0, 0])).argmax(1)
    return pred


def main():
    y = C.Y; scores, cl = [], {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for r in (0, 1, 2):
        g, keep = stage1("AGT-img", r)
        assert (decisions(r, 0.0)[keep] == g[keep]).all(), "answered triage class must equal the IMG-C3 decision"
        P = np.eye(4)[g]
        scores.append(B.metrics(y[keep], P[keep], g[keep]))
        for c in C.cells(P, g, r, mask=keep):
            for k in cl:
                cl[k].append(c[k])
    S = {k: float(np.mean([m[k] for m in scores])) for k in ["accuracy", "balanced_accuracy", "macro_f1"] +
         [f"{a}_{c}" for c in C.CLASSES for a in ("precision", "recall", "f1")]}
    S.update({f"{k}_ci95_corrected": C.corr_ci(v) for k, v in cl.items()})
    S["answered_share"] = float(np.mean([stage1("AGT-img", r)[1].mean() for r in (0, 1, 2)]))
    S["answered_true_class_counts"] = {c: float(np.mean([np.sum(stage1("AGT-img", r)[1] & (y == i)) for r in (0, 1, 2)])) for i, c in enumerate(C.CLASSES)}
    S["answered_predicted_counts"] = {c: float(np.mean([np.sum(stage1("AGT-img", r)[1] & (stage1("AGT-img", r)[0] == i)) for r in (0, 1, 2)])) for i, c in enumerate(C.CLASSES)}
    rows = []
    for b in OFFSETS:
        v = []
        for r in (0, 1, 2):
            _, keep = stage1("AGT-img", r); p = decisions(r, b); m = keep & (p == 1)
            v.append((m.sum() / keep.sum(), np.mean(p[keep] == y[keep]), np.mean(y[m] == 1) if m.any() else np.nan,
                      np.mean(p[keep & (y == 1)] == 1), np.sum(m & (y == 1)) / (y == 1).sum(),
                      B.metrics(y[keep], np.eye(4)[p[keep]], p[keep])["macro_f1"]))
        a = np.nanmean(np.array(v, float), 0)
        rows.append(dict(offset=b, pct_answered_predicted_mild=round(100 * a[0], 1), accuracy_answered=round(a[1], 4),
                         mild_precision_answered=round(a[2], 4), mild_recall_answered=round(a[3], 4),
                         mild_recall_all_patients=round(a[4], 4), macro_f1_answered=round(a[5], 4)))
    D = pd.DataFrame(rows)
    D.to_csv(C.guard7(C.RES7 / "triage_mild_curve.csv"), index=False)
    (C.guard7(C.RES7) / "triage_scores.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1)); print(D.to_string(index=False))


if __name__ == "__main__":
    main()
