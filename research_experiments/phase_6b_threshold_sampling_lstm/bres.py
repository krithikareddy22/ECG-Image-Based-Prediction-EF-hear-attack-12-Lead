"""Exploratory BRES configuration (AMENDMENT_BRES.md): existing model recipe re-trained with borderline Mild resolved
to the nearest neighbouring class; scored on original and BRES labels.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.bres
"""
import json

import numpy as np

from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B


def y_bres():
    T = B.table(); lab, ef = T.label.values, T.ef.values
    cons = (lab == "Mild") & (T.derived.values == "Mild")
    up = cons & (np.abs(ef - T.sex_boundary.values) <= 2.0)
    lo = cons & (np.abs(ef - 41.0) <= 2.0)
    y = T.y.values.copy(); y[up] = 0; y[lo] = 2
    return y, int(up.sum()), int(lo.sum())


def main():
    yb, n_up, n_lo = y_bres()
    E.y_def = lambda cfg: yb if cfg == "BRES" else B.y_def(cfg)        # training labels for this exploratory config
    E.run_config("BRES")
    out = dict(moved_to_normal=n_up, moved_to_moderate=n_lo, class_counts=np.bincount(yb, minlength=4).tolist(), models={})
    for m, seeds, rule in (("mlpfusion", B.SEEDS, False), ("vote", (0,), False), ("c3", (0,), True)):
        res = {"original": [], "BRES": [], "B1_on_BRES": []}
        for r in (0, 1, 2):
            Ps = [B.load_run(f"P6B-{m}-BRES-v1__s{s}_r{r}") for s in seeds]
            P = np.mean([p[0] for p in Ps], 0); pred = Ps[0][1] if rule else P.argmax(1)
            res["original"].append(B.metrics(B.Y, P, pred)); res["BRES"].append(B.metrics(yb, P, pred))
            b1 = {"mlpfusion": [f"P6A-mlpfusion-B1-v1__s{s}_r{r}" for s in B.SEEDS], "vote": [f"P6A-vote-B1-v1__s0_r{r}"],
                  "c3": [f"P6A-c3-B1-v1__s0_r{r}"]}[m]
            Qs = [B.load_run(x) for x in b1]; Q = np.mean([q[0] for q in Qs], 0); qp = Qs[0][1] if rule else Q.argmax(1)
            res["B1_on_BRES"].append(B.metrics(yb, Q, qp))
        out["models"][m] = {k: {mm: float(np.mean([x[mm] for x in v])) for mm in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild",
                                                                                    "recall_Mild", "min_recall")} for k, v in res.items()}
    (B.guard(B.RES) / "bres_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
