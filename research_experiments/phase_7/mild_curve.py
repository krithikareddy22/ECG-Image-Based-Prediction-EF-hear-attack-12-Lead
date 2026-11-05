"""Descriptive curve for the phase-1 statement: share of ECGs predicted Mild vs four-class accuracy (all 837 ECGs).
Image-only IMG-vote probabilities with a FIXED Mild logit offset b (same b in every fold, no tuning); one point per b,
mean over the 3 CV repeats. Also Mild precision / recall. Aggregates only.
    python -m research_experiments.phase_7.mild_curve
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research_experiments.phase_7 import p7common as C

OFFSETS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def main():
    rows = []
    for b in OFFSETS:
        v = []
        for r in (0, 1, 2):
            P, _ = C.B.load_run(f"P6B-imgvote-IMGF-v1__s0_r{r}")
            pred = (np.log(P.clip(1e-9)) + np.array([0, b, 0, 0])).argmax(1); m = pred == 1; y = C.Y
            v.append((m.mean(), np.mean(pred == y), np.mean(y[m] == 1) if m.any() else np.nan, np.mean(pred[y == 1] == 1)))
        a = np.nanmean(np.array(v), 0)
        rows.append(dict(offset=b, pct_predicted_mild=round(100 * a[0], 1), accuracy=round(a[1], 4), mild_precision=round(a[2], 4),
                         mild_recall=round(a[3], 4)))
    D = pd.DataFrame(rows)
    D.to_csv(C.guard7(C.RES7 / "mild_share_vs_accuracy.csv"), index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
