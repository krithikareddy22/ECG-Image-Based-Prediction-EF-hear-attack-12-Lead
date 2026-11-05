"""Phase-1 result statement numbers (owner framing "triage + Mild limitation"; header-free). Uses the E6 header-free
stage-1 abstention models (thresholds set inside the training folds) and the header-free four-class references.
Reports, with corrected repeated-CV CIs over fold cells: coverage, accuracy on answered, composition of the answered
"Normal" calls (how many are truly Mild / Moderate / Severe), rule-out performance for EF < 41 % among answered
"Normal" calls, and precision of answered "Moderate" / "Severe" calls. Aggregates only.
    python -m research_experiments.phase_7.triage_statement
"""
from __future__ import annotations

import json

import numpy as np

from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.e6_cascade import stage1

B = C.B


def per_fold(fn, keep_pred):
    vals = []
    for r, (pred, keep) in enumerate(keep_pred):
        fv = B.fold_vec(r)
        for k in range(5):
            s = fv == k
            v = fn(pred[s], keep[s], C.Y[s])
            if v is not None and np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)), C.corr_ci(vals)


def main():
    out = {}
    for s1 in ("AGT-img", "G2T-img"):
        kp = [stage1(s1, r) for r in (0, 1, 2)]
        m = {}
        m["coverage"] = per_fold(lambda p, k, y: k.mean(), kp)
        m["accuracy_answered"] = per_fold(lambda p, k, y: np.mean(p[k] == y[k]) if k.any() else None, kp)
        nn = lambda p, k, y: k & (p == 0)                                              # noqa: E731
        for c, i in (("Normal", 0), ("Mild", 1), ("Moderate", 2), ("Severe", 3)):
            m[f"answered_Normal_calls_truly_{c}"] = per_fold(lambda p, k, y, i=i: np.mean(y[nn(p, k, y)] == i) if nn(p, k, y).any() else None, kp)
        m["ruleout_EF_lt_41_among_Normal_calls"] = per_fold(lambda p, k, y: np.mean(y[nn(p, k, y)] <= 1) if nn(p, k, y).any() else None, kp)
        m["share_of_all_EF_lt_41_called_Normal"] = per_fold(lambda p, k, y: np.mean(nn(p, k, y)[y >= 2]) if (y >= 2).any() else None, kp)
        m["share_of_all_Mild_called_Normal"] = per_fold(lambda p, k, y: np.mean(nn(p, k, y)[y == 1]) if (y == 1).any() else None, kp)
        red = lambda p, k, y: k & (p >= 2)                                             # noqa: E731
        m["answered_ModSev_calls_truly_EF_lt_41"] = per_fold(lambda p, k, y: np.mean(y[red(p, k, y)] >= 2) if red(p, k, y).any() else None, kp)
        m["answered_calls_per_run"] = {c: float(np.mean([np.sum(k & (p == i)) for p, k in kp])) for i, c in enumerate(C.CLASSES)}
        out[s1] = {k: (dict(mean=v[0], ci95_corrected=v[1]) if isinstance(v, tuple) else v) for k, v in m.items()}
    (C.guard7(C.RES7) / "triage_statement.json").write_text(json.dumps(out, indent=1, default=float))
    for s1, m in out.items():
        print(s1)
        for k, v in m.items():
            print(f"  {k:40s} {v['mean']:.3f} [{v['ci95_corrected'][0]:.3f}, {v['ci95_corrected'][1]:.3f}]" if "mean" in v else f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
