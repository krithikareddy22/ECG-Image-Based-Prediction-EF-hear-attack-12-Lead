"""Selective four-class prediction for the header-free IMG-vote (AMENDMENT_IMGONLY.md follow-up, owner request).
The model abstains when its confidence (max class probability) is below a threshold. For each target accuracy the
threshold is the lowest one that reaches the target on the training portion's nested inner OOF (>= 20 ECGs kept);
it is applied unchanged to the outer fold. Reported per target: achieved accuracy, coverage (share of ECGs given a
prediction), coverage per true class, predicted-class mix, and per-class recall among covered ECGs. A descriptive
accuracy-coverage curve is added. Development CV, original labels, no header input.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.selective_img
"""
import json

import numpy as np

from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

P2 = B.ROOT / "research_experiments" / "phase_2" / "results"


def sfx(r):
    return "" if r == 0 else f"_r{r}"


def outer(r):
    mlp = [P2 / ("ecgf_aligned_f6_mlp" if (s == 42 and r == 0) else f"ecgf_f6_mlp_seed{s}{sfx(r)}") / "oof.csv" for s in B.SEEDS]
    Pm = np.mean([B.load_oof_csv(p)[0] for p in mlp], 0)
    Pl = B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]
    return (Pm + Pl) / 2


def inner(r, k):
    zs = [np.load(INNER6 / b / f"r{r}_k{k}.npz") for b in ("ecgfmlp", "ecgflindist")]
    return zs[0]["TR"], np.mean([z["P"] for z in zs], 0)


def main():
    out = dict(targets={}, curve={})
    for target in (0.60, 0.65, 0.70):
        reps = []
        for r in (0, 1, 2):
            P = outer(r); conf, pred = P.max(1), P.argmax(1); fv = B.fold_vec(r); keep = np.zeros(B.nD, bool)
            for k in range(5):
                TR, Pi = inner(r, k); ci, ok = Pi.max(1), Pi.argmax(1) == B.Y[TR]
                ts = [t for t in np.unique(np.round(ci, 3)) if (ci >= t).sum() >= 20 and ok[ci >= t].mean() >= target]
                t = ts[0] if ts else 1.01
                keep[fv == k] = conf[fv == k] >= t
            y, p = B.Y[keep], pred[keep]
            reps.append(dict(coverage=float(keep.mean()), accuracy=float(np.mean(p == y)) if keep.any() else None,
                             coverage_by_class={c: float(keep[B.Y == i].mean()) for i, c in enumerate(B.CLASSES)},
                             recall_by_class_covered={c: (float(np.mean(p[y == i] == i)) if (y == i).any() else None) for i, c in enumerate(B.CLASSES)},
                             predicted_mix={c: int((p == i).sum()) for i, c in enumerate(B.CLASSES)}))
        ok = [x for x in reps if x["accuracy"] is not None]
        out["targets"][str(target)] = dict(per_repeat=reps, mean_coverage=float(np.mean([x["coverage"] for x in reps])),
                                           mean_accuracy=float(np.mean([x["accuracy"] for x in ok])) if ok else None)
    for c in np.arange(0.1, 1.01, 0.1):
        a = []
        for r in (0, 1, 2):
            P = outer(r); o = np.argsort(-P.max(1)); n = int(round(c * B.nD)); a.append(np.mean(P.argmax(1)[o[:n]] == B.Y[o[:n]]))
        out["curve"][f"{c:.1f}"] = float(np.mean(a))
    (B.guard(B.RES) / "selective_imgonly.json").write_text(json.dumps(out, indent=1))
    for t, v in out["targets"].items():
        print(f"target {t}: coverage {v['mean_coverage']:.3f} accuracy {v['mean_accuracy']}  by-class coverage (r0) "
              f"{ {k: round(x, 2) for k, x in v['per_repeat'][0]['coverage_by_class'].items()} } predicted (r0) {v['per_repeat'][0]['predicted_mix']}")
    print("curve (coverage: accuracy):", {k: round(v, 3) for k, v in out["curve"].items()})


if __name__ == "__main__":
    main()
