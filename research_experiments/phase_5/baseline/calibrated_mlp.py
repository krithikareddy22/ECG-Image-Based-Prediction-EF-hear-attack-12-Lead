"""5A: reproduce the 56.8 % calibrated MLP fusion WITH stored per-ECG predictions (they were never saved in phase 2).

Outer predictions = seed mean (42/43/44) of the imported phase-2 MLP-fusion OOF (P5A-import-mlpfusion-v1).
Decision rule: class-prior correction  pred = argmax(p * prior_k ** alpha), prior_k = label frequencies of the outer
training portion; alpha (grid 0-1.5) chosen on the nested inner OOF predictions of that training portion only.
Objectives:  protected  max accuracy s.t. balanced acc and macro-F1 >= argmax - 0.01   (= the 56.8 % rule)
             accuracy   max accuracy, unconstrained
             floor      max accuracy s.t. every class recall >= RECALL_FLOOR (0.15) on the inner OOF (pre-registered)
    python -m research_experiments.phase_5.baseline.calibrated_mlp --repeat 0
"""
import argparse

import numpy as np

from research_experiments.phase_5.experiments.nested import inner_oof
from research_experiments.phase_5.p5common import (ALPHAS, RECALL_FLOOR, SEEDS, Y, decide, done, fold_vec, load_run,
                                                   mark_running, metrics, nD, save_run)


def objective_ok(m, base, obj):
    if obj == "protected":
        return m["balanced_accuracy"] >= base["balanced_accuracy"] - 0.01 and m["macro_f1"] >= base["macro_f1"] - 0.01
    if obj == "floor":
        return m["min_recall"] >= RECALL_FLOOR
    return True


def choose_alpha(P, y, prior, obj):
    base = metrics(y, P, P.argmax(1)); best = None
    for a in ALPHAS:
        m = metrics(y, P, decide(P, "prior", a, prior))
        if not objective_ok(m, base, obj):
            continue
        if best is None or m["accuracy"] > best[1] + 1e-12:
            best = (float(a), m["accuracy"])
    return best[0] if best else 0.0


def run(r, base_import="P5A-import-mlpfusion-v1", inner_base="mlpfusion", name="calmlp"):
    rids = {o: f"P5A-{name}-{o}-v1__s0_r{r}" for o in ("protected", "accuracy", "floor")}
    if all(done(x) for x in rids.values()):
        return
    for x in rids.values():
        mark_running(x, f"python -m research_experiments.phase_5.baseline.calibrated_mlp --repeat {r}")
    P_out = np.mean([load_run(f"{base_import}__s{s}_r{r}")[0] for s in SEEDS], 0)
    fo = fold_vec(r)
    pred = {o: np.zeros(nD, int) for o in rids}; params = {o: [] for o in rids}
    for k in range(5):
        TR, P_in = inner_oof(inner_base, r, k)
        prior = np.bincount(Y[TR], minlength=4) / len(TR)
        for o in rids:
            a = choose_alpha(P_in, Y[TR], prior, o)
            pred[o][fo == k] = decide(P_out[fo == k], "prior", a, prior); params[o].append(a)
    for o, rid in rids.items():
        if done(rid):
            continue
        agg = save_run(rid, P_out, pred[o], r, dict(kind="nested prior calibration", objective=o, base=base_import, seeds=list(SEEDS),
                                                    inner="phase_5.experiments.nested (4-fold, rs 1000+10r+k; inner split rs 2000+10k+j)",
                                                    recall_floor=0.15 if o == "floor" else None),
                       extra=dict(alpha_per_fold=params[o]))
        m = agg["metrics"]
        print(f"[{rid}] acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} "
              f"recalls {[round(m[f'recall_{c}'], 3) for c in ('Normal', 'Mild', 'Moderate', 'Severe')]} alpha {params[o]}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeat", type=int, nargs="+", default=[0, 1, 2])
    for r in ap.parse_args().repeat:
        run(r)


if __name__ == "__main__":
    main()
