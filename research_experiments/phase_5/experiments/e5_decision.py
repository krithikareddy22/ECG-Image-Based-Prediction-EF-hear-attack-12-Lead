"""E5: nested decision optimisation (class-prior power or per-class log-offset), chosen on the training-portion nested
inner OOF predictions only, applied to the outer fold. Objectives: accuracy (unconstrained), floor (max accuracy s.t.
every class recall >= 0.15 on the inner OOF; pre-registered), protected (balanced acc and macro-F1 >= argmax - 0.01).
Candidates: vote ensemble (inner OOF = mean of the members' inner OOF) and MLP fusion (bias rule; the prior rule is 5A).
Baseline: the same model with argmax.
    python -m research_experiments.phase_5.experiments.e5_decision
"""
import numpy as np

from research_experiments.phase_5.baseline.calibrated_mlp import objective_ok
from research_experiments.phase_5.experiments.nested import inner_oof
from research_experiments.phase_5.p5common import (ALPHAS, BIASES, SEEDS, Y, decide, done, fold_vec, load_run, mark_running, metrics,
                                                   nD, save_run)

CMD = "python -m research_experiments.phase_5.experiments.e5_decision"


def choose(P, y, prior, rule, obj):
    base = metrics(y, P, P.argmax(1)); best = None
    for c in (ALPHAS if rule == "prior" else BIASES):
        m = metrics(y, P, decide(P, rule, c, prior))
        if objective_ok(m, base, obj) and (best is None or m["accuracy"] > best[1] + 1e-12):
            best = (c, m["accuracy"])
    return best[0] if best else (0.0 if rule == "prior" else np.zeros(4))


def main():
    for r in (0, 1, 2):
        fo = fold_vec(r)
        outer = {"vote": load_run(f"P5A-import-vote-v1__s0_r{r}")[0],
                 "mlpfusion": np.mean([load_run(f"P5A-import-mlpfusion-v1__s{s}_r{r}")[0] for s in SEEDS], 0)}
        for cand in ("vote", "mlpfusion"):
            for rule in ("prior", "bias"):
                if cand == "mlpfusion" and rule == "prior":
                    continue
                for obj in ("accuracy", "floor", "protected"):
                    rid = f"P5D-E5-{cand}-{rule}-{obj}-v1__s0_r{r}"
                    if done(rid):
                        continue
                    mark_running(rid, CMD)
                    pred = np.zeros(nD, int); params = []
                    for k in range(5):
                        if cand == "vote":
                            inn = [inner_oof(b, r, k) for b in ("mlpfusion", "ecgflindist", "hdrclinlr")]
                            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
                        else:
                            TR, Pin = inner_oof("mlpfusion", r, k)
                        prior = np.bincount(Y[TR], minlength=4) / len(TR)
                        c = choose(Pin, Y[TR], prior, rule, obj)
                        pred[fo == k] = decide(outer[cand][fo == k], rule, c, prior)
                        params.append(float(c) if rule == "prior" else np.asarray(c).round(2).tolist())
                    agg = save_run(rid, outer[cand], pred, r, dict(kind="nested decision rule", candidate=cand, rule=rule, objective=obj,
                                                                   recall_floor=0.15 if obj == "floor" else None),
                                   extra=dict(params_per_fold=params))
                    m = agg["metrics"]; print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} minR {m['min_recall']:.2f}", flush=True)


if __name__ == "__main__":
    main()
