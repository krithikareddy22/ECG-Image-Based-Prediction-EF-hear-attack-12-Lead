"""E7: training-strategy comparison on the MLP fusion (header numeric || ECGFounder f6), ONE change at a time vs the
identical default (mlp5 default == phase-2 MLP fusion, verified): focal loss (gamma 2), label smoothing 0.1, mixup
(alpha 0.2), cosine learning-rate schedule, early stopping / L2 selection on inner-val ACCURACY instead of S3.
Already completed in phase 2 (repeat 0, seeds 42-44) and re-used here instead of repeated (--import-phase2):
class-unweighted CE, class-balanced sampler, dropout 0.3 / 0.7, hidden 32 / 128.
Weight decay is selected on inner-val in every run (L2 grid); input dropout 0.5 is the default feature dropout.
    python -m research_experiments.phase_5.experiments.e7_training --jobs 3 --part 0
    python -m research_experiments.phase_5.experiments.e7_training --import-phase2
"""
import argparse

import torch

from research_experiments.phase_5.experiments import mlp5
from research_experiments.phase_5.experiments.runner import cv_run
from research_experiments.phase_5.p5common import P2, REPEATS, SEEDS, done, load_oof_csv, mark_running, save_run
from research_experiments.phase_5.registry_ext import E7_PHASE2, E7_VARIANTS

CMD = "python -m research_experiments.phase_5.experiments.e7_training"


def import_phase2():
    for v, tag in E7_PHASE2.items():
        for s in SEEDS:
            rid = f"P5D-E7-p2-{v}-v1__s{s}_r0"
            if done(rid):
                continue
            src = P2 / "results" / (tag if s == 42 else f"{tag}_seed{s}") / "oof.csv"
            mark_running(rid, CMD + " --import-phase2")
            P, pred = load_oof_csv(src)
            save_run(rid, P, pred, 0, dict(kind="import (phase-2 completed run)", source=str(src.relative_to(P2.parent.parent)), seed=s))


def augment_run(s, r, cmd):
    """MLP fusion trained on original + 2 augmented ECGFounder embeddings of the inner-train ECGs (validation and outer
    folds use the original embeddings only). Block standardisation uses the ORIGINAL inner-train statistics."""
    import numpy as np
    from research_experiments.phase_5.p5common import FEAT5, Y, ecgf_f6, header_numeric, nD, split_indices
    rid = f"P5D-E7-augment-v1__s{s}_r{r}"
    if done(rid):
        return
    mark_running(rid, cmd)
    E = ecgf_f6().astype(float); EA = [np.load(FEAT5 / f"ecgf_aug{k}.npz")["f6"].astype(float) for k in (1, 2)]
    P = np.zeros((nD, 4))
    for k in range(5):
        itr, iva, ote = split_indices(r, k); fm = np.zeros(nD, bool); fm[itr] = True
        H = header_numeric(fm)
        bs = lambda B, ref: (B - ref[fm].mean(0)) / (ref[fm].std(0) + 1e-6) / np.sqrt(B.shape[1])
        X = np.concatenate([bs(H, H), bs(E, E)], 1)
        XA = [np.concatenate([bs(H, H), bs(Ea, E)], 1) for Ea in EA]
        Xtr = np.concatenate([X[itr]] + [xa[itr] for xa in XA]); ytr = np.tile(Y[itr], 1 + len(XA))
        f, _ = mlp5.fit(Xtr, ytr, X[iva], Y[iva], dict(no_std=True), seed=s)
        P[ote] = f(X[ote])
    agg = save_run(rid, P, P.argmax(1), r, dict(kind="MLP fusion + waveform augmentation", n_augmented_copies=2, seed=s,
                                                augmentation="features/ecgf_augment.py", baseline="P5A-import-mlpfusion-v1"))
    print(f"[{rid}] acc {agg['metrics']['accuracy']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--import-phase2", action="store_true"); ap.add_argument("--augment", action="store_true")
    ap.add_argument("--unweighted", action="store_true"); a = ap.parse_args()
    if a.unweighted:
        torch.set_num_threads(1)
        for s, r in [(s, r) for r in REPEATS for s in SEEDS][a.part::a.jobs]:
            rid = f"P5D-E7-unweighted-v1__s{s}_r{r}"
            if done(rid):
                continue
            if r == 0:                      # completed in phase 2 -> re-use, do not retrain
                src = P2 / "results" / ("ff_mlp_unweighted" if s == 42 else f"ff_mlp_unweighted_seed{s}") / "oof.csv"
                mark_running(rid, CMD + " --unweighted"); P, pred = load_oof_csv(src)
                save_run(rid, P, pred, 0, dict(kind="import (phase-2 completed run)", variant="unweighted", source=str(src.relative_to(P2.parent.parent)), seed=s))
            else:
                cv_run("P5D-E7-unweighted-v1", ["hdr", "ecgf"], "fitter", s, r, fitter=mlp5.fitter(dict(weight="none", no_std=True)),
                       command=f"{CMD} --unweighted --jobs {a.jobs} --part {a.part}",
                       extra_cfg=dict(variant="unweighted", change=dict(weight="none"), baseline="P5A-import-mlpfusion-v1"))
        return
    if a.import_phase2:
        return import_phase2()
    if a.augment:
        torch.set_num_threads(1)
        for s, r in [(s, r) for r in REPEATS for s in SEEDS][a.part::a.jobs]:
            augment_run(s, r, f"{CMD} --augment --jobs {a.jobs} --part {a.part}")
        return
    torch.set_num_threads(1)
    for v, s, r in [(v, s, r) for v in E7_VARIANTS for r in REPEATS for s in SEEDS][a.part::a.jobs]:
        cfg = dict(E7_VARIANTS[v], no_std=True)
        cv_run(f"P5D-E7-{v}-v1", ["hdr", "ecgf"], "fitter", s, r, fitter=mlp5.fitter(cfg), command=f"{CMD} --jobs {a.jobs} --part {a.part}",
               extra_cfg=dict(variant=v, change=E7_VARIANTS[v], baseline="P5A-import-mlpfusion-v1"))


if __name__ == "__main__":
    main()
