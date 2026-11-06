"""E6: signal-quality-aware classification (quality from the phase-5 'sq' family, v2; label-free).
  qfilter  MLP fusion trained without training ECGs of quality_score < 0.75 (>= 3 of 12 leads flagged); evaluated on all
  qweight  MLP fusion with per-ECG training weights 0.5 + 0.5 * quality_score (mlp5)
  qgate    nested quality-gated blend w * P(MLP fusion) + (1 - w) * P(header + clinical LR), one w per quality stratum
           (<= 1 vs >= 2 flagged leads), chosen on the training-portion inner OOF (accuracy); applied to outer predictions
Baseline: MLP fusion (qfilter / qweight); vote ensemble and MLP fusion (qgate). The 'sq' features as MLP inputs = E1.
    python -m research_experiments.phase_5.experiments.e6_quality --jobs 3 --part 0
"""
import argparse

import numpy as np
import pandas as pd
import torch

from research_experiments.phase_5.experiments import mlp5
from research_experiments.phase_5.experiments.nested import MLPF, inner_oof
from research_experiments.phase_5.experiments.runner import cv_run
from research_experiments.phase_5.p5common import (FEAT5, IDS, REPEATS, SEEDS, Y, done, fold_vec, load_run, make_X, mark_running, nD,
                                                   save_run, split_indices)

SQ = pd.read_csv(FEAT5 / "sq.csv").set_index("ecg_id").loc[IDS]
QS, NPOOR = SQ.quality_score.values, SQ.n_poor_leads.values
CMD = "python -m research_experiments.phase_5.experiments.e6_quality"


def good_quality(itr, r, k):
    return QS[itr] >= 0.75


def qweight(s, r):
    rid = f"P5D-E6-qweight-v1__s{s}_r{r}"
    if done(rid):
        return
    mark_running(rid, CMD); P = np.zeros((nD, 4))
    for k in range(5):
        itr, iva, ote = split_indices(r, k); fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(["hdr", "ecgf"], fm)
        f, _ = mlp5.fit(X[itr], Y[itr], X[iva], Y[iva], dict(no_std=pre, sample_w=0.5 + 0.5 * QS[itr]), seed=s)
        P[ote] = f(X[ote])
    agg = save_run(rid, P, P.argmax(1), r, dict(kind="quality-weighted MLP fusion", weights="0.5 + 0.5 * quality_score", seed=s))
    print(f"[{rid}] acc {agg['metrics']['accuracy']:.3f}", flush=True)


def qgate(r):
    rid = f"P5D-E6-qgate-v1__s0_r{r}"
    if done(rid):
        return
    mark_running(rid, CMD); fo = fold_vec(r)
    Pm = np.mean([load_run(f"P5A-import-mlpfusion-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
    Ph = load_run(f"P5A-import-hdrclinlr-v1__s0_r{r}")[0]
    P = np.zeros((nD, 4)); params = []; poor = NPOOR >= 2
    for k in range(5):
        (TR, Pmi), (_, Phi) = inner_oof("mlpfusion", r, k), inner_oof("hdrclinlr", r, k)
        ws = {}
        for name, msk in (("good", ~poor[TR]), ("poor", poor[TR])):
            acc = [np.mean((w * Pmi[msk] + (1 - w) * Phi[msk]).argmax(1) == Y[TR][msk]) for w in np.arange(0, 1.01, 0.1)]
            ws[name] = float(np.round(np.arange(0, 1.01, 0.1)[int(np.argmax(acc))], 1))
        te = fo == k
        w = np.where(poor[te], ws["poor"], ws["good"])[:, None]
        P[te] = w * Pm[te] + (1 - w) * Ph[te]; params.append(ws)
    agg = save_run(rid, P, P.argmax(1), r, dict(kind="nested quality-gated blend", strata="n_poor_leads >= 2"), extra=dict(weights_per_fold=params))
    print(f"[{rid}] acc {agg['metrics']['accuracy']:.3f} {params}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    jobs = [("qgate", 0, r) for r in REPEATS] + [(v, s, r) for v in ("qfilter", "qweight") for r in REPEATS for s in SEEDS]
    for v, s, r in jobs[a.part::a.jobs]:
        if v == "qgate":
            qgate(r)
        elif v == "qfilter":
            cv_run("P5D-E6-qfilter-v1", ["hdr", "ecgf"], "p2heads", s, r, cfg=MLPF, train_filter=good_quality, command=CMD,
                   extra_cfg=dict(rule="exclude training ECGs with quality_score < 0.75"))
        else:
            qweight(s, r)


if __name__ == "__main__":
    main()
