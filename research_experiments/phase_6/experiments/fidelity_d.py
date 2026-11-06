"""Experiment D (PHASE6_PLAN.md section 4.4): digitisation and input fidelity. Development ECGs only.

D1 (analysis, no training): absolute EF error of R0 (seed-combined, 3 repeats averaged per ECG) and four-class
    correctness of C3 (mean over repeats per ECG) by stratum of the Phase-5 sq-v2 quality features: poor-lead count
    (0 / 1-2 / >= 3), quality-score levels (value bins; the score has many ties, so rank tertiles would split tied ECGs arbitrarily), rhythm-strip poor flag (kurtosis-SQI < 3 or HF > 30 % or BW > 50 %, the
    sq-v2 rule), and missing leads. Spearman correlation of |EF error| with each quality variable; accuracy per stratum
    with Wilson 95 % CI; strata with < 5 ECGs are suppressed.
D2 (one input variant): the aligned input with the 10-s lead II strip replaced by lead II's own 2.5-s window (zeros
    elsewhere), frozen ECGFounder f6 -> the B pipeline (`P6D2-leadII-2p5s-v1`), compared with B0 (same pipeline, full
    strip). Tests only whether the 10-s rhythm strip adds EF information beyond its 2.5-s window.
    python -m research_experiments.phase_6.experiments.fidelity_d --d1
    python -m research_experiments.phase_6.experiments.fidelity_d --d2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from research_experiments.phase_6.p6common import (EMB6, FEAT6, IDS, RES6, SEEDS, Y, ef_target, ecgf_f6, guard6, load_efhat,
                                                   load_run)

SEG = 1250


def proportion_confint(k, n, z=1.959964, method="wilson"):
    """Wilson score interval."""
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def d1():
    ef = ef_target()["ef"]
    E = np.mean([np.mean([load_efhat(f"P6R0-joint-frozen-v1__s{s}_r{r}") for s in SEEDS], 0) for r in (0, 1, 2)], 0)
    err = np.abs(E - ef)
    correct = np.mean([load_run(f"P5D-E5-vote-prior-protected-v1__s0_r{r}")[1] == Y for r in (0, 1, 2)], 0)
    sq = pd.read_csv(FEAT6 / "sq.csv").set_index("ecg_id").loc[IDS]
    rhythm_poor = (sq.rhythm_ksqi < 3) | (sq.rhythm_hf > 0.30) | (sq.rhythm_bw > 0.50)
    strata = {
        "poor_leads": pd.cut(sq.n_poor_leads, [-1, 0, 2, 12], labels=["0", "1-2", ">=3"]).astype(str),
        "quality_score_level": pd.cut(sq.quality_score, [-0.01, 0.75, 0.95, 1.0], labels=["<=0.75", "0.75-0.95", ">0.95 (all leads ok)"]).astype(str),
        "rhythm_strip_poor": rhythm_poor.map({True: "poor", False: "ok"}),
    }
    out = dict(n=len(IDS), correlations_with_abs_EF_error={}, strata={})
    for v in ("n_poor_leads", "quality_score", "ksqi_mean", "hf_mean", "bw_mean", "amp_mean"):
        ok = np.isfinite(err) & np.isfinite(sq[v].values)
        rho, p = spearmanr(sq[v].values[ok], err[ok])
        rc, pc = spearmanr(sq[v].values, correct)
        out["correlations_with_abs_EF_error"][v] = dict(spearman_rho=float(rho), p=float(p), rho_with_C3_correct=float(rc), p_correct=float(pc))
    for name, s in strata.items():
        rows = {}
        for lev in sorted(set(s)):
            m = (s == lev).values
            if m.sum() < 5:
                rows[lev] = "suppressed (<5)"; continue
            k = int(round(correct[m].sum())); lo, hi = proportion_confint(k, m.sum(), method="wilson")
            rows[lev] = dict(n=int(m.sum()), c3_accuracy=float(correct[m].mean()), wilson95=[float(lo), float(hi)],
                             mean_abs_EF_error=float(np.nanmean(err[m])))
        out["strata"][name] = rows
    (guard6(RES6) / "fidelity_d1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def d2():
    from research_experiments.phase_2.ecgf_features import load_model
    from research_experiments.phase_6.experiments.embed_ecgf_cache import dev_aligned_inputs, run_cache
    out = EMB6 / "ecgf_leadII_2p5s.npz"
    if not out.exists():
        torch.set_num_threads(2)
        X, _ = dev_aligned_inputs()
        X[:, 1, SEG:] = 0.0                     # lead II (row 1) keeps only its own first-column window (0-2.5 s)
        _, _, f6 = run_cache(load_model(False), X)
        np.savez(out, ecg_id=np.asarray(IDS, dtype="U32"), E=f6.astype(np.float32))
    from research_experiments.phase_6.experiments import encoders_b as B
    B.EMBS["D2"] = ("P6D2-leadII-2p5s-v1", "ecgf_leadII_2p5s.npz")
    torch.set_num_threads(1)
    for r in (0, 1, 2):
        B.run("D2", 42, r)
    _ = ecgf_f6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d1", action="store_true"); ap.add_argument("--d2", action="store_true")
    a = ap.parse_args()
    if a.d1:
        d1()
    if a.d2:
        d2()


if __name__ == "__main__":
    main()
