"""Regenerate the git-ignored per-ECG predictions of the Phase-5 runs that Phase 6 needs (MISSING_PRIVATE in a fresh
clone) - and nothing else:
  C1  P5A-import-vote-v1                 (Phase-3 vote ensemble; no training: committed Phase-3 OOF)
  C2  P5A-calmlp-protected-v1            (calibrated MLP fusion: seed-mean Phase-2 MLP OOF + nested prior rule)
  C3  P5D-E5-vote-prior-protected-v1     (Phase-5 selected model: vote + nested prior rule, protected objective)
and the base models they are built from (vote members' outer OOF, nested inner OOF of mlpfusion / ecgflindist /
hdrclinlr), which Phase-6 stacking (E) re-uses.
Phase-5 code is imported (registry, nested specs, decision rules); outputs go to phase_6/data/private/ only. No Phase-5
aggregate is written. Every regenerated prediction file is checked against the committed Phase-5 aggregate:
  exact   SHA-256 of the regenerated oof.csv == committed oof_sha256
  close   accuracy, balanced accuracy and macro-F1 within 0.005 of the committed values (required; else STOP)
Report (aggregates only): phase_6/results/baseline_regeneration.json
    python -m research_experiments.phase_6.baselines
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.baseline.import_existing import src_path
from research_experiments.phase_5.experiments.e5_decision import choose
from research_experiments.phase_5.experiments.nested import BASES
from research_experiments.phase_5.registry import IMPORTS
from research_experiments.phase_6.p6common import (IDS, PCOLS, RES6, RUNS6, SEEDS, Y, decide, fold_vec, guard6, inner_oof,
                                                   load_agg5, load_oof_csv, metrics, nD, sha256_file)

IMPORT_IDS = ("P5A-import-vote-v1", "P5A-import-mlpfusion-v1", "P5A-import-ecgflindist-v1", "P5A-import-hdrclinlr-v1")
REPORT = RES6 / "baseline_regeneration.json"


def write_oof(rid, P, pred, r):
    """Same file layout and float format as phase_5.p5common.save_run (so the SHA-256 is comparable)."""
    d = guard6(RUNS6 / rid); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(r)); o["pred"] = pred
    o.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    return d / "oof.csv"


def check(rid, path):
    ref = load_agg5(rid)
    P, pred = load_oof_csv(path)
    m = metrics(Y, P, pred)
    diffs = {k: abs(m[k] - ref["metrics"][k]) for k in ("accuracy", "balanced_accuracy", "macro_f1", "min_recall")}
    return dict(run_id=rid, exact_sha256=sha256_file(path) == ref["oof_sha256"],
                max_abs_diff=max(diffs[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")), diffs=diffs,
                accuracy=m["accuracy"], committed_accuracy=ref["metrics"]["accuracy"],
                within_0_005=all(diffs[k] <= 0.005 for k in ("accuracy", "balanced_accuracy", "macro_f1")))


def load6(rid):
    return load_oof_csv(RUNS6 / rid / "oof.csv")


def main():
    torch.set_num_threads(1)
    t0 = time.time(); out = []
    # 1. imports (no training)
    for eid in IMPORT_IDS:
        pat, seeds = IMPORTS[eid]
        for r in (0, 1, 2):
            for s in seeds:
                rid = f"{eid}__s{s}_r{r}"
                P, pred = load_oof_csv(src_path(pat, s, r))
                out.append(check(rid, write_oof(rid, P, pred, r)))
    # 2. nested inner OOF of the three vote members (needed by C2, C3 and Phase-6 stacking)
    for r in (0, 1, 2):
        for k in range(5):
            for b in ("mlpfusion", "ecgflindist", "hdrclinlr"):
                inner_oof(b, r, k, BASES[b])
        print(f"[inner r{r}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # 3. C2 (calibrated MLP, protected) and C3 (vote + prior rule, protected)
    for r in (0, 1, 2):
        fo = fold_vec(r)
        P_mlp = np.mean([load6(f"P5A-import-mlpfusion-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        P_vote = load6(f"P5A-import-vote-v1__s0_r{r}")[0]
        pred2, pred3 = np.zeros(nD, int), np.zeros(nD, int); a2, a3 = [], []
        for k in range(5):
            TR, Pin = inner_oof("mlpfusion", r, k, BASES["mlpfusion"])
            prior = np.bincount(Y[TR], minlength=4) / len(TR)
            a = choose_alpha(Pin, Y[TR], prior, "protected"); a2.append(a)
            pred2[fo == k] = decide(P_mlp[fo == k], "prior", a, prior)
            inn = [inner_oof(b, r, k, BASES[b]) for b in ("mlpfusion", "ecgflindist", "hdrclinlr")]
            TRv, Pv = inn[0][0], np.mean([x[1] for x in inn], 0)
            assert all((x[0] == TRv).all() for x in inn)
            c = choose(Pv, Y[TRv], prior, "prior", "protected"); a3.append(float(c))
            pred3[fo == k] = decide(P_vote[fo == k], "prior", c, prior)
        for rid, P, pred, al in ((f"P5A-calmlp-protected-v1__s0_r{r}", P_mlp, pred2, a2),
                                 (f"P5D-E5-vote-prior-protected-v1__s0_r{r}", P_vote, pred3, a3)):
            res = check(rid, write_oof(rid, P, pred, r))
            ref = load_agg5(rid)
            res["params_per_fold"] = al
            res["committed_params_per_fold"] = ref.get("alpha_per_fold", ref.get("params_per_fold"))
            out.append(res)
    rep = dict(created=time.strftime("%F %T"), minutes=(time.time() - t0) / 60, n_runs=len(out),
               n_exact=sum(o["exact_sha256"] for o in out), all_within_0_005=all(o["within_0_005"] for o in out),
               max_abs_diff=max(o["max_abs_diff"] for o in out), runs=out)
    guard6(RES6).mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, default=float))
    for o in out:
        print(f"{o['run_id']:<48} exact={o['exact_sha256']!s:<5} acc {o['accuracy']:.4f} (committed {o['committed_accuracy']:.4f})"
              + (f" params {o.get('params_per_fold')} vs {o.get('committed_params_per_fold')}" if "params_per_fold" in o else ""))
    print(f"\n{rep['n_exact']}/{rep['n_runs']} byte-identical; all within 0.005: {rep['all_within_0_005']} "
          f"(max |diff| {rep['max_abs_diff']:.4f}); {rep['minutes']:.1f} min")
    if not rep["all_within_0_005"]:
        sys.exit("STOP: a regenerated baseline does not reproduce its committed aggregate within 0.005")


if __name__ == "__main__":
    main()
