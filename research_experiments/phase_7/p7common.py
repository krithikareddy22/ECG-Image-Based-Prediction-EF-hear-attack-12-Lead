"""Phase 7 shared code (PHASE7_PLAN.md). Development set only (837 ECGs); Phase 2 folds; no recurrent models.
Per-ECG predictions only in phase_7/data/private (git-ignored); aggregates in phase_7/results/experiments.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import cells, corr_ci, corr_t, holm  # noqa: F401
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

ROOT = B.ROOT
P7 = ROOT / "research_experiments" / "phase_7"
PRIV7 = P7 / "data" / "private"; RUNS7 = PRIV7 / "runs"
RES7 = P7 / "results"; EXP7 = RES7 / "experiments"; PLOTS7 = P7 / "plots"
Y, IDS, PAT, nD, CLASSES = B.Y, B.IDS, B.PAT, B.nD, B.CLASSES
SEEDS = (42, 43, 44)
REF_MEMBER = "P6A-mlpfusion-B1-v1__s42_r{r}"       # pilot reference: Phase 2 MLP fusion [header || f6], seed 42
REF_C3 = "P6A-c3-B1-v1__s0_r{r}"                   # confirmatory reference: C3 (0.562)


def guard7(path) -> Path:
    path = Path(path).resolve()
    if P7.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_7/: {path}")
    return path


_V = None


def verified():
    """(mask, EF): clean-link ECGs whose nearest-echo EF is used as continuous supervision (NF set; 366)."""
    global _V
    if _V is None:
        from research_experiments.phase_6b_threshold_sampling_lstm import mmclean as M
        _V = (~M.flags().NF.values, B.table().ef.values.astype(float))
    return _V


def done(rid):
    return (EXP7 / f"{rid}.json").exists() and (RUNS7 / f"{rid}.npz").exists()


def save(rid, P, pred, r, config, extra=None):
    guard7(RUNS7).mkdir(parents=True, exist_ok=True); guard7(EXP7).mkdir(parents=True, exist_ok=True)
    np.savez(RUNS7 / f"{rid}.npz", P=P.astype(np.float32), pred=pred.astype(np.int8))
    from research_experiments.phase_6.p6common import cells as cells6
    agg = dict(run_id=rid, repeat=r, config=config, created=time.strftime("%F %T"), git=B.git_head(), n_ecgs=nD,
               metrics=B.metrics(Y, P, pred), confusion=B.confusion(Y, pred).tolist(), per_fold=cells6(P, pred, r), **(extra or {}))
    (EXP7 / f"{rid}.json").write_text(json.dumps(agg, indent=1, default=float))
    return agg


def load(rid):
    if rid.startswith("P6A-") or rid.startswith("P6B-"):
        return B.load_run(rid)
    if rid.startswith("P5"):                                   # Phase 5 runs (e.g. C2) regenerated under phase_6/
        from research_experiments.phase_6.p6common import RUNS6
        return B.load_oof_csv(RUNS6 / rid / "oof.csv")
    z = np.load(RUNS7 / f"{rid}.npz"); return z["P"].astype(float), z["pred"].astype(int)


def summary(rids_by_repeat, mask=None, y=None):
    """rids_by_repeat: list over repeats of list of run ids (averaged probabilities; argmax unless one rule run)."""
    y = Y if y is None else y; mask = np.ones(nD, bool) if mask is None else mask
    ms, cl, cm = [], [], []
    for r, rids in enumerate(rids_by_repeat):
        Ps = [load(x) for x in rids]
        P = np.mean([p[0] for p in Ps], 0); pred = Ps[0][1] if len(Ps) == 1 else P.argmax(1)
        ms.append(B.metrics(y[mask], P[mask], pred[mask])); cl += [c["accuracy"] for c in cells(P, pred, r, y=y, mask=mask)]
        cm.append(B.confusion(y[mask], pred[mask]))
    keys = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1")]
    o = {k: float(np.mean([x[k] for x in ms])) for k in keys}
    o["min_recall"] = min(o[f"recall_{c}"] for c in CLASSES); o["admissible"] = bool(o["min_recall"] >= 0.15)
    o["accuracy_ci95_corrected"] = corr_ci(cl); o["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in ms]
    o["confusion_mean"] = np.mean(cm, 0).round(1).tolist()
    return o


def paired(a_by_repeat, b_by_repeat, metric):
    d, dr = [], []
    for r, (ra, rb) in enumerate(zip(a_by_repeat, b_by_repeat)):
        def pp(rids):
            Ps = [load(x) for x in rids]; P = np.mean([p[0] for p in Ps], 0)
            return P, (Ps[0][1] if len(Ps) == 1 else P.argmax(1))
        Pa, pa = pp(ra); Pb, pb = pp(rb)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r), cells(Pb, pb, r))]
        dr.append(B.metrics(Y, Pa, pa)[metric] - B.metrics(Y, Pb, pb)[metric])
    _, p = corr_t(d)
    return dict(metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr], ci95_corrected_delta=corr_ci(d),
                p_corrected_t=p, same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def gate(cand, ref):
    """Pilot gate (PHASE7_PLAN.md section 3): acc >= +0.010 or Mild F1 >= +0.03, and macro-F1 not lower by > 0.010."""
    da, dm, df = cand["accuracy"] - ref["accuracy"], cand["f1_Mild"] - ref["f1_Mild"], cand["macro_f1"] - ref["macro_f1"]
    return dict(d_accuracy=da, d_f1_Mild=dm, d_macro_f1=df, passes=bool((da >= 0.010 or dm >= 0.03) and df >= -0.010))
