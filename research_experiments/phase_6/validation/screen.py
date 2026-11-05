"""Stage-1 screen (pre-registered, PHASE6_PLAN.md §3) for every candidate with Stage-1 runs.

Per candidate (seed 42 for stochastic models, or the deterministic run; repeats 0-2):
  r_A        Pearson r of out-of-fold predicted EF vs nearest-echo EF on all 837 ECGs, per repeat, mean over repeats
  macro_f1   four-class macro-F1 (argmax), mean over repeats; plus accuracy, balanced accuracy, recalls, min recall
  delta_r    vs the same-code reference: mean over repeats, the 15 per-fold differences (mean, corrected t p), and a
             patient-cluster paired bootstrap 95 % CI on repeat 0 (2,000 resamples)
  PASS       r_A >= 0.69, or (macro-F1 >= 0.465 and every class recall >= 0.15)
Output (aggregates only): phase_6/results/stage1_screen.csv / .json
    python -m research_experiments.phase_6.validation.screen
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import (CLASSES, PAT, R_PASS, RES6, Y, corr_t, done, ef_target, fold_vec, guard6,
                                                   load_efhat, load_run, metrics)

MF1_PASS = 0.465
# candidate -> (run-id prefix, seed, reference candidate)
CANDIDATES = {
    "R0": ("P6R0-joint-frozen-v1", 42, None),
    "A1": ("P6A1-ft-s6-joint-v1", 42, "R0"),
    "A2": ("P6A2-ft-s56-joint-v1", 42, "R0"),
    "B0": ("P6B0-ecgf-f6-v1", 42, None),
    "B-ECGFM": ("P6B-ecgfm-v1", 42, "B0"),
    "B-F6+ECGFM (exploratory)": ("P6B-f6+ecgfm-v1", 42, "B0"),
    "D2 lead II 2.5 s": ("P6D2-leadII-2p5s-v1", 42, "B0"),
}


def register(name, prefix, seed, ref):
    CANDIDATES[name] = (prefix, seed, ref)


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def cluster_boot_dr(e_c, e_r, ef, n=2000, seed=0):
    ok = np.isfinite(ef)
    pats = pd.Series(np.arange(len(ef))).groupby(PAT).apply(list)
    groups = [np.array(g) for g in pats.values]
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        idx = idx[ok[idx]]
        out.append(_pearson(e_c[idx], ef[idx]) - _pearson(e_r[idx], ef[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def evaluate(name):
    prefix, seed, ref = CANDIDATES[name]
    rids = [f"{prefix}__s{seed}_r{r}" for r in (0, 1, 2)]
    if not all(done(x) for x in rids):
        return None
    ef = ef_target()["ef"]; ok = np.isfinite(ef)
    row = dict(candidate=name, run_prefix=prefix, seed=seed, reference=ref)
    rs, ms, cellr = [], [], []
    for r, rid in enumerate(rids):
        P, pred = load_run(rid); E = load_efhat(rid)
        rs.append(_pearson(E[ok], ef[ok])); ms.append(metrics(Y, P, pred))
        fv = fold_vec(r)
        cellr.append([_pearson(E[(fv == k) & ok], ef[(fv == k) & ok]) for k in range(5)])
    row.update(r_A=float(np.mean(rs)), r_A_per_repeat=[round(x, 4) for x in rs])
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "min_recall"):
        row[k] = float(np.mean([m[k] for m in ms]))
    for c in CLASSES:
        row[f"recall_{c}"] = float(np.mean([m[f"recall_{c}"] for m in ms]))
    if ref is not None:
        rp, rseed, _ = CANDIDATES[ref]
        rr, refcell = [], []
        for r in (0, 1, 2):
            Er = load_efhat(f"{rp}__s{rseed}_r{r}"); fv = fold_vec(r)
            rr.append(_pearson(Er[ok], ef[ok]))
            refcell.append([_pearson(Er[(fv == k) & ok], ef[(fv == k) & ok]) for k in range(5)])
        d = (np.array(cellr) - np.array(refcell)).ravel()
        mean_d, p = corr_t(d)
        row.update(delta_r_vs_ref=float(np.mean(rs) - np.mean(rr)), delta_r_per_repeat=[round(a - b, 4) for a, b in zip(rs, rr)],
                   delta_r_fold_mean=mean_d, delta_r_fold_p=p,
                   delta_r_boot95_r0=cluster_boot_dr(load_efhat(rids[0]), load_efhat(f"{rp}__s{rseed}_r0"), ef))
    row["pass_r"] = row["r_A"] >= R_PASS
    row["pass_f1"] = row["macro_f1"] >= MF1_PASS and row["min_recall"] >= 0.15
    row["PASS"] = bool(row["pass_r"] or row["pass_f1"])
    return row


def main():
    rows = [x for x in (evaluate(n) for n in CANDIDATES) if x is not None]
    guard6(RES6).mkdir(parents=True, exist_ok=True)
    (RES6 / "stage1_screen.json").write_text(json.dumps(rows, indent=1, default=float))
    flat = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows])
    flat.to_csv(RES6 / "stage1_screen.csv", index=False, float_format="%.4f")
    for r in rows:
        print(f"{r['candidate']:<6} r_A {r['r_A']:.3f} {r['r_A_per_repeat']}  mF1 {r['macro_f1']:.3f} acc {r['accuracy']:.3f} "
              f"minR {r['min_recall']:.3f}" + (f"  dr {r['delta_r_vs_ref']:+.3f} (fold p {r['delta_r_fold_p']:.2f}; boot r0 "
              f"[{r['delta_r_boot95_r0'][0]:+.3f}, {r['delta_r_boot95_r0'][1]:+.3f}])" if r.get("reference") else "")
              + f"  -> {'PASS' if r['PASS'] else 'fail'}")


if __name__ == "__main__":
    main()
