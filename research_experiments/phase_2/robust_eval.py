"""Robust validation of the phase-2 candidates on the DEVELOPMENT set only (locked test untouched).

For every model: all available runs (CV repeat r in {0,1,2} x seed s), each with pooled out-of-fold (OOF)
predictions for all 837 development ECGs.
  * mean +- SD over runs of each pooled-OOF metric
  * corrected repeated-CV 95 % CI (Bouckaert & Frank 2004): x_rk = fold-level metric (averaged over seeds),
    CI = mean +- t_{RK-1} * sqrt((1/RK + n_test/n_train) * var(x_rk))
  * 3-seed ensembles (probability mean of the seeds within a repeat) as separate models
Paired comparisons B - A on matched (repeat, fold) cells: corrected repeated-CV t-test (primary), patient-
cluster paired bootstrap on repeat-0 pooled OOF (secondary), Holm within metric.

    python -m research_experiments.phase_2.robust_eval
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from research_experiments.common import CLASSES, paired_bootstrap
from research_experiments.phase_2.p2common import LOCK, PCOLS, RES, guard, holm, save_cv
from research_experiments.phase_2.protocol import dev_and_test, folds

META, D, T = dev_and_test()
Y = D.label_idx.values
N_TR, N_TE = 535, 167
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "f1_Normal", "f1_Mild", "f1_Moderate", "f1_Severe",
           "recall_Mild", "qwk", "macro_auroc", "s3"]


def metrics(y, p, pred):
    f1 = f1_score(y, pred, labels=range(4), average=None, zero_division=0)
    rec = [np.mean(pred[y == k] == k) for k in range(4)]
    m = dict(accuracy=float(np.mean(pred == y)), balanced_accuracy=float(np.mean(rec)),
             macro_f1=float(f1.mean()), **{f"f1_{c}": float(f1[k]) for k, c in enumerate(CLASSES)},
             recall_Mild=float(rec[1]), qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
             macro_auroc=float(np.mean([roc_auc_score(y == k, p[:, k]) for k in range(4)])))
    m["s3"] = (m["accuracy"] + m["balanced_accuracy"] + m["macro_f1"]) / 3
    return m


def load_run(tag):
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    P = o[PCOLS].values
    pred = o.pred.values if "pred" in o else P.argmax(1)
    info = json.loads((RES / tag / "info.json").read_text())
    return P, pred, int(info.get("repeat", 0)), int(info.get("seed", 42))


def fold_of(r):
    f = folds(r); return f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values


def model_cells(tags):
    """-> runs list, cell table {(r,k): {metric: value averaged over seeds}}, repeat-0 probs (seed mean)."""
    runs = []; cells = {}
    by_r = {}
    for t in tags:
        if not (RES / t / "oof.csv").exists():
            continue
        P, pred, r, s = load_run(t)
        runs.append(dict(tag=t, repeat=r, seed=s, **metrics(Y, P, pred)))
        by_r.setdefault(r, []).append((P, pred))
        fo = fold_of(r)
        for k in range(5):
            mk = metrics(Y[fo == k], P[fo == k], pred[fo == k])
            cells.setdefault((r, k), []).append(mk)
    cells = {c: {m: float(np.mean([v[m] for v in vs])) for m in METRICS} for c, vs in cells.items()}
    p0 = None
    if 0 in by_r:
        p0 = np.mean([p for p, _ in by_r[0]], 0)
    return runs, cells, p0, by_r


def corrected_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return [np.nan, np.nan]
    se = np.sqrt((1 / n + N_TE / N_TR) * x.var(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * se
    return [float(x.mean() - h), float(x.mean() + h)]


def corrected_t(d):
    d = np.asarray(d, float); n = len(d)
    v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + N_TE / N_TR) * v) if v > 0 else 0.0
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1)), n


def make_seed_ensembles(name, tags):
    """Mean of the seeds' OOF/locked probabilities within each repeat -> saved as new runs."""
    out = []
    groups = {}
    for t in tags:
        if (RES / t / "oof.csv").exists():
            groups.setdefault(load_run(t)[2], []).append(t)
    for r, ts in groups.items():
        if len(ts) < 2:
            continue
        tag = f"ens{len(ts)}__{name}_r{r}"
        P = np.mean([load_run(t)[0] for t in ts], 0)
        PT = np.mean([pd.read_csv(LOCK / f"{t}.csv").set_index("ecg_id").loc[T.ecg_id][PCOLS].values for t in ts], 0)
        oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", fold_of(r))
        locked = pd.DataFrame(PT, columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values)
        save_cv(tag, oof, locked, dict(members=ts, repeat=r, seed=-1, kind="seed ensemble"), overwrite=True)
        out.append(tag)
    return out


def summarise(models: dict, comparisons: list, out_name: str):
    rows, cellmap, p0map, runrows = [], {}, {}, []
    for name, tags in models.items():
        runs, cells, p0, _ = model_cells(tags)
        if not runs:
            continue
        cellmap[name], p0map[name] = cells, p0
        R = pd.DataFrame(runs); runrows += [dict(model=name, **r) for r in runs]
        row = dict(model=name, n_runs=len(R), repeats=sorted(R.repeat.unique().tolist()), seeds=sorted(R.seed.unique().tolist()))
        for m in METRICS:
            row[f"{m}_mean"] = float(R[m].mean()); row[f"{m}_sd"] = float(R[m].std(ddof=1)) if len(R) > 1 else np.nan
            row[f"{m}_ci"] = corrected_ci([v[m] for v in cells.values()])
        rows.append(row)
    S = pd.DataFrame(rows)
    comp = []
    for a, b in comparisons:
        if a not in cellmap or b not in cellmap:
            continue
        common = sorted(set(cellmap[a]) & set(cellmap[b]))
        row = dict(reference=a, candidate=b, n_cells=len(common), repeats=sorted({c[0] for c in common}))
        for m in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "qwk"):
            d = [cellmap[b][c][m] - cellmap[a][c][m] for c in common]
            row[f"{m}_diff"], row[f"{m}_p_corr_t"], _ = corrected_t(d)
        if p0map[a] is not None and p0map[b] is not None:
            bt = paired_bootstrap(Y, p0map[a], p0map[b], D.patient.values)
            for m in ("accuracy", "balanced_accuracy", "macro_f1", "mild_f1"):
                row[f"{m}_boot_r0_ci"] = bt[m]["ci95"]; row[f"{m}_boot_r0_p"] = bt[m]["p_two_sided"]
        comp.append(row)
    C = pd.DataFrame(comp)
    if len(C):
        for m in ("accuracy", "balanced_accuracy", "macro_f1"):
            C[f"{m}_p_corr_t_holm"] = holm(C[f"{m}_p_corr_t"].values)
    guard(RES).mkdir(exist_ok=True, parents=True)
    S.to_csv(RES / f"{out_name}_summary.csv", index=False)
    C.to_csv(RES / f"{out_name}_comparisons.csv", index=False)
    pd.DataFrame(runrows).to_csv(RES / f"{out_name}_runs.csv", index=False)
    return S, C


def fmt(S, C):
    lines = ["| Model | runs (repeats x seeds) | Accuracy | Balanced acc. | Macro-F1 | F1 N / Mi / Mo / S | Mild recall | QWK | S3 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in S.itertuples():
        g = lambda m: f"{getattr(r, m + '_mean'):.3f} ± {0 if np.isnan(getattr(r, m + '_sd')) else getattr(r, m + '_sd'):.3f} [{getattr(r, m + '_ci')[0]:.3f}, {getattr(r, m + '_ci')[1]:.3f}]"
        lines.append(f"| {r.model} | {r.n_runs} ({len(r.repeats)}x{len(r.seeds)}) | {g('accuracy')} | {g('balanced_accuracy')} | {g('macro_f1')} | "
                     f"{r.f1_Normal_mean:.3f} / {r.f1_Mild_mean:.3f} / {r.f1_Moderate_mean:.3f} / {r.f1_Severe_mean:.3f} | "
                     f"{r.recall_Mild_mean:.3f} | {r.qwk_mean:.3f} ± {0 if np.isnan(r.qwk_sd) else r.qwk_sd:.3f} | {r.s3_mean:.3f} |")
    lines += ["", "| Candidate vs reference | cells | Δ accuracy (p corr-t, Holm) | Δ balanced acc. (p, Holm) | Δ macro-F1 (p, Holm) | Δ Mild F1 (p) | Δ Mild recall | Δ QWK (p) | bootstrap r0: Δacc 95% CI (p) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        bt = (f"[{r.accuracy_boot_r0_ci[0]:+.3f}, {r.accuracy_boot_r0_ci[1]:+.3f}] ({r.accuracy_boot_r0_p:.3f})"
              if isinstance(getattr(r, "accuracy_boot_r0_ci", None), list) else "–")
        lines.append(f"| {r.candidate} vs {r.reference} | {r.n_cells} | {r.accuracy_diff:+.3f} ({r.accuracy_p_corr_t:.3f}, {r.accuracy_p_corr_t_holm:.3f}) | "
                     f"{r.balanced_accuracy_diff:+.3f} ({r.balanced_accuracy_p_corr_t:.3f}, {r.balanced_accuracy_p_corr_t_holm:.3f}) | "
                     f"{r.macro_f1_diff:+.3f} ({r.macro_f1_p_corr_t:.3f}, {r.macro_f1_p_corr_t_holm:.3f}) | {r.f1_Mild_diff:+.3f} ({r.f1_Mild_p_corr_t:.3f}) | "
                     f"{r.recall_Mild_diff:+.3f} | {r.qwk_diff:+.3f} ({r.qwk_p_corr_t:.3f}) | {bt} |")
    return "\n".join(lines)
