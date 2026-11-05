"""Phase-3 robust comparison across seeds x CV repeats (development set only; locked test untouched).

Runs are read from phase_2/results or phase_3/results (read-only). For each model:
  mean +- SD over runs of pooled-OOF metrics (accuracy, balanced accuracy, macro-F1, per-class precision /
  recall / F1, Mild recall, QWK, S3), corrected repeated-CV 95 % CI (Bouckaert & Frank 2004).
Paired comparisons on matched (repeat, fold) cells (seed-averaged): corrected repeated-CV t-test (primary),
patient-cluster paired bootstrap on repeat-0 seed-averaged probabilities (secondary), Holm within metric.

    python -m research_experiments.phase_3.p3_robust
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, roc_auc_score

from research_experiments.common import CLASSES, paired_bootstrap
from research_experiments.phase_2.p2common import PCOLS, holm
from research_experiments.phase_2.protocol import dev_and_test, folds
from research_experiments.phase_3.p3common import P2RES, RES3

META, D, T = dev_and_test()
Y = D.label_idx.values
N_TR, N_TE = 535, 167
BASE_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auroc", "s3", "recall_Mild", "f1_Mild"]


def metrics(y, p, pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    m = dict(accuracy=float(np.mean(pred == y)), balanced_accuracy=float(rc.mean()), macro_f1=float(f1.mean()),
             qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
             macro_auroc=float(np.mean([roc_auc_score(y == k, p[:, k]) for k in range(4)])))
    for k, c in enumerate(CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = float(pr[k]), float(rc[k]), float(f1[k])
    m["s3"] = (m["accuracy"] + m["balanced_accuracy"] + m["macro_f1"]) / 3
    return m


ALL_METRICS = list(metrics(np.array([0, 1, 2, 3]), np.eye(4), np.array([0, 1, 2, 3])).keys())


def find(tag):
    for base in (RES3, P2RES):
        if (base / tag / "oof.csv").exists():
            return base / tag
    return None


def load(tag):
    d = find(tag)
    o = pd.read_csv(d / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    P = o[PCOLS].values; pred = o.pred.values if "pred" in o else P.argmax(1)
    info = json.loads((d / "info.json").read_text())
    return P, pred, int(info.get("repeat", 0)), int(info.get("seed", 42))


def fold_of(r):
    f = folds(r); return f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values


def cells_of(tags):
    runs, cells, by_r = [], {}, {}
    for t in tags:
        if find(t) is None:
            continue
        P, pred, r, s = load(t)
        runs.append(dict(tag=t, repeat=r, seed=s, **metrics(Y, P, pred))); by_r.setdefault(r, []).append(P)
        fo = fold_of(r)
        for k in range(5):
            cells.setdefault((r, k), []).append(metrics(Y[fo == k], P[fo == k], pred[fo == k]))
    cells = {c: {m: float(np.mean([v[m] for v in vs])) for m in ALL_METRICS} for c, vs in cells.items()}
    p0 = np.mean(by_r[0], 0) if 0 in by_r else None
    return runs, cells, p0, by_r


def ens_metrics(by_r):
    """Seed-ensemble (probability mean within each repeat), metrics averaged over repeats."""
    out = [metrics(Y, np.mean(ps, 0), np.mean(ps, 0).argmax(1)) for r, ps in by_r.items() if len(ps) > 1]
    return {m: float(np.mean([o[m] for o in out])) for m in ALL_METRICS} if out else None


def corr_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return [np.nan, np.nan]
    h = stats.t.ppf(0.975, n - 1) * np.sqrt((1 / n + N_TE / N_TR) * x.var(ddof=1))
    return [float(x.mean() - h), float(x.mean() + h)]


def corr_t(d):
    d = np.asarray(d, float); n = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + N_TE / N_TR) * v) if v > 0 else 0.0
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1))


def seeds_repeats(prefix, seeds=(42, 43, 44), repeats=(0, 1, 2), s42=None, sep="_seed"):
    out = []
    for r in repeats:
        for s in seeds:
            base = (s42 or prefix) if s == 42 else f"{prefix}{sep}{s}"
            out.append(base + (f"_r{r}" if r else ""))
    return out


def models():
    det = lambda t: [t, t + "_r1", t + "_r2"]
    ecgf_mlp = ["ecgf_aligned_f6_mlp", "ecgf_f6_mlp_seed43", "ecgf_f6_mlp_seed44"] + \
        [f"ecgf_f6_mlp_seed{s}_r{r}" for r in (1, 2) for s in (42, 43, 44)]
    return {
        "Header + ResNet-18 (current default, R)": ["R_hdrall+img_wavg"],
        "ResNet-18 image": ["img_resnet18_baseline_recipe"],
        "Header numeric (LR)": det("hdr_numeric_skLR"),
        "Header numeric + clinical (LR)": det("clin_hdrnum+clin_skLR"),
        "ECGFounder frozen (linear)": det("ecgf_aligned_f6_lin"),
        "ECGFounder frozen (linear + distance penalty)": det("ecgf_f6_lin_cedist1"),
        "ECGFounder frozen (MLP)": ecgf_mlp,
        "ECGFounder fine-tuned (last block)": ["ft_b3_base"],
        "2-way MLP fusion: header + ECGFounder": seeds_repeats("ff_hdrnum+ecgf_mlp"),
        "3-way MLP fusion: header + clinical + ECGFounder": seeds_repeats("clin_mlpfusion_hdr+clin+ecgf"),
        "3-way gated fusion (+distance penalty)": seeds_repeats("gate_scalar_hdr+clin+ecgf_dist"),
    }


COMPARISONS = [   # (reference, candidate): family for Holm
    ("2-way MLP fusion: header + ECGFounder", "3-way MLP fusion: header + clinical + ECGFounder"),
    ("Header numeric (LR)", "Header numeric + clinical (LR)"),
    ("2-way MLP fusion: header + ECGFounder", "3-way gated fusion (+distance penalty)"),
    ("ECGFounder frozen (MLP)", "2-way MLP fusion: header + ECGFounder"),
    ("Header numeric (LR)", "2-way MLP fusion: header + ECGFounder"),
    ("Header + ResNet-18 (current default, R)", "2-way MLP fusion: header + ECGFounder"),
    ("Header + ResNet-18 (current default, R)", "3-way MLP fusion: header + clinical + ECGFounder"),
    ("ECGFounder fine-tuned (last block)", "ECGFounder frozen (linear)"),
]


def main(extra_models=None, extra_comparisons=None, out="p3_robust"):
    M = models(); M.update(extra_models or {})
    rows, cellmap, p0map = [], {}, {}
    for name, tags in M.items():
        runs, cells, p0, by_r = cells_of(tags)
        if not runs:
            continue
        R = pd.DataFrame(runs); cellmap[name], p0map[name] = cells, p0
        row = dict(model=name, n_runs=len(R), repeats=sorted(R.repeat.unique().tolist()), seeds=sorted(R.seed.unique().tolist()))
        for m in ALL_METRICS:
            row[f"{m}_mean"] = float(R[m].mean()); row[f"{m}_sd"] = float(R[m].std(ddof=1)) if len(R) > 1 else np.nan
        for m in BASE_METRICS:
            row[f"{m}_ci"] = corr_ci([v[m] for v in cells.values()])
        e = ens_metrics(by_r)
        if e:
            for m in BASE_METRICS:
                row[f"ens_{m}"] = e[m]
        rows.append(row)
    S = pd.DataFrame(rows)
    comp = []
    for a, b in COMPARISONS + list(extra_comparisons or []):
        if a not in cellmap or b not in cellmap:
            continue
        common = sorted(set(cellmap[a]) & set(cellmap[b]))
        r = dict(reference=a, candidate=b, n_cells=len(common), repeats=sorted({c[0] for c in common}))
        for m in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "qwk", "s3"):
            r[f"{m}_diff"], r[f"{m}_p"] = corr_t([cellmap[b][c][m] - cellmap[a][c][m] for c in common])
        if p0map[a] is not None and p0map[b] is not None:
            bt = paired_bootstrap(Y, p0map[a], p0map[b], D.patient.values)
            r["accuracy_boot_r0"] = [bt["accuracy"]["diff"]] + bt["accuracy"]["ci95"] + [bt["accuracy"]["p_two_sided"]]
            r["balanced_accuracy_boot_r0"] = [bt["balanced_accuracy"]["diff"]] + bt["balanced_accuracy"]["ci95"] + [bt["balanced_accuracy"]["p_two_sided"]]
        comp.append(r)
    C = pd.DataFrame(comp)
    for m in ("accuracy", "balanced_accuracy", "macro_f1", "s3"):
        C[f"{m}_p_holm"] = holm(C[f"{m}_p"].values)
    S.to_csv(RES3 / f"{out}_summary.csv", index=False); C.to_csv(RES3 / f"{out}_comparisons.csv", index=False)
    lines = ["| Model | runs | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | Mild recall | QWK | S3 | 3-seed ens. acc / bal / mF1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, m: (f"{r[m + '_mean']:.3f} ± {0 if np.isnan(r[m + '_sd']) else r[m + '_sd']:.3f}"
                      + (f" [{r[m + '_ci'][0]:.3f}, {r[m + '_ci'][1]:.3f}]" if m + "_ci" in r and np.isfinite(r[m + "_ci"][0]) else ""))
    for _, r in S.iterrows():
        ens = f"{r['ens_accuracy']:.3f} / {r['ens_balanced_accuracy']:.3f} / {r['ens_macro_f1']:.3f}" if "ens_accuracy" in r and pd.notna(r.get("ens_accuracy")) else "–"
        lines.append(f"| {r.model} | {r.n_runs} | {f(r, 'accuracy')} | {f(r, 'balanced_accuracy')} | {f(r, 'macro_f1')} | {f(r, 'f1_Mild')} | "
                     f"{r['recall_Mild_mean']:.3f} | {f(r, 'qwk')} | {r['s3_mean']:.3f} | {ens} |")
    lines += ["", "| Candidate vs reference | matched cells | Δ acc (p, Holm) | Δ bal (p, Holm) | Δ mF1 (p, Holm) | Δ Mild F1 (p) | Δ Mild recall (p) | Δ QWK (p) | Δ S3 (p, Holm) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in C.iterrows():
        g = lambda m, h=True: f"{r[m + '_diff']:+.3f} ({r[m + '_p']:.3f}" + (f", {r[m + '_p_holm']:.3f})" if h else ")")
        lines.append(f"| {r.candidate} vs {r.reference} | {r.n_cells} | {g('accuracy')} | {g('balanced_accuracy')} | {g('macro_f1')} | "
                     f"{g('f1_Mild', False)} | {g('recall_Mild', False)} | {g('qwk', False)} | {g('s3')} |")
    pc = ["", "Per-class precision / recall (mean over runs)", "| Model | " + " | ".join(f"{c} P / R" for c in CLASSES) + " |",
          "|---|" + "---|" * 4]
    for _, r in S.iterrows():
        pc.append(f"| {r.model} | " + " | ".join(f"{r[f'precision_{c}_mean']:.3f} / {r[f'recall_{c}_mean']:.3f}" for c in CLASSES) + " |")
    (RES3 / f"{out}_tables.md").write_text("\n".join(lines + pc) + "\n")
    print("\n".join(lines + pc))
    return S, C


if __name__ == "__main__":
    main()
