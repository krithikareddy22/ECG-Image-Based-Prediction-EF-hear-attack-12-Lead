"""Phase-4 shared code: development set D, cohorts, features, EF <-> class, metrics, storage.

Privacy / storage:
  per-ECG outputs (anything keyed by ecg_id, predicted EF)  -> phase_4/data/private/runs/<tag>/  (git-ignored)
  aggregate metrics only                                    -> phase_4/results/runs/<tag>.json  (committed)
The locked test set T is never predicted or scored in phase 4. Same folds / inner splits as phases 2-3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research_experiments.common import CLASSES, ROOT, bootstrap_ci
from research_experiments.phase_2.cv_features import ALL, D, blockstd, ecgf, header_block, nD
from research_experiments.phase_2.protocol import folds, split_indices  # noqa: F401  (re-exported)
from research_experiments.phase_3.p3_robust import ALL_METRICS, corr_ci, corr_t, metrics  # noqa: F401

P4 = ROOT / "research_experiments" / "phase_4"
PRIV = P4 / "data" / "private"
RUNS = PRIV / "runs"
RES4 = P4 / "results"
AGG = RES4 / "runs"
PLOT4 = P4 / "plots"
REPEATS = (0, 1, 2)
SEEDS = (42, 43, 44)
PCOLS_ = [f"p_{c}" for c in CLASSES]

COH = pd.read_csv(PRIV / "cohorts.csv").set_index("ecg_id").loc[D.ecg_id]
Y = D.label_idx.values
assert (COH.label_idx.values == Y).all() and (COH.patient.values == D.patient.values).all(), "cohort table misaligned"
EF = COH.ef.values.astype(float)
MALE = COH.male.values.astype(int)
YD = COH.derived_idx.values.astype(int)            # class of the nearest-echo EF (same rule as the folder labels)
PAT = D.patient.values
COHORTS = {"A": np.ones(nD, bool), "C": COH.C_14d.values.astype(bool), "G": COH.G_hc.values.astype(bool)}
ASSERT_FOLDER = (Y >= 0).all() and len(Y) == 837


def guard4(path) -> Path:
    path = Path(path).resolve()
    if P4.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_4/: {path}")
    return path


# ------------------------------------------------------------------ EF <-> class (exact recovered rule)
def ef_class(ef, male):
    """0 Normal / 1 Mild / 2 Moderate / 3 Severe.  M: Normal > 52; F: Normal >= 54; Mild >= 41; Moderate >= 31."""
    ef = np.asarray(ef, float); male = np.asarray(male)
    normal = np.where(male == 1, ef > 52, ef >= 54)
    return np.where(normal, 0, np.where(ef >= 41, 1, np.where(ef >= 31, 2, 3)))


def classify_cuts(ef, male, c1, c2, c3):
    """Ordinal decoding with cut-points (Severe < c1 <= Moderate < c2 <= Mild <= c3(M) / < c3+2 (F) < Normal)."""
    ef = np.asarray(ef, float)
    normal = np.where(np.asarray(male) == 1, ef > c3, ef >= c3 + 2)
    return np.where(normal, 0, np.where(ef >= c2, 1, np.where(ef >= c1, 2, 3)))


def gauss_probs(mu, sigma, male):
    """P(class) when EF ~ N(mu, sigma^2), clinical intervals (sex-specific Normal cut-off)."""
    mu = np.asarray(mu, float); up = np.where(np.asarray(male) == 1, 52.0, 54.0)
    cdf = lambda x: stats.norm.cdf((x - mu) / sigma)
    p = np.stack([1 - cdf(up), cdf(up) - cdf(41), cdf(41) - cdf(31), cdf(31)], 1)
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)


def tune_cuts(ef_hat, y, male):
    """Cut-points maximising S3 (accuracy + balanced acc + macro-F1) on a selection set (inner-val)."""
    from research_experiments.phase_2.p2common import s3
    best = None
    lo, hi = np.percentile(ef_hat, 1), np.percentile(ef_hat, 99)
    grid = np.arange(np.floor(lo) - 1, np.ceil(hi) + 2, 1.0)
    for c1 in grid:
        for c2 in grid[grid >= c1 + 1]:
            for c3 in grid[grid >= c2 + 1]:
                pr = classify_cuts(ef_hat, male, c1, c2, c3)
                sc = s3(quick(y, pr))
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, (float(c1), float(c2), float(c3)))
    return best[1], best[0]


def quick(y, pred):
    rc = [np.mean(pred[y == k] == k) if np.any(y == k) else 0.0 for k in range(4)]
    f1 = []
    for k in range(4):
        tp = np.sum((pred == k) & (y == k)); d = np.sum(pred == k) + np.sum(y == k)
        f1.append(2 * tp / d if d else 0.0)
    return dict(accuracy=float(np.mean(pred == y)), balanced_accuracy=float(np.mean(rc)), macro_f1=float(np.mean(f1)))


# ------------------------------------------------------------------ features (phase-2 definitions)
def X_of(feat: str, fit_mask_D: np.ndarray):
    """Feature matrix for the D rows. fit_mask_D: rows of D used for fitting (vocab/medians/standardisation)."""
    fm = np.zeros(len(ALL), bool); fm[:nD] = fit_mask_D
    if feat == "hdr":
        return header_block(fm, "numeric")[:nD], False
    if feat == "ecgf":
        return ecgf("aligned", "f6")[:nD], False
    if feat == "hdr+ecgf":
        return blockstd([header_block(fm, "numeric"), ecgf("aligned", "f6")], fm)[:nD], True
    raise KeyError(feat)


def fold_vec(repeat):
    f = folds(repeat); return f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values


# ------------------------------------------------------------------ metrics
def reg_metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); e = yhat - y
    sl = np.polyfit(yhat, y, 1)[0] if np.std(yhat) > 0 else np.nan
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), pearson=float(stats.pearsonr(y, yhat)[0]),
                spearman=float(stats.spearmanr(y, yhat)[0]), bias=float(np.mean(e)), calib_slope=float(sl))


def binary_auc(p, ef_hat=None):
    """AUROC for EF < 41 (Moderate+Severe) and EF < 31 (Severe) from class probabilities (and from -EF_hat)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, ycls, s_cls in (("ef_lt41", [2, 3], p[:, 2] + p[:, 3]), ("ef_lt31", [3], p[:, 3])):
        for tgt_name, yy, mask in (("all_folder", Y, np.ones(nD, bool)), ("G_derived", YD, COHORTS["G"]),
                                   ("all_echo", YD, np.ones(nD, bool))):
            yb = np.isin(yy[mask], ycls)
            out[f"{name}__{tgt_name}__probs"] = float(roc_auc_score(yb, s_cls[mask]))
            if ef_hat is not None:
                out[f"{name}__{tgt_name}__efhat"] = float(roc_auc_score(yb, -ef_hat[mask]))
    return out


def mild_aucs(p, y=None, mask=None):
    from sklearn.metrics import roc_auc_score
    y = Y if y is None else y; mask = np.ones(nD, bool) if mask is None else mask
    yy, pp = y[mask], p[mask]
    out = dict(mild_ovr=float(roc_auc_score(yy == 1, pp[:, 1])))
    for other, nm in ((0, "mild_vs_normal"), (2, "mild_vs_moderate")):
        m = np.isin(yy, [1, other])
        # score: P(Mild) / (P(Mild) + P(other)) - pairwise discrimination
        s = pp[m, 1] / (pp[m, 1] + pp[m, other] + 1e-12)
        out[nm] = float(roc_auc_score(yy[m] == 1, s))
    return out


def eval_sets(P, pred, ef_hat=None, with_ci=False):
    """All aggregate evaluations of one run (one repeat). Returns dict of named blocks."""
    G = COHORTS["G"]
    out = {"all_folder": metrics(Y, P, pred), "G_folder": metrics(Y[G], P[G], pred[G]),
           "G_derived": metrics(YD[G], P[G], pred[G]), "nonG_folder": metrics(Y[~G], P[~G], pred[~G]),
           "all_derived": metrics(YD, P, pred)}
    for k, m in (("all_folder", np.ones(nD, bool)), ("G_folder", G), ("nonG_folder", ~G)):
        out[k]["n_ecgs"] = int(m.sum()); out[k]["n_patients"] = int(pd.Series(PAT[m]).nunique())
    out["binary_auroc"] = binary_auc(P, ef_hat)
    out["mild_auroc_all_folder"] = mild_aucs(P)
    out["mild_auroc_G_derived"] = mild_aucs(P, YD, G)
    if ef_hat is not None:
        out["regression"] = {c: reg_metrics(EF[m], ef_hat[m]) for c, m in
                             (("A", COHORTS["A"]), ("C", COHORTS["C"]), ("G", G), ("nonG", ~G))}
    if with_ci:
        out["all_folder"]["ci95_cluster_bootstrap"] = bootstrap_ci(Y, P, PAT, pred)
    return out


def per_fold(P, pred, repeat, y=None, mask=None):
    y = Y if y is None else y; mask = np.ones(nD, bool) if mask is None else mask
    fo = fold_vec(repeat)
    return [metrics(y[(fo == k) & mask], P[(fo == k) & mask], pred[(fo == k) & mask]) for k in range(5)]


# ------------------------------------------------------------------ storage
def agg_path(tag):
    return AGG / f"{tag}.json"


def done(tag):
    """A run is complete only if BOTH the committed aggregate and the git-ignored per-ECG predictions exist. In a fresh
    clone the aggregates are present but the private predictions are not, so the run is recomputed (deterministic:
    same folds, seeds and single-thread torch) and the aggregate is rewritten with the same values."""
    return agg_path(tag).exists() and (RUNS / tag / "oof.csv").exists()


def save_run(tag, P, pred, repeat, info, ef_hat=None, extra_cols=None):
    """Private per-ECG OOF + aggregate metrics json (committed)."""
    d = guard4(RUNS / tag); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES]); o.insert(0, "ecg_id", D.ecg_id.values)
    o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    if ef_hat is not None:
        o["ef_hat"] = ef_hat
    for k, v in (extra_cols or {}).items():
        o[k] = v
    o.to_csv(d / "oof.csv", index=False, float_format="%.5f")
    ev = eval_sets(P, pred, ef_hat)
    ev["per_fold_all_folder"] = per_fold(P, pred, repeat)
    agg = dict(tag=tag, repeat=repeat, info=info, eval=ev)
    guard4(AGG).mkdir(parents=True, exist_ok=True)
    agg_path(tag).write_text(json.dumps(agg, indent=1, default=float))
    return agg


def load_run(tag):
    o = pd.read_csv(RUNS / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    P = o[[f"p_{c}" for c in CLASSES]].values
    return P, o.pred.values.astype(int), (o.ef_hat.values if "ef_hat" in o else None), o


def short(ev):
    a = ev["all_folder"]; g = ev["G_folder"]
    s = (f"all acc {a['accuracy']:.3f} bal {a['balanced_accuracy']:.3f} mF1 {a['macro_f1']:.3f} "
         f"Mild {a['f1_Mild']:.3f} S3 {a['s3']:.3f} | G acc {g['accuracy']:.3f} S3 {g['s3']:.3f}")
    if "regression" in ev:
        r = ev["regression"]
        s += f" | MAE A {r['A']['mae']:.2f} G {r['G']['mae']:.2f} R2 A {r['A']['r2']:.3f} G {r['G']['r2']:.3f}"
    return s


def privacy_scan(paths):
    """Files (to be committed) that contain any of the 1,047 real ECG ids / acquisition timestamps or patient hashes."""
    allc = pd.read_csv(PRIV / "cohorts.csv", usecols=["ecg_id", "patient"])
    needles = set(allc.ecg_id) | {e.split("_", 1)[1] for e in allc.ecg_id} | set(allc.patient.astype(str))
    bad = []
    for p in paths:
        p = Path(p)
        if p.suffix not in (".json", ".csv", ".md", ".txt", ".py", ".xlsx"):
            continue
        if p.suffix == ".xlsx":
            txt = "\n".join(df.to_csv() for df in pd.read_excel(p, sheet_name=None, header=None).values())
        else:
            txt = p.read_text(errors="ignore")
        if any(n in txt for n in needles):
            bad.append(str(p))
    return bad
