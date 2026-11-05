"""Phase-5 shared code: development-set-only data, features, folds, metrics, statistics, storage, guards.

LOCKED-TEST PROTECTION
  * The manifest is read once (common.load_meta, which drops the name-bearing path column) and every row with
    split == "test" is dropped immediately; nothing downstream can see a locked-test row, label or feature.
  * This module deliberately does NOT import phase_2.cv_features, phase_2.accuracy_tradeoff, phase_3.p3common,
    phase_3.p3_robust or phase_4.p4common: those load all 1,047 ECGs (incl. the locked test) at import time.
    tests/test_phase5.py asserts this.
  * Folds come from phase_2/configs/folds_D.csv (development ECGs only; asserted).
STORAGE (additive; refuses to write outside phase_5/)
  aggregate results  -> phase_5/results/experiments/<run_id>.json      (committed)
  per-ECG outputs    -> phase_5/data/private/runs/<run_id>/oof.csv     (git-ignored; never committed)
  run markers        -> phase_5/results/running/<run_id>.json          (exists only while a run is executing)
A run is DONE only if both the aggregate and the private predictions exist (see done()).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from research_experiments.common import CLASSES, ROOT, bootstrap_ci, load_meta, paired_bootstrap  # noqa: F401
from ecg_ef.header import design_matrix

P5 = ROOT / "research_experiments" / "phase_5"
PRIV5 = P5 / "data" / "private"
RUNS5, FEAT5 = PRIV5 / "runs", PRIV5 / "features"
RES5 = P5 / "results"
EXP5 = RES5 / "experiments"
RUN_MARK = RES5 / "running"
CONF5 = P5 / "configs"
P2 = ROOT / "research_experiments" / "phase_2"
P3 = ROOT / "research_experiments" / "phase_3"
P4 = ROOT / "research_experiments" / "phase_4"
REPEATS, SEEDS = (0, 1, 2), (42, 43, 44)
DEMO = ["age", "male"]                    # identical to research_experiments/header_ablation.py (copied: no import side effects)
MEAS = ["Rate", "RR", "PR", "QRSD", "QT", "QTcB", "QTcF", "PR_missing", "P_axis_sin", "P_axis_cos",
        "QRS_axis_sin", "QRS_axis_cos", "T_axis_sin", "T_axis_cos"]
PCOLS = [f"p_{c}" for c in CLASSES]
N_TR, N_TE = 535, 167                      # corrected repeated-CV t-test (as in phases 3-4)
RECALL_FLOOR = 0.15                        # pre-registered admissibility floor (PHASE5_PLAN.md)


# ------------------------------------------------------------------ development set only
def _load_dev():
    meta = load_meta()
    D = meta[meta.split != "test"].reset_index(drop=True)
    assert len(D) == 837 and D.patient.nunique() == 794, "development set changed"
    assert (D.split != "test").all()
    return D


D = _load_dev()
nD = len(D)
IDS = D.ecg_id.values
Y = D.label_idx.values.astype(int)
PAT = D.patient.values
_POS = pd.Series(np.arange(nD), index=IDS)


def folds(repeat):
    f = pd.read_csv(P2 / "configs" / "folds_D.csv")
    f = f[f.repeat == repeat]
    assert set(f.ecg_id) <= set(IDS), "fold file contains non-development ECGs"
    return f


def split_indices(repeat, k):
    """-> index arrays into D for inner_train, inner_val, outer_test of outer fold k (phase-2 protocol)."""
    f = folds(repeat); f = f[f.fold == k]
    return tuple(_POS.loc[f[f.role == r].ecg_id].values for r in ("inner_train", "inner_val", "outer_test"))


def fold_vec(repeat):
    f = folds(repeat); return f[f.role == "outer_test"].set_index("ecg_id").loc[IDS].fold.values


# ------------------------------------------------------------------ features (identical to phases 2-4 for D rows)
_H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[IDS].reset_index()
_ECGF = None


def header_numeric(fit_mask):
    """Header numeric block (age, sex, 14 machine measurements): medians from the fitting rows only."""
    X = design_matrix(_H, [])[DEMO + MEAS]
    return X.fillna(X[fit_mask].median()).fillna(0).values.astype(float)


def ecgf_f6():
    global _ECGF
    if _ECGF is None:
        z = np.load(P2 / "models" / "ecgf_aligned_features.npz")
        pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[IDS].values
        _ECGF = z["f6"][pos].astype(np.float32)
    return _ECGF


def blockstd(blocks, fit_mask, weights=None):
    out = []
    for i, B in enumerate(blocks):
        mu, sd = B[fit_mask].mean(0), B[fit_mask].std(0) + 1e-6
        out.append((B - mu) / sd * ((weights[i] if weights else 1.0) / np.sqrt(B.shape[1])))
    return np.concatenate(out, 1)


def feature_family(name):
    """Engineered feature family (git-ignored store, D rows only) -> (matrix with NaN, column names)."""
    F = pd.read_csv(FEAT5 / f"{name}.csv").set_index("ecg_id").loc[IDS]
    V = F.values.astype(float); V[~np.isfinite(V)] = np.nan
    return V, list(F.columns)


def impute(B, fit_mask):
    med = np.nanmedian(B[fit_mask], 0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(B), med[None], B)


def make_X(blocks, fit_mask, weights=None):
    """blocks: list of 'hdr' | 'ecgf' | 'feat:<family>' | 'feat:<a>+<b>'. One block -> raw (model standardises);
    several -> block-standardised concatenation (phase-2 recipe). Returns (X, prestandardised)."""
    mats = []
    for b in blocks:
        if b == "hdr":
            mats.append(header_numeric(fit_mask))
        elif b == "ecgf":
            mats.append(ecgf_f6().astype(float))
        elif b.startswith("ecgfpca:"):                       # PCA of ECGFounder f6 fitted on the fitting rows only
            from sklearn.decomposition import PCA
            E = ecgf_f6().astype(float); mu, sd = E[fit_mask].mean(0), E[fit_mask].std(0) + 1e-6
            pca = PCA(int(b.split(":")[1]), random_state=0).fit(((E - mu) / sd)[fit_mask])
            mats.append(pca.transform((E - mu) / sd))
        elif b.startswith("feat:"):
            mats.append(np.concatenate([impute(feature_family(n)[0], fit_mask) for n in b[5:].split("+")], 1))
        else:
            raise KeyError(b)
    if len(mats) == 1:
        return mats[0], False
    return blockstd(mats, fit_mask, weights), True


# ------------------------------------------------------------------ metrics and statistics
def metrics(y, p, pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    m = dict(accuracy=float(np.mean(pred == y)), balanced_accuracy=float(rc.mean()), macro_f1=float(f1.mean()),
             qwk=float(cohen_kappa_score(y, pred, weights="quadratic")))
    try:
        m["macro_auroc"] = float(np.mean([roc_auc_score(y == k, p[:, k]) for k in range(4)]))
    except ValueError:
        m["macro_auroc"] = float("nan")
    for k, c in enumerate(CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = float(pr[k]), float(rc[k]), float(f1[k])
    m["s3"] = (m["accuracy"] + m["balanced_accuracy"] + m["macro_f1"]) / 3
    m["min_recall"] = float(rc.min())
    return m


def confusion(y, pred):
    return confusion_matrix(y, pred, labels=range(4))


def corr_t(d):
    d = np.asarray(d, float); n = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + N_TE / N_TR) * v) if v > 0 else 0.0
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1))


def corr_ci(x):
    x = np.asarray(x, float); n = len(x)
    h = stats.t.ppf(0.975, n - 1) * np.sqrt((1 / n + N_TE / N_TR) * x.var(ddof=1))
    return [float(x.mean() - h), float(x.mean() + h)]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def cells(P, pred, repeat, y=None, mask=None):
    y = Y if y is None else y; mask = np.ones(nD, bool) if mask is None else mask
    fo = fold_vec(repeat)
    return [metrics(y[(fo == k) & mask], P[(fo == k) & mask], pred[(fo == k) & mask]) for k in range(5)]


# ------------------------------------------------------------------ decision rules (copied from phase_2.accuracy_tradeoff)
ALPHAS = np.round(np.arange(0, 1.51, 0.1), 2)
OFFS = np.arange(-1.0, 1.01, 0.25)
BIASES = [np.array([0.0, a, b, c]) for a in OFFS for b in OFFS for c in OFFS]


def decide(P, method, param, prior):
    if method == "prior":
        return (P * prior[None] ** param).argmax(1)
    return (np.log(np.clip(P, 1e-9, 1)) + param[None]).argmax(1)


# ------------------------------------------------------------------ storage, run markers, provenance
def guard5(path) -> Path:
    path = Path(path).resolve()
    if P5.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_5/: {path}")
    return path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def agg_path(run_id):
    return EXP5 / f"{run_id}.json"


def done(run_id):
    return agg_path(run_id).exists() and (RUNS5 / run_id / "oof.csv").exists()


def mark_running(run_id, cmd):
    guard5(RUN_MARK).mkdir(parents=True, exist_ok=True)
    (RUN_MARK / f"{run_id}.json").write_text(json.dumps(dict(run_id=run_id, pid=os.getpid(), started=time.strftime("%F %T"),
                                                            command=cmd)))


def clear_running(run_id):
    f = RUN_MARK / f"{run_id}.json"
    if f.exists():
        f.unlink()


def save_run(run_id, P, pred, repeat, config, extra=None, overwrite=False):
    """Private per-ECG predictions + committed aggregate. Never overwrites a completed run (versions instead)."""
    if agg_path(run_id).exists() and not overwrite:
        raise FileExistsError(f"{run_id} already completed; create a new version instead of overwriting")
    d = guard5(RUNS5 / run_id); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    o.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    m = metrics(Y, P, pred)
    agg = dict(run_id=run_id, repeat=repeat, config=config, config_hash=config_hash(config), created=time.strftime("%F %T"),
               n_ecgs=nD, n_patients=int(pd.Series(PAT).nunique()), metrics=m, confusion=confusion(Y, pred).tolist(),
               per_fold=cells(P, pred, repeat), oof_sha256=sha256_file(d / "oof.csv"), **(extra or {}))
    guard5(EXP5).mkdir(parents=True, exist_ok=True)
    agg_path(run_id).write_text(json.dumps(agg, indent=1, default=float))
    clear_running(run_id)
    return agg


def load_oof_csv(path):
    o = pd.read_csv(path).set_index("ecg_id")
    assert set(IDS) <= set(o.index), f"{path} misses development ECGs"
    o = o.loc[IDS]
    P = o[PCOLS].values
    pred = o.pred.values.astype(int) if "pred" in o else P.argmax(1)
    return P, pred


def load_run(run_id):
    return load_oof_csv(RUNS5 / run_id / "oof.csv")


def privacy_needles():
    m = load_meta()          # all rows, ids and hashes only (used as search needles, never as data)
    return set(m.ecg_id) | {e.split("_", 1)[1] for e in m.ecg_id} | set(m.patient.astype(str))


def privacy_scan(paths):
    """Files that contain any real ECG id, 14-digit acquisition timestamp or patient hash."""
    needles = privacy_needles(); bad = []
    for p in map(Path, paths):
        if not p.is_file() or p.suffix not in (".json", ".csv", ".md", ".txt", ".py", ".xlsx", ".sh"):
            continue
        txt = ("\n".join(x.to_csv() for x in pd.read_excel(p, sheet_name=None, header=None).values())
               if p.suffix == ".xlsx" else p.read_text(errors="ignore"))
        if any(n in txt for n in needles):
            bad.append(str(p))
    return bad


def git_head():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def forbidden_modules_loaded():
    bad = ("research_experiments.phase_2.cv_features", "research_experiments.phase_2.accuracy_tradeoff",
           "research_experiments.phase_3.p3common", "research_experiments.phase_3.p3_robust", "research_experiments.phase_4.p4common")
    return [m for m in bad if m in sys.modules]
