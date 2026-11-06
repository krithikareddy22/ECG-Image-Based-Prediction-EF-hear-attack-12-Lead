"""Phase-2 shared utilities: saving CV predictions, CV metrics, cross-fitting, statistics.
All writes are confined to research_experiments/phase_2/ (guarded)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research_experiments.common import CLASSES, ROOT, _stat_fns, bootstrap_ci, full_metrics, paired_bootstrap
from research_experiments.phase_2.protocol import P2, dev_and_test, folds

RES, MOD, LOCK, PLOTS, LOGS, CONF = (P2 / d for d in ("results", "models", "locked_test", "plots", "logs", "configs"))
PCOLS = [f"p_{c}" for c in CLASSES]


def guard(path) -> Path:
    path = Path(path).resolve()
    if P2.resolve() not in path.parents and path != P2.resolve():
        raise PermissionError(f"refusing to write outside phase_2/: {path}")
    return path


def s3(m: dict) -> float:
    """Pre-registered phase-2 selection score: mean(accuracy, balanced accuracy, macro-F1)."""
    return (m["accuracy"] + m["balanced_accuracy"] + m["macro_f1"]) / 3


def save_cv(tag: str, oof: pd.DataFrame, locked: pd.DataFrame | None, info: dict, overwrite: bool = False):
    """oof: ecg_id, fold, p_* (one row per D ECG; outer-fold predictions);
    locked: ecg_id, p_* (mean of the fold models on T) - stored, NOT scored."""
    d = guard(RES / tag)
    if d.exists() and not overwrite and (d / "oof.csv").exists():
        raise FileExistsError(f"{tag} exists; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    oof.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    if locked is not None:
        guard(LOCK).mkdir(parents=True, exist_ok=True)
        locked.to_csv(LOCK / f"{tag}.csv", index=False, float_format="%.6f")
    (d / "info.json").write_text(json.dumps(info, indent=2, default=float))


def load_oof(tag: str, D: pd.DataFrame | None = None) -> np.ndarray:
    """OOF probabilities aligned to D's order."""
    if D is None:
        D = dev_and_test()[1]
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    return o[PCOLS].values


def load_oof_pred(tag, D):
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    return o["pred"].values if "pred" in o else o[PCOLS].values.argmax(1)


def cv_report(tag: str, D: pd.DataFrame, probs=None, pred=None, repeat: int = 0, write=True) -> dict:
    """Pooled-OOF metrics (+ patient-cluster bootstrap CI) and per-fold metrics on D."""
    probs = load_oof(tag, D) if probs is None else probs
    if pred is None and (RES / tag / "oof.csv").exists():
        o = pd.read_csv(RES / tag / "oof.csv")
        if "pred" in o:
            pred = o.set_index("ecg_id").loc[D.ecg_id].pred.values
    y = D.label_idx.values
    pooled = full_metrics(y, probs, 0.5, pred)
    pooled["ci95"] = bootstrap_ci(y, probs, D.patient.values, pred)
    pooled["s3"] = s3(pooled)
    f = folds(repeat); f = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
    per = []
    pr = probs.argmax(1) if pred is None else pred
    for k in range(5):
        m = full_metrics(y[f == k], probs[f == k], 0.5, pr[f == k])
        per.append(dict(fold=k, accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                        mild_f1=m["per_class"]["Mild"]["f1"], qwk=m["qwk"], macro_auroc=m["auroc_ovr"]["macro"], s3=s3(m)))
    out = dict(tag=tag, pooled=pooled, per_fold=per,
               fold_mean={k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "fold"},
               fold_sd={k: float(np.std([p[k] for p in per], ddof=1)) for k in per[0] if k != "fold"})
    if write:
        guard(RES / tag).mkdir(parents=True, exist_ok=True)
        (RES / tag / "cv_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def short(m: dict) -> str:
    p = m["pooled"]
    return (f"acc {p['accuracy']:.3f} bal {p['balanced_accuracy']:.3f} mF1 {p['macro_f1']:.3f} "
            f"Mild {p['per_class']['Mild']['f1']:.3f} QWK {p['qwk']:.3f} AUC {p['auroc_ovr']['macro']:.3f} S3 {p['s3']:.3f}")


def nadeau_bengio(da: np.ndarray, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on per-fold differences."""
    da = np.asarray(da, float); k = len(da)
    var = da.var(ddof=1)
    t = da.mean() / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else np.inf * np.sign(da.mean())
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return dict(mean_diff=float(da.mean()), t=float(t), p=float(p))


def compare_cv(tag_a: str, tag_b: str, D: pd.DataFrame, pa=None, pb=None, preds=(None, None)) -> dict:
    """B - A on pooled OOF (patient-cluster paired bootstrap) + per-fold corrected t-test."""
    pa = load_oof(tag_a, D) if pa is None else pa
    pb = load_oof(tag_b, D) if pb is None else pb
    y = D.label_idx.values
    out = paired_bootstrap(y, pa, pb, D.patient.values, preds[0], preds[1])
    f = folds(0); f = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
    fns = _stat_fns()
    da_ = preds[0] if preds[0] is not None else pa.argmax(1)
    db_ = preds[1] if preds[1] is not None else pb.argmax(1)
    for k_ in ("accuracy", "balanced_accuracy", "macro_f1", "mild_f1"):
        d = [fns[k_](y[f == k], db_[f == k], pb[f == k]) - fns[k_](y[f == k], da_[f == k], pa[f == k]) for k in range(5)]
        out[k_]["nadeau_bengio"] = nadeau_bengio(np.array(d), n_train=int(0.8 * 0.8 * len(D)), n_test=int(0.2 * len(D)))
    return out


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def fold_vector(D, repeat=0):
    f = folds(repeat); return f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
