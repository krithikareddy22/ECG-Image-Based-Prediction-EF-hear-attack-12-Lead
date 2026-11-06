"""Shared, read-only helpers for the research experiments.

* All outputs go to research_experiments/ (never to data/, results/, models/ or the
  LV_ECG_Dataset_* folders).
* The manifest's `source` column (file paths, which contain patient names) is dropped
  on load and never written anywhere; ECGs are identified by `ecg_id` (year_timestamp)
  and patients by the existing anonymous hash in `patient`.
* Uses the ORIGINAL patient-level split (data/manifest.csv == results/split.csv).
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg_ef.config import CLASSES, HF_TARGET_SENSITIVITY  # noqa: E402
from ecg_ef.metrics import evaluate, fast_macro_f1, pick_hf_threshold  # noqa: E402

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
MOD = EXP / "models"
PLOTS = EXP / "plots"
LOGS = EXP / "logs"
CONFIGS = EXP / "configs"
SEED = 42
N_BOOT = 2000


def _guard(path: Path) -> Path:
    """Refuse to write anywhere outside research_experiments/."""
    path = Path(path).resolve()
    if EXP not in path.parents and path != EXP:
        raise PermissionError(f"refusing to write outside research_experiments/: {path}")
    return path


def seed_all(seed: int = SEED):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_meta() -> pd.DataFrame:
    """Manifest WITHOUT the file-path column (names). Sorted by ecg_id within split order."""
    m = pd.read_csv(ROOT / "data" / "manifest.csv")
    m = m.drop(columns=["source"])
    s = pd.read_csv(ROOT / "results" / "split.csv")
    assert (m.set_index("ecg_id").split == s.set_index("ecg_id").split.reindex(m.ecg_id)).all()
    assert (m.groupby("patient").split.nunique() == 1).all(), "patient leakage across splits"
    return m


# --------------------------------------------------------------------------- metrics
def ovr_auroc(y, probs) -> dict:
    out = {}
    for k, c in enumerate(CLASSES):
        out[c] = float(roc_auc_score(y == k, probs[:, k]))
    out["macro"] = float(np.mean([out[c] for c in CLASSES]))
    return out


def full_metrics(y, probs, hf_threshold=None, pred=None) -> dict:
    """ecg_ef.metrics.evaluate (+ OvR AUROC, error severity). `pred` overrides argmax
    (e.g. ordinal decoding); hf_threshold must come from VALIDATION data."""
    y = np.asarray(y); probs = np.asarray(probs, dtype=float)
    m = evaluate(y, probs, hf_threshold if hf_threshold is not None else 0.5)
    if pred is not None:   # recompute the class metrics for a non-argmax decision rule
        fake = np.eye(4)[pred] * 0.97 + 0.01   # argmax(fake) == pred; HF part recomputed below
        m2 = evaluate(y, fake, 0.5)
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "qwk",
                  "within_one_category", "per_class", "confusion_matrix"):
            m[k] = m2[k]
    else:
        pred = probs.argmax(1)
    m["auroc_ovr"] = ovr_auroc(y, probs)
    err = np.abs(np.asarray(pred) - y)
    m["error_severity"] = {"exact": int((err == 0).sum()), "off_by_1": int((err == 1).sum()),
                           "off_by_2": int((err == 2).sum()), "off_by_3": int((err == 3).sum()),
                           "mean_abs_class_error": float(err.mean())}
    return m


def _stat_fns():
    from sklearn.metrics import cohen_kappa_score

    def bal(y, p):
        return float(np.mean([np.mean(p[y == k] == k) for k in range(4) if np.any(y == k)]))

    def f1_mild(y, p):
        tp = np.sum((p == 1) & (y == 1)); d = np.sum(p == 1) + np.sum(y == 1)
        return float(2 * tp / d) if d else 0.0

    return {
        "accuracy": lambda y, p, s: float(np.mean(p == y)),
        "balanced_accuracy": lambda y, p, s: bal(y, p),
        "macro_f1": lambda y, p, s: fast_macro_f1(y, p),
        "qwk": lambda y, p, s: float(cohen_kappa_score(y, p, weights="quadratic")),
        "mild_f1": lambda y, p, s: f1_mild(y, p),
        "mild_recall": lambda y, p, s: float(np.mean(p[y == 1] == 1)) if np.any(y == 1) else np.nan,
        "macro_auroc": lambda y, p, s: float(np.mean([roc_auc_score(y == k, s[:, k]) for k in range(4)])),
        "mild_auroc": lambda y, p, s: float(roc_auc_score(y == 1, s[:, 1])),
        "hf_auc": lambda y, p, s: float(roc_auc_score(y > 0, 1 - s[:, 0])),
    }


def _cluster_indices(groups, rng, n_boot):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])


def bootstrap_ci(y, probs, groups, pred=None, n_boot=N_BOOT, seed=SEED) -> dict:
    """Patient-cluster bootstrap 95 % CIs (test ECGs of the same patient resampled together)."""
    y = np.asarray(y); probs = np.asarray(probs); pred = probs.argmax(1) if pred is None else np.asarray(pred)
    fns = _stat_fns(); rng = np.random.default_rng(seed)
    vals = {k: [] for k in fns}
    for ii in _cluster_indices(groups, rng, n_boot):
        yy, pp, ss = y[ii], pred[ii], probs[ii]
        if len(np.unique(yy)) < 4:
            continue
        for k, f in fns.items():
            vals[k].append(f(yy, pp, ss))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def paired_bootstrap(y, probs_a, probs_b, groups, pred_a=None, pred_b=None, n_boot=N_BOOT, seed=SEED) -> dict:
    """Difference B - A with patient-cluster paired bootstrap: point estimate, 95 % CI,
    two-sided p (2 x the smaller tail proportion around 0)."""
    y = np.asarray(y)
    pa = probs_a.argmax(1) if pred_a is None else np.asarray(pred_a)
    pb = probs_b.argmax(1) if pred_b is None else np.asarray(pred_b)
    fns = _stat_fns(); rng = np.random.default_rng(seed)
    point = {k: f(y, pb, probs_b) - f(y, pa, probs_a) for k, f in fns.items()}
    diffs = {k: [] for k in fns}
    for ii in _cluster_indices(groups, rng, n_boot):
        yy = y[ii]
        if len(np.unique(yy)) < 4:
            continue
        for k, f in fns.items():
            diffs[k].append(f(yy, pb[ii], probs_b[ii]) - f(yy, pa[ii], probs_a[ii]))
    out = {}
    for k, d in diffs.items():
        d = np.asarray(d)
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        out[k] = {"diff": float(point[k]), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_two_sided": float(min(1.0, p))}
    # exact McNemar on per-ECG correctness
    from scipy.stats import binomtest
    a_ok, b_ok = pa == y, pb == y
    n01, n10 = int(np.sum(~a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))
    out["mcnemar_accuracy"] = {"b_right_a_wrong": n01, "a_right_b_wrong": n10,
                               "p_exact": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}
    return out


# --------------------------------------------------------------------------- saving
def save_run(tag: str, meta_val: pd.DataFrame, pva, meta_test: pd.DataFrame, pte, extra: dict | None = None,
             pred_val=None, pred_test=None, hf_threshold=None, decoding="argmax"):
    """Save predictions + full metrics + CIs + confusion plot for one model."""
    from ecg_ef.train import plot_confusion
    out = _guard(RES / tag); out.mkdir(parents=True, exist_ok=True)
    yv, yt = meta_val.label_idx.values, meta_test.label_idx.values
    thr = hf_threshold if hf_threshold is not None else pick_hf_threshold(yv, pva, HF_TARGET_SENSITIVITY)
    mv = full_metrics(yv, pva, thr, pred_val)
    mt = full_metrics(yt, pte, thr, pred_test)
    mt["ci95"] = bootstrap_ci(yt, pte, meta_test.patient.values, pred_test)
    mt["per_year_accuracy"] = {str(yr): float(np.mean(((pte.argmax(1) if pred_test is None else pred_test)
                                                       == yt)[meta_test.year.values == yr]))
                               for yr in sorted(meta_test.year.unique())}
    summary = dict(tag=tag, decoding=decoding, hf_threshold=float(thr), val=mv, test=mt, **(extra or {}))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, md, p, pr in (("val", meta_val, pva, pred_val), ("test", meta_test, pte, pred_test)):
        df = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES])
        df.insert(0, "ecg_id", md.ecg_id.values); df.insert(1, "true", md.label.values)
        df.insert(2, "pred", [CLASSES[i] for i in (p.argmax(1) if pr is None else pr)])
        df["hf_likelihood"] = 1 - p[:, 0]
        df.to_csv(out / f"{name}_predictions.csv", index=False, float_format="%.6f")
    _guard(PLOTS).mkdir(parents=True, exist_ok=True)
    plot_confusion(mt["confusion_matrix"], f"{tag}\ntest acc {mt['accuracy']:.1%}, bal {mt['balanced_accuracy']:.1%}",
                   PLOTS / f"cm_{tag}.png")
    print(f"[{tag}] TEST acc {mt['accuracy']:.3f} bal {mt['balanced_accuracy']:.3f} mF1 {mt['macro_f1']:.3f} "
          f"QWK {mt['qwk']:.3f} MildF1 {mt['per_class']['Mild']['f1']:.3f} macroAUC {mt['auroc_ovr']['macro']:.3f} "
          f"HF-AUC {mt['hf_screen']['auc']:.3f} | VAL bal {mv['balanced_accuracy']:.3f} mF1 {mv['macro_f1']:.3f}",
          flush=True)
    return summary


def load_run_probs(tag: str, split: str, root: Path | None = None):
    """(ecg_id, probs) of a saved run, from research_experiments/results (default) or the
    original results/ directory (read-only)."""
    base = root if root is not None else RES
    df = pd.read_csv(base / tag / f"{split}_predictions.csv")
    return df.ecg_id.values, df[[f"p_{c}" for c in CLASSES]].values


def env_record() -> dict:
    import sklearn
    import torch
    rec = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
               cpu_threads=torch.get_num_threads(), cuda=torch.cuda.is_available(),
               cpu_count=os.cpu_count(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    for mod in ("timm", "scipy", "xgboost", "cv2", "pymupdf"):
        try:
            rec[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            rec[mod] = "n/a"
    return rec
