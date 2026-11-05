"""Phase-6B shared code: EF-threshold definitions and Normal training-size configurations for FOUR-CLASS LVEF.
Development set only; builds on Phase 6A (label table, LSTM inputs, completed runs are re-used, never repeated).

Label DEFINITIONS (four classes; every model is scored against the original labels AND against its own definition):
  ORIG  original rule (primary reference): Normal M > 52 / F >= 54, Mild >= 41, Moderate 31-<41, Severe < 31
  T50   heart-failure guideline boundary (preserved EF >= 50 %): consistently linked Mild ECGs with EF >= 50 -> Normal
        (= Phase 6A "B2"; 27 ECGs)
  T55   British Society of Echocardiography 2020 normal range (>= 55 %; 50-54 % "borderline low"): consistently linked
        Normal ECGs with EF < 55 -> Mild (10 ECGs)
TRAINING-ONLY configurations (original labels; inner-val and outer test identical to ORIG):
  XUNC  uncertain Normal/Mild borderline ECGs removed from training (= Phase 6A "B3"; 45 ECGs)
  N300 / N270 / N240 / N200  Normal ECGs in the outer-training portion (inner-train + inner-val; 307-308 originally,
        the CV analogue of the original 669-ECG training split with 307 Normal) reduced to the target by removing whole
        Normal patients from inner-train only (seeded order per fold, nested: N200 removals contain N240's, ...)
  A270X Phase 6A A270 (inner-train Normal EXPANDED to 270 from inner-val) - reported, not re-run
Removed training rows are removed from every fit (outer members, nested inner models, decision-rule fitting).
Storage: phase_6b_threshold_sampling_lstm/results/experiments (aggregates) and data/private (per-ECG, git-ignored).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_6a_normal_mild_lstm import p6acommon as A
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (  # noqa: F401
    CLASSES, IDS, PAT, PCOLS, ROOT, SEEDS, Y, config_hash, confusion, corr_ci, corr_t, decide, fold_vec, forbidden_modules_loaded,
    git_head, holm, load_oof_csv, make_X, metrics, nD, privacy_scan, sha256_file, split_indices)

P6B = ROOT / "research_experiments" / "phase_6b_threshold_sampling_lstm"
PRIV = P6B / "data" / "private"
RUNS, FOLDC, INNER, CKPT, LOGS, RUN_MARK = (PRIV / x for x in ("runs", "fold_cache", "inner", "checkpoints", "logs", "running"))
RES = P6B / "results"; EXP = RES / "experiments"; CONF = P6B / "configs"; PLOTS = P6B / "plots"
DEFINITIONS = ("ORIG", "T50", "T55")
TRAINING_CONFIGS = ("XUNC", "N300", "N270", "N240", "N200")
SIZES = {"N300": 300, "N270": 270, "N240": 240, "N200": 200}
REUSED_6A = {"ORIG": "B1", "T50": "B2", "XUNC": "B3", "A270X": "A270"}     # existing-model runs already done in 6A


def guard(path) -> Path:
    path = Path(path).resolve()
    if P6B.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_6b_threshold_sampling_lstm/: {path}")
    return path


_T = None


def table():
    global _T
    if _T is None:
        T = A.label_table().copy()
        T["t55_move"] = (T.label.values == "Normal") & (T.derived.values == "Normal") & (T.ef.values < 55)
        T["y_t50"] = T.y_b2.values.astype(int)
        T["y_t55"] = np.where(T.t55_move, 1, T.y.values).astype(int)
        _T = T
    return _T


def y_def(cfg):
    """Labels of a definition (training labels for T50 / T55; original otherwise)."""
    T = table()
    return T.y_t50.values if cfg == "T50" else T.y_t55.values if cfg == "T55" else Y


def n_size_removed(r, k, target):
    """Normal ECGs removed from inner-train so that the outer-training portion holds `target` Normal ECGs."""
    itr, iva, _ = split_indices(r, k)
    n_out = int((Y[itr] == 0).sum() + (Y[iva] == 0).sum())
    pats = pd.unique(PAT[itr])
    rng = np.random.default_rng(6200 + 10 * r + k)                  # same order for every target -> nested removals
    removed = []
    for p in rng.permutation(pats):
        if n_out - len(removed) <= target:
            break
        rows = itr[PAT[itr] == p]
        if (Y[rows] != 0).any():                                    # only patients whose inner-train ECGs are all Normal
            continue
        if n_out - len(removed) - len(rows) >= target:
            removed.extend(rows.tolist())
    rem = np.zeros(nD, bool); rem[removed] = True
    assert n_out - rem.sum() == target, (r, k, target)
    return rem


def train_exclude(cfg, r, k):
    """Boolean mask over all ECGs that may not be used for ANY fitting in outer fold k of repeat r."""
    if cfg == "XUNC":
        return table().b3_uncertain.values.copy()
    if cfg in SIZES:
        return n_size_removed(r, k, SIZES[cfg])
    return np.zeros(nD, bool)


def split(cfg, r, k):
    itr, iva, ote = split_indices(r, k)
    ex = train_exclude(cfg, r, k)
    return itr[~ex[itr]], iva, ote


# ------------------------------------------------------------------ storage
def agg_path(rid):
    return EXP / f"{rid}.json"


def oof_path(rid):
    return RUNS / rid / "oof.csv"


def done(rid):
    a, o = agg_path(rid), oof_path(rid)
    return a.exists() and o.exists() and json.loads(a.read_text()).get("oof_sha256") in (None, sha256_file(o))


def mark_running(rid, cmd):
    guard(RUN_MARK).mkdir(parents=True, exist_ok=True)
    (RUN_MARK / f"{rid}.json").write_text(json.dumps(dict(run_id=rid, pid=os.getpid(), started=time.strftime("%F %T"), command=cmd)))


def save_run(rid, P, pred, repeat, cfg, config, extra=None):
    if agg_path(rid).exists():
        raise FileExistsError(f"{rid} already completed")
    d = guard(RUNS / rid); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    o.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    from research_experiments.phase_6.p6common import cells
    agg = dict(run_id=rid, repeat=repeat, config_name=cfg, config=config, config_hash=config_hash(config), created=time.strftime("%F %T"),
               git=git_head(), n_ecgs=nD, metrics=metrics(Y, P, pred), confusion=confusion(Y, pred).tolist(),
               per_fold=cells(P, pred, repeat), oof_sha256=sha256_file(d / "oof.csv"), **(extra or {}))
    for dname in ("T50", "T55"):
        agg[f"metrics_vs_{dname}_labels"] = metrics(y_def(dname), P, pred)
        agg[f"confusion_vs_{dname}_labels"] = confusion(y_def(dname), pred).tolist()
    guard(EXP).mkdir(parents=True, exist_ok=True)
    agg_path(rid).write_text(json.dumps(agg, indent=1, default=float))
    f = RUN_MARK / f"{rid}.json"
    if f.exists():
        f.unlink()
    return agg


def run_path(rid):
    """Per-ECG predictions of a 6B run, or of the re-used Phase 6A / Phase 6 run with the same content."""
    for base in (RUNS, A.RUNS, ROOT / "research_experiments" / "phase_6" / "data" / "private" / "runs"):
        if (base / rid / "oof.csv").exists():
            return base / rid / "oof.csv"
    raise FileNotFoundError(rid)


def load_run(rid):
    return load_oof_csv(run_path(rid))
