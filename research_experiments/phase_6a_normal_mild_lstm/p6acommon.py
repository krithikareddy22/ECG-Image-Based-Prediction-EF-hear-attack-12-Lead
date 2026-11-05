"""Phase-6A shared code (Normal/Mild boundary + LSTM study). Development set only; re-uses phase_5.p5common (via
phase_6.p6common): loader without locked-test rows, folds, features, metrics, statistics, decision rules.

Label configurations (PHASE6A_PLAN.md section 4):
  B1   original folder labels
  B2   Mild -> Normal for clearly defined borderline Mild ECGs (EF >= 50 by a nearest echo that is itself Mild)
  B3   original labels; uncertain Normal/Mild borderline ECGs removed from inner-train only
  A270 original labels; inner-train Normal raised to 270 by moving whole Normal patients from inner-val
Echo EF / link data are used ONLY to define these label configurations and for analysis - never as model inputs.
Storage (refuses to write outside phase_6a_normal_mild_lstm/):
  results/experiments/<run_id>.json  aggregates (committed)      data/private/runs/<run_id>/oof.csv  per-ECG (ignored)
  data/private/fold_cache/, inner/, checkpoints/, logs/, labels.csv  (git-ignored)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import (  # noqa: F401  (re-exported)
    CLASSES, D, IDS, N_TE, N_TR, PAT, PCOLS, REPEATS, ROOT, SEEDS, Y, config_hash, confusion, corr_ci, corr_t, decide,
    fold_vec, forbidden_modules_loaded, git_head, holm, load_oof_csv, make_X, metrics, nD, privacy_scan, sha256_file,
    split_indices)

P6A = ROOT / "research_experiments" / "phase_6a_normal_mild_lstm"
PRIV = P6A / "data" / "private"
RUNS, FOLDC, INNER, CKPT, LOGS = PRIV / "runs", PRIV / "fold_cache", PRIV / "inner", PRIV / "checkpoints", PRIV / "logs"
RES = P6A / "results"
EXP = RES / "experiments"
RUN_MARK = PRIV / "running"
CONF = P6A / "configs"
PLOTS = P6A / "plots"
CONFIGS = ("B1", "B2", "B3", "A270")
B2_BOUNDARY = 50.0
B3_BAND = 2.0
A_TARGET = 270


def guard(path) -> Path:
    path = Path(path).resolve()
    if P6A.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_6a_normal_mild_lstm/: {path}")
    return path


# ------------------------------------------------------------------ label configurations (private, development only)
_LAB = None


def _links_dev():
    L = pd.read_csv(ROOT / "research_experiments" / "phase_4" / "data" / "private" / "ecg_echo_links.csv",
                    usecols=["ecg_id", "split", "ef", "derived"])
    L = L[L.split != "test"].drop(columns="split")
    return L[L.ecg_id.isin(set(IDS))]


def label_table():
    """Per development ECG: original label, nearest-echo EF and link facts, B2 label, B3 'uncertain' flag."""
    global _LAB
    if _LAB is not None:
        return _LAB
    c = pd.read_csv(ROOT / "research_experiments" / "phase_4" / "data" / "private" / "cohorts.csv")
    c = c[c.split != "test"]
    c = c[c.ecg_id.isin(set(IDS))].set_index("ecg_id").loc[IDS]
    assert (c.label_idx.values == Y).all(), "label table does not match the development labels"
    male = c.sex_ecg.astype(str).str.upper().str.startswith("M").values
    bnd = np.where(male, 52.0, 54.0)
    lab = c.label.values
    T = pd.DataFrame(dict(label=lab, y=Y, male=male, ef=c.ef.values, derived=c.derived.values, n_links=c.n_links.values,
                          ef_spread=c.ef_spread_all_links.values, score=c.score.values, abs_days=c.abs_days.values,
                          G=c.G_hc.values.astype(bool), patient=PAT), index=pd.Index(IDS, name="ecg_id"))
    nm = np.isin(lab, ["Normal", "Mild"])
    # B2: clearly defined borderline Mild -> Normal
    T["b2_move"] = (lab == "Mild") & (T.derived.values == "Mild") & (T.ef.values >= B2_BOUNDARY)
    T["y_b2"] = np.where(T.b2_move, 0, Y)
    # B3: uncertain Normal/Mild borderline (band around the sex-specific boundary, or a conflicting Normal/Mild link)
    L = _links_dev()
    other = {"Normal": "Mild", "Mild": "Normal"}
    lab_of = pd.Series(lab, index=IDS)
    conf_ids = set(L[L.apply(lambda r: other.get(lab_of[r.ecg_id]) == r.derived, axis=1)].ecg_id)
    T["b3_band"] = nm & (np.abs(T.ef.values - bnd) <= B3_BAND)
    T["b3_conflict"] = nm & T.index.isin(conf_ids)
    T["b3_uncertain"] = T.b3_band | T.b3_conflict
    T["sex_boundary"] = bnd
    _LAB = T
    return T


def y_train(cfg):
    return label_table().y_b2.values.astype(int) if cfg == "B2" else Y


# ------------------------------------------------------------------ splits per configuration
def a270_split(r, k):
    """Inner-train Normal raised to exactly 270 by moving whole Normal patients from inner-val (seeded)."""
    itr, iva, ote = split_indices(r, k)
    n_norm = int((Y[itr] == 0).sum())
    va_norm_pats = pd.unique(PAT[iva][Y[iva] == 0])
    rng = np.random.default_rng(6270 + 10 * r + k)
    move = []
    for p in rng.permutation(va_norm_pats):
        rows = iva[PAT[iva] == p]
        if (Y[rows] != 0).any():                      # patient also has a non-Normal ECG in inner-val: keep grouping intact
            continue
        if n_norm + len(rows) <= A_TARGET:
            move.extend(rows.tolist()); n_norm += len(rows)
        if n_norm == A_TARGET:
            break
    move = np.array(sorted(move), int)
    itr2 = np.sort(np.r_[itr, move]); iva2 = np.setdiff1d(iva, move)
    assert not set(PAT[itr2]) & set(PAT[iva2]), "patient in both inner-train and inner-val"
    assert not set(PAT[itr2]) & set(PAT[ote]) and not set(PAT[iva2]) & set(PAT[ote])
    return itr2, iva2, ote, dict(normal_inner_train=int((Y[itr2] == 0).sum()), moved_ecgs=int(len(move)),
                                 normal_inner_val=int((Y[iva2] == 0).sum()))


def split(cfg, r, k):
    """-> (inner_train, inner_val, outer_test) index arrays for a configuration."""
    if cfg == "A270":
        return a270_split(r, k)[:3]
    itr, iva, ote = split_indices(r, k)
    if cfg == "B3":
        itr = itr[~label_table().b3_uncertain.values[itr]]
    return itr, iva, ote


# ------------------------------------------------------------------ storage
def agg_path(rid):
    return EXP / f"{rid}.json"


def oof_path(rid):
    return RUNS / rid / "oof.csv"


# MISSING_PRIVATE regeneration (fresh clone / another machine): the committed aggregate stays untouched; the regenerated
# git-ignored per-ECG file is accepted only if it reproduces the committed accuracy, balanced accuracy and macro-F1 within
# REGEN_TOL (floating-point results of retrained models differ in the last digits across CPUs, so the file hash may
# differ). A committed record per run (hashes and metric differences only) documents every regeneration.
REGEN = RES / "regeneration"
REGEN_TOL = 0.005
REGEN_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")


def regen_record(rid, regen_dir=None):
    f = (regen_dir or REGEN) / f"{rid}.json"
    return json.loads(f.read_text()) if f.exists() else None


def accepted_sha(rid, agg, regen_dir=None):
    """Hashes a private prediction file may have: the committed one, or a recorded within-tolerance regeneration."""
    ok = {agg.get("oof_sha256")}
    rec = regen_record(rid, regen_dir)
    if rec and rec.get("within_tolerance"):
        ok.add(rec["regenerated_sha256"])
    return ok


def run_done(a, o, rid, regen_dir=None):
    if not (a.exists() and o.exists()):
        return False
    agg = json.loads(a.read_text())
    return agg.get("oof_sha256") is None or sha256_file(o) in accepted_sha(rid, agg, regen_dir)


def done(rid):
    return run_done(agg_path(rid), oof_path(rid), rid)


def write_oof(path, P, pred, repeat):
    path.parent.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    o.to_csv(path, index=False, float_format="%.6f")
    return path


def regenerate_private(rid, P, pred, repeat, agg_file, oof_file, regen_dir, guard_fn):
    """Aggregate committed, private per-ECG file absent: write the regenerated file + a committed regeneration record.
    Never modifies the aggregate. Raises (STOP) if the committed metrics are not reproduced within REGEN_TOL."""
    import platform

    import torch
    agg = json.loads(agg_file.read_text())
    o = write_oof(guard_fn(oof_file), P, pred, repeat)
    h = sha256_file(o); m = metrics(Y, P, pred)
    diffs = {k: abs(m[k] - agg["metrics"][k]) for k in agg["metrics"] if k in m and np.isfinite(agg["metrics"][k])}
    mx = max(diffs[k] for k in REGEN_KEYS)
    rec = dict(run_id=rid, committed_sha256=agg.get("oof_sha256"), regenerated_sha256=h, exact=h == agg.get("oof_sha256"),
               max_abs_diff_acc_bal_mf1=mx, metric_abs_diff=diffs, tolerance=REGEN_TOL, within_tolerance=bool(mx <= REGEN_TOL),
               regenerated=time.strftime("%F %T"), git=git_head(), platform=platform.platform(), torch=torch.__version__)
    guard_fn(regen_dir).mkdir(parents=True, exist_ok=True)
    (regen_dir / f"{rid}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"[regenerated {rid}] exact={rec['exact']} max |diff| {mx:.4f}", flush=True)
    if not rec["within_tolerance"]:
        raise RuntimeError(f"STOP: regenerated {rid} does not reproduce its committed aggregate within {REGEN_TOL} (max {mx:.4f})")
    return agg


def mark_running(rid, cmd):
    guard(RUN_MARK).mkdir(parents=True, exist_ok=True)
    (RUN_MARK / f"{rid}.json").write_text(json.dumps(dict(run_id=rid, pid=os.getpid(), started=time.strftime("%F %T"), command=cmd)))


def clear_running(rid):
    f = RUN_MARK / f"{rid}.json"
    if f.exists():
        f.unlink()


def save_run(rid, P, pred, repeat, cfg_name, config, extra=None):
    """Private per-ECG predictions + committed aggregate. Metrics against the ORIGINAL labels always; B2 models are
    additionally scored against the B2 labels. Never overwrites a completed run; a committed run whose private file is
    missing (fresh clone) is regenerated under the REGEN_TOL check instead."""
    if agg_path(rid).exists():
        if oof_path(rid).exists():
            raise FileExistsError(f"{rid} already completed")
        out = regenerate_private(rid, P, pred, repeat, agg_path(rid), oof_path(rid), REGEN, guard)
        clear_running(rid)
        return out
    d = guard(RUNS / rid)
    write_oof(d / "oof.csv", P, pred, repeat)
    from research_experiments.phase_6.p6common import cells
    agg = dict(run_id=rid, repeat=repeat, label_config=cfg_name, config=config, config_hash=config_hash(config),
               created=time.strftime("%F %T"), git=git_head(), n_ecgs=nD, metrics=metrics(Y, P, pred),
               confusion=confusion(Y, pred).tolist(), per_fold=cells(P, pred, repeat), oof_sha256=sha256_file(d / "oof.csv"),
               **(extra or {}))
    yb2 = label_table().y_b2.values.astype(int)
    agg["metrics_vs_B2_labels"] = metrics(yb2, P, pred)
    agg["confusion_vs_B2_labels"] = confusion(yb2, pred).tolist()
    guard(EXP).mkdir(parents=True, exist_ok=True)
    agg_path(rid).write_text(json.dumps(agg, indent=1, default=float))
    clear_running(rid)
    return agg


def load_run(rid):
    return load_oof_csv(oof_path(rid))
