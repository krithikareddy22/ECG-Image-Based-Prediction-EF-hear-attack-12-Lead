"""Phase-2 evaluation protocol (fixed BEFORE any phase-2 model was trained).

* Development set D = original train + validation ECGs (837 ECGs). Locked test set T = original
  test split (210 ECGs, 199 patients). T was already exposed in phase 1, so it is NOT untouched;
  in phase 2 it is used at most once, for a pre-registered shortlist, and only with the user's
  authorisation (see configs/preregistration.md). No phase-2 selection uses T.
* Outer CV on D: StratifiedGroupKFold(5, shuffle=True, random_state=2025 + repeat); groups =
  anonymous patient hash; repeat 0 is the primary design (identical folds for all models);
  repeats 1-2 are used for cheap models to measure split variance.
* Inner split inside each outer-training part: StratifiedGroupKFold(5, shuffle=True,
  random_state=100 + fold), first split -> inner-train (~80 %) / inner-val (~20 %). EVERY base
  model in fold k is trained on inner-train_k and selected / early-stopped on inner-val_k
  (identical data for all models).
* Post-hoc steps (fusion weights, stackers, decision-rule calibration) are cross-fitted on the
  out-of-fold (OOF) predictions: the parameters used for fold k are fitted on the OOF
  predictions of the other four folds only.
* Locked-test predictions of a base model = mean of its 5 fold models' probabilities; they are
  written to locked_test/ and NOT scored until the final authorised evaluation.

    python -m research_experiments.phase_2.protocol      # writes configs/folds_D.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.common import ROOT, load_meta

P2 = ROOT / "research_experiments" / "phase_2"
FOLDS = P2 / "configs" / "folds_D.csv"
N_REPEATS = 3


def make_folds() -> pd.DataFrame:
    meta = load_meta()
    D = meta[meta.split != "test"].reset_index(drop=True)
    rows = []
    for r in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2025 + r)
        fold = np.full(len(D), -1); inner_val = np.zeros(len(D), bool)
        for k, (tr, te) in enumerate(outer.split(np.zeros(len(D)), D.label_idx, D.patient)):
            fold[te] = k
        for k in range(5):
            tr = np.flatnonzero(fold != k)
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + k)
            a, b = next(inner.split(np.zeros(len(tr)), D.label_idx.values[tr], D.patient.values[tr]))
            rows.append(pd.DataFrame({"ecg_id": D.ecg_id.values[tr[b]], "repeat": r, "fold": k, "role": "inner_val"}))
            rows.append(pd.DataFrame({"ecg_id": D.ecg_id.values[tr[a]], "repeat": r, "fold": k, "role": "inner_train"}))
            rows.append(pd.DataFrame({"ecg_id": D.ecg_id.values[fold == k], "repeat": r, "fold": k, "role": "outer_test"}))
    f = pd.concat(rows, ignore_index=True)
    # leakage checks
    pat = meta.set_index("ecg_id").patient
    for (r, k), g in f.groupby(["repeat", "fold"]):
        sets = {role: set(pat.loc[g[g.role == role].ecg_id]) for role in ("inner_train", "inner_val", "outer_test")}
        assert not (sets["inner_train"] & sets["inner_val"]) and not (sets["inner_train"] & sets["outer_test"]) \
            and not (sets["inner_val"] & sets["outer_test"]), f"patient leakage r{r} k{k}"
        assert not (set(g.ecg_id) & set(meta[meta.split == "test"].ecg_id)), "locked test inside CV"
    for r in range(N_REPEATS):
        ot = f[(f.repeat == r) & (f.role == "outer_test")]
        assert ot.ecg_id.is_unique and len(ot) == len(D)
    return f


def folds(repeat: int = 0) -> pd.DataFrame:
    f = pd.read_csv(FOLDS)
    return f[f.repeat == repeat]


def split_indices(meta_D: pd.DataFrame, repeat: int, k: int):
    """-> index arrays into meta_D for inner_train, inner_val, outer_test of fold k."""
    f = folds(repeat); f = f[f.fold == k]
    pos = pd.Series(np.arange(len(meta_D)), index=meta_D.ecg_id)
    return tuple(pos.loc[f[f.role == r].ecg_id].values for r in ("inner_train", "inner_val", "outer_test"))


def dev_and_test():
    meta = load_meta()
    D = meta[meta.split != "test"].reset_index(drop=True)
    T = meta[meta.split == "test"].reset_index(drop=True)
    return meta, D, T


if __name__ == "__main__":
    f = make_folds()
    FOLDS.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(FOLDS, index=False)
    meta, D, T = dev_and_test()
    lab = D.set_index("ecg_id").label
    s = f[(f.repeat == 0)].assign(label=lambda x: lab.loc[x.ecg_id].values).groupby(["fold", "role", "label"]).size().unstack()
    print(f"D = {len(D)} ECGs / {D.patient.nunique()} patients; T = {len(T)} ECGs / {T.patient.nunique()} patients (locked)")
    print(s)
