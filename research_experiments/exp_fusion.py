"""Late fusion of the per-input models (probability averaging), as in the project's default
model: weights on a 0.1 grid (2 members) or 0.1 simplex (3 members) chosen on VALIDATION by
mean(macro-F1, QWK, HF AUC); the YES/NO threshold is re-tuned on validation (>= 80 % sens).
Also builds the ECGFounder A2 seed-ensemble (mean of the 3 seeds) and the final comparison
table with patient-cluster paired bootstrap tests against the default model.

    python -m research_experiments.exp_fusion
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from research_experiments.common import (CLASSES, RES, load_meta, load_run_probs, paired_bootstrap, save_run,
                                         seed_all)
from ecg_ef.metrics import evaluate
from ecg_ef.train import selection_score

BASE = "B0_default_fusion_header0.6_resnet18"


def probs(tag, meta, split):
    ids, p = load_run_probs(tag, split)
    order = pd.Series(range(len(ids)), index=ids).loc[meta[meta.split == split].ecg_id].values
    return p[order]


def seed_ensemble(name, tags, meta):
    mv = meta[meta.split == "val"].reset_index(drop=True); mt = meta[meta.split == "test"].reset_index(drop=True)
    pv = np.mean([probs(t, meta, "val") for t in tags], 0); pt = np.mean([probs(t, meta, "test") for t in tags], 0)
    return save_run(name, mv, pv, mt, pt, dict(input="mean of seeds", members=tags))


def fuse(name, tags, meta, step=0.1, criterion="score"):
    """criterion 'score' = project rule mean(macro-F1, QWK, HF AUC) [primary];
    'accuracy' = validation 4-class accuracy [secondary analysis, because 4-class accuracy is
    the user's stated main objective; still validation-only]."""
    mv = meta[meta.split == "val"].reset_index(drop=True); mt = meta[meta.split == "test"].reset_index(drop=True)
    PV = [probs(t, meta, "val") for t in tags]; PT = [probs(t, meta, "test") for t in tags]
    grid = np.round(np.arange(0, 1 + 1e-9, step), 2)
    cands = [w for w in itertools.product(grid, repeat=len(tags) - 1) if sum(w) <= 1 + 1e-9]
    best = None
    for w in cands:
        w = list(w) + [round(1 - sum(w), 2)]
        mval = evaluate(mv.label_idx.values, sum(wi * p for wi, p in zip(w, PV)))
        sc = selection_score(mval) if criterion == "score" else mval["accuracy"]
        if best is None or sc > best[0] + 1e-12:
            best = (sc, w)
    sc, w = best
    pv = sum(wi * p for wi, p in zip(w, PV)); pt = sum(wi * p for wi, p in zip(w, PT))
    return save_run(name, mv, pv, mt, pt, dict(input="late fusion", members=tags, weights=w, val_selection_score=sc,
                                               weight_criterion=criterion))


def main():
    seed_all()
    meta = load_meta()
    have = lambda t: (RES / t / "metrics.json").exists()
    # A2 seed ensembles (if the queue produced them)
    for head in ("ce", "cumlink"):
        tags = [f"A2_ecgf_ft56_{head}_seed{s}" for s in (42, 43, 44)]
        if all(have(t) for t in tags):
            seed_ensemble(f"A2_ecgf_ft56_{head}_3seed_ens", tags, meta)

    H, H3, I = "B0_header_logreg_multinomial", "HA_H3_demographics+measurements", "B0_resnet18_image_ce"
    F1, F2 = "A1_ecgf_aligned_deep_multinomial", "A2_ecgf_ft56_ce_3seed_ens"
    Hc, Fc1, Fc2, Ic = "B_header_cumlink", "AB_ecgf_aligned_deep_cumlink", "A2_ecgf_ft56_cumlink_3seed_ens", \
        "B_resnet18_image_cumlink_seed42"
    plan = {
        "FU_header+image (re-tuned)": [H, I],
        "FU_header+ecgfA1": [H, F1],
        "FU_header+ecgfA2": [H, F2],
        "FU_headerNumbersOnly+ecgfA1": [H3, F1],
        "FU_headerNumbersOnly+ecgfA2": [H3, F2],
        "FU_header+image+ecgfA1": [H, I, F1],
        "FU_header+image+ecgfA2": [H, I, F2],
        "FU_ordinal_header+ecgfA1": [Hc, Fc1],
        "FU_ordinal_header+ecgfA2": [Hc, Fc2],
        "FU_ordinal_header+image": [Hc, Ic],
    }
    done = {}
    for name, tags in plan.items():
        if all(have(t) for t in tags):
            done[name] = fuse(name.replace(" ", "_").replace("(", "").replace(")", ""), tags, meta)
    # secondary: same fusions, weights chosen by VALIDATION accuracy
    for name in ("FU_header+image (re-tuned)", "FU_header+ecgfA1", "FU_header+ecgfA2", "FU_header+image+ecgfA1",
                 "FU_header+image+ecgfA2", "FU_ordinal_header+ecgfA2"):
        tags = plan[name]
        if all(have(t) for t in tags):
            nm = "FUacc_" + name[3:].replace(" ", "_").replace("(", "").replace(")", "")
            done[nm] = fuse(nm, tags, meta, criterion="accuracy")
    json.dump({k: dict(members=v["members"], weights=v["weights"], val_score=v["val_selection_score"])
               for k, v in done.items()}, open(RES / "fusion_weights.json", "w"), indent=2)


if __name__ == "__main__":
    main()
