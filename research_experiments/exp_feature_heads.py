"""Experiment A1 (frozen ECGFounder + lightweight classifier) and Experiment B (ordinal heads)
on fixed features, original patient-level split, selection on validation only.

Pre-specified primary A1 model: 'aligned' input (chosen by the label-free domain check in
ecgfounder/sanity_check.py), 1024-d deep features, multinomial head. All other input
constructions / feature sets are reported as an ablation.

    python -m research_experiments.exp_feature_heads
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from research_experiments.common import MOD, RES, load_meta, save_run, seed_all
from research_experiments.heads import HEADS, fit_head
from ecg_ef.header import load_header_data, NUMERIC, SUMMARY


def ecgf_features(meta, mode, kind):
    z = np.load(MOD / f"features_ecgfounder_{mode}.npz")
    assert (z["ecg_id"] == meta.ecg_id.values).all()
    return {"deep": z["feat"], "logits": z["logits"],
            "deep+logits": np.concatenate([z["feat"], z["logits"]], 1)}[kind]


def header_matrix(meta):
    h, X, vocab, med, Xs, ys = load_header_data()
    X = X.copy(); X.insert(0, "ecg_id", h.ecg_id.values)
    X = meta[["ecg_id"]].merge(X, on="ecg_id", how="left")
    assert len(X) == len(meta) and not X.isna().any().any()
    return X.drop(columns="ecg_id"), vocab


def run(tag, X, meta, head, extra):
    sp = meta.split.values
    y = meta.label_idx.values
    t0 = time.time()
    fit = fit_head(X[sp == "train"], y[sp == "train"], X[sp == "val"], y[sp == "val"], head)
    train_s = time.time() - t0
    pv, pt = fit["predict"](X[sp == "val"]), fit["predict"](X[sp == "test"])
    t0 = time.time(); fit["predict"](X[sp == "test"]); inf = (time.time() - t0) / (sp == "test").sum()
    ex = dict(extra, head=head, l2=fit["l2"], val_selection_score=fit["val_score"],
              head_trainable_params=fit["n_params"], head_train_seconds=train_s,
              head_inference_seconds_per_ecg=inf, feature_dim=X.shape[1])
    if "n_rank_inconsistent" in pt:
        ex["rank_inconsistent_test"] = pt["n_rank_inconsistent"]; ex["rank_inconsistent_val"] = pv["n_rank_inconsistent"]
    dec = "rank (count P(y>k)>0.5)" if head == "coral" else "argmax"
    return save_run(tag, meta[sp == "val"].reset_index(drop=True), pv["probs"],
                    meta[sp == "test"].reset_index(drop=True), pt["probs"], ex,
                    pred_val=pv["pred"] if head == "coral" else None,
                    pred_test=pt["pred"] if head == "coral" else None, decoding=dec)


def main():
    seed_all()
    meta = load_meta()
    rows = []

    # A1 ablation: input construction x feature set (multinomial head)
    for mode in ("aligned", "tiled", "short"):
        for kind in ("deep", "logits", "deep+logits"):
            s = run(f"A1_ecgf_{mode}_{kind.replace('+', '_')}_multinomial", ecgf_features(meta, mode, kind), meta,
                    "multinomial", dict(input=f"ECGFounder frozen ({mode} waveform, {kind})"))
            rows.append(dict(tag=s["tag"], val_score=s["val_selection_score"], test_bal=s["test"]["balanced_accuracy"]))

    # A1 primary + Experiment B heads on the same frozen ECGFounder features
    X = ecgf_features(meta, "aligned", "deep")
    for head in HEADS:
        if head != "multinomial":
            run(f"AB_ecgf_aligned_deep_{head}", X, meta, head, dict(input="ECGFounder frozen (aligned waveform, deep)"))

    # Experiment B on the header features (same design matrix as the baseline header model)
    Xh, vocab = header_matrix(meta)
    for head in HEADS:
        run(f"B_header_{head}", Xh.values, meta, head, dict(input="printed header (all features)"))

    pd.DataFrame(rows).to_csv(RES / "A1_ablation_selection.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
