"""Phase 0 - reproduce the existing default model from its saved checkpoints (no retraining).

1. ResNet-18 (filtered images) probabilities recomputed from models/resnet18_filtered.pt
2. Header logistic regression recomputed from models/header_logreg.joblib, and re-fitted
   from scratch with the original recipe (deterministic) to check it reproduces
3. Default fusion = 0.6 x header + 0.4 x ResNet-18 (results/best_model.json)
4. Compared with the stored predictions/metrics in results/ (read-only)

    python -m research_experiments.phase0_reproduce
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.common import (CLASSES, RES, ROOT, env_record, full_metrics, load_meta,
                                         load_run_probs, save_run, seed_all)
from ecg_ef.data import Normalizer, load_split
from ecg_ef.header import load_header_data
from ecg_ef.metrics import evaluate
from ecg_ef.predict import load_checkpoint
from ecg_ef.train import predict_probs, selection_score


def main():
    torch.set_num_threads(4); seed_all()
    meta = load_meta()
    mv = meta[meta.split == "val"].reset_index(drop=True)
    mt = meta[meta.split == "test"].reset_index(drop=True)
    report = {"env": env_record()}

    # 1. ResNet-18 from checkpoint
    model, ckpt = load_checkpoint(ROOT / "models" / "resnet18_filtered.pt")
    norm = Normalizer(**ckpt["normalizer"])
    img = {}
    for s, md in (("val", mv), ("test", mt)):
        x, _, df = load_split(s, ckpt["variant"], tuple(ckpt["img_hw"]))
        assert (df.ecg_id.values == md.ecg_id.values).all()
        t0 = time.time(); img[s] = predict_probs(model, x, norm); dt = time.time() - t0
        report[f"resnet18_{s}_cnn_seconds_per_ecg"] = dt / len(x)
    n_params = sum(p.numel() for p in model.parameters())

    # 2. header model from the saved bundle, and re-fitted
    h, X, vocab, med, Xs, ys = load_header_data()
    bundle = joblib.load(ROOT / "models" / "header_logreg.joblib")
    assert list(X.columns) == bundle["columns"], "header design matrix differs from the saved bundle"
    hdr = {}
    for s, md in (("val", mv), ("test", mt)):
        hs = h[h.split == s]
        assert (hs.ecg_id.values == md.ecg_id.values).all()
        hdr[s] = bundle["model"].predict_proba(Xs[s])
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        clf.fit(Xs["train"], ys["train"])
        sc = selection_score(evaluate(ys["val"], clf.predict_proba(Xs["val"])))
        if best is None or sc > best[0]:
            best = (sc, C, clf)
    refit = {s: best[2].predict_proba(Xs[s]) for s in ("val", "test")}
    report["header_refit_C"] = best[1]; report["header_saved_C"] = bundle["C"]
    report["header_refit_max_abs_prob_diff_test"] = float(np.abs(refit["test"] - hdr["test"]).max())

    # 3. fusion
    bm = json.loads((ROOT / "results" / "best_model.json").read_text())
    w = bm["header_weight"]
    fus = {s: w * hdr[s] + (1 - w) * img[s] for s in ("val", "test")}

    # 4. compare with stored outputs
    for tag, probs in (("resnet18_filtered", img), ("header_logreg", hdr), ("fusion_header_resnet18", fus)):
        ids, stored = load_run_probs(tag, "test", ROOT / "results")
        order = pd.Series(range(len(ids)), index=ids).loc[mt.ecg_id].values
        diff = float(np.abs(stored[order] - probs["test"]).max())
        old = json.loads((ROOT / "results" / tag / "metrics.json").read_text())["test"]
        thr = bm["hf_threshold"] if tag.startswith("fusion") else None
        new = full_metrics(mt.label_idx.values, probs["test"], thr)
        report[tag] = {"max_abs_prob_diff_vs_stored": diff,
                       "stored": {k: old[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk")},
                       "reproduced": {k: new[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk")}}
        print(tag, json.dumps(report[tag]))

    extra = dict(input="ECG image (filtered, 256x512)", params=n_params, source="models/resnet18_filtered.pt")
    save_run("B0_resnet18_image_ce", mv, img["val"], mt, img["test"], extra)
    save_run("B0_header_logreg_multinomial", mv, hdr["val"], mt, hdr["test"],
             dict(input="printed header (measurements + statements + age/sex)", C=bundle["C"]))
    save_run("B0_default_fusion_header0.6_resnet18", mv, fus["val"], mt, fus["test"],
             dict(input="header + image", header_weight=w), hf_threshold=bm["hf_threshold"])
    (RES / "phase0_reproduction.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
