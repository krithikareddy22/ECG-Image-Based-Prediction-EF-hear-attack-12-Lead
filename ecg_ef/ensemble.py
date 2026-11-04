"""Soft-voting ensemble: average the class probabilities of several trained CNNs.

    python -m ecg_ef.ensemble                       # top-3 CNNs by validation score
    python -m ecg_ef.ensemble --top-k 5
    python -m ecg_ef.ensemble --members ecgnet_filtered resnet18_filtered
    python -m ecg_ef.ensemble --members header_logreg resnet1d_x5 --tune-weight --name fusion

Works from each model's saved val/test predictions (re-computed from the checkpoint if
missing) and writes results/ensemble/ in the same format as a single model.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import CLASS_TO_IDX, CLASSES, HF_TARGET_SENSITIVITY, MODELS_DIR, RESULTS_DIR
from .data import Normalizer, load_split
from .metrics import evaluate, pick_hf_threshold
from .predict import load_checkpoint
from .train import plot_confusion, predict_probs, save_predictions, selection_score


def load_predictions(tag: str, split: str):
    """(y_true [N], probs [N,4], ecg_ids) for one model, recomputing from the checkpoint
    if the CSV is missing (runs made before val predictions were saved)."""
    path = RESULTS_DIR / tag / f"{split}_predictions.csv"
    if not path.exists():
        model, ckpt = load_checkpoint(MODELS_DIR / f"{tag}.pt")
        norm = Normalizer(**ckpt["normalizer"])
        for s in ("val", "test"):
            x, _, df = load_split(s, ckpt["variant"], tuple(ckpt["img_hw"]))
            save_predictions(predict_probs(model, x, norm), df,
                             RESULTS_DIR / tag / f"{s}_predictions.csv")
    df = pd.read_csv(path).sort_values("ecg_id").reset_index(drop=True)
    return (df.true.map(CLASS_TO_IDX).values, df[[f"p_{c}" for c in CLASSES]].values,
            df.ecg_id.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=None, help="checkpoint tags")
    ap.add_argument("--top-k", type=int, default=3,
                    help="if --members is not given: the k CNNs with the best VALIDATION score")
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--tune-weight", action="store_true",
                    help="2 members: choose the mixing weight w*p1 + (1-w)*p2 on VALIDATION")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.members:
        tags = args.members
    else:
        scores = {p.stem: selection_score(json.loads(
                      (RESULTS_DIR / p.stem / "metrics.json").read_text())["val"])
                  for p in MODELS_DIR.glob("*.pt") if (RESULTS_DIR / p.stem / "metrics.json").exists()}
        tags = sorted(scores, key=scores.get, reverse=True)[:args.top_k]
        print("validation scores:", {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])})
    print("members:", tags)

    P = {}
    for s in ("val", "test"):
        ys, ps, ids = zip(*(load_predictions(t, s) for t in tags))
        assert all((i == ids[0]).all() for i in ids), "members disagree on the split"
        P[s] = (ys[0], np.mean(ps, 0), pd.DataFrame({"ecg_id": ids[0],
                                                     "label": [CLASSES[k] for k in ys[0]]}))
    (yva, pva, dva), (yte, pte, dte) = P["val"], P["test"]
    weight = None
    if args.tune_weight:
        assert len(tags) == 2, "--tune-weight needs exactly 2 members"
        pv = [load_predictions(t, "val")[1] for t in tags]
        pt = [load_predictions(t, "test")[1] for t in tags]
        grid = np.round(np.linspace(0, 1, 11), 2)
        sc = {w: selection_score(evaluate(yva, w * pv[0] + (1 - w) * pv[1])) for w in grid}
        weight = float(max(sc, key=sc.get))
        print("validation score by weight of", tags[0], {k: round(v, 3) for k, v in sc.items()})
        pva, pte = weight * pv[0] + (1 - weight) * pv[1], weight * pt[0] + (1 - weight) * pt[1]

    hf_thr = pick_hf_threshold(yva, pva, HF_TARGET_SENSITIVITY)
    val_m, test_m = evaluate(yva, pva, hf_thr), evaluate(yte, pte, hf_thr)
    out = RESULTS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(tag=args.name, model="ensemble", variant="mixed", pretrained=False,
                   img_hw=[256, 512], params=None, members=tags, weight=weight, hf_threshold=hf_thr,
                   val=val_m, test=test_m)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(test_m["confusion_matrix"],
                   f"Ensemble (top-{len(tags)} CNNs) - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    save_predictions(pte, dte, out / "test_predictions.csv")
    save_predictions(pva, dva, out / "val_predictions.csv")
    print(f"[{args.name}] VAL macro-F1 {val_m['macro_f1']:.3f} | TEST acc {test_m['accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}  "
          f"HF-AUC {test_m['hf_screen']['auc']:.3f}")


if __name__ == "__main__":
    main()
