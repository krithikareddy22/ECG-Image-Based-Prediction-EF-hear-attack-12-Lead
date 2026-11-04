"""Does training on an EQUAL number of ECGs per category help?

Compares ways of handling the unequal classes (train: Normal 307, Mild 148, Moderate 141,
Severe 73) on the same 210 test ECGs:

  none         all images, no correction
  weights      all images, loss weights ~ 1/class size            (method used everywhere)
  undersample  equal images per class: keep 73 of each, drop the rest (random, repeated)
  oversample   equal images per class: copy rarer-class images up to 307 each
  balanced     (1-D CNN only) all images, batches drawn with equal class probability

Header model: fitted here (seconds per fit). 1-D CNN: run first with
    python -m ecg_ef.train1d --sampling {weights,undersample,balanced} --seeds 3 --tag bal1d_<s>

    python -m ecg_ef.balance          # -> results/balance_experiment.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, RESULTS_DIR
from .header import load_header_data
from .metrics import evaluate
from .train import selection_score

ROW = ["accuracy", "balanced_accuracy", "macro_f1", "hf_auc"]


def summarise(ms: list[dict]) -> dict:
    out = {}
    for k in ROW:
        v = np.array([m[k] for m in ms])
        out[k] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
    out["recall"] = np.mean([m["recall"] for m in ms], 0)
    out["n_runs"] = len(ms)
    return out


def flat(m: dict) -> dict:
    return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "hf_auc": m["hf_screen"]["auc"],
            "recall": [m["per_class"][c]["recall"] for c in CLASSES]}


def header_runs(variant: str, Xs, ys, n_repeats: int = 20) -> tuple[list[dict], int]:
    Xtr, ytr = Xs["train"], ys["train"]
    counts = np.bincount(ytr, minlength=4)
    runs, n_train = [], len(ytr)
    for r in range(n_repeats if variant == "undersample" else 1):
        rng = np.random.default_rng(r)
        if variant == "undersample":
            idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), counts.min(), replace=False)
                                  for c in range(4)])
        elif variant == "oversample":
            idx = np.concatenate([np.flatnonzero(ytr == c) for c in range(4)] +
                                 [rng.choice(np.flatnonzero(ytr == c), counts.max() - counts[c])
                                  for c in range(4)])
        else:
            idx = np.arange(len(ytr))
        n_train = len(idx)
        best = None
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):            # C chosen on validation, as in header.py
            clf = make_pipeline(StandardScaler(), LogisticRegression(
                C=C, class_weight="balanced" if variant == "weights" else None, max_iter=5000))
            clf.fit(Xtr[idx], ytr[idx])
            sc = selection_score(evaluate(ys["val"], clf.predict_proba(Xs["val"])))
            if best is None or sc > best[0]:
                best = (sc, clf)
        runs.append(flat(evaluate(ys["test"], best[1].predict_proba(Xs["test"]))))
    return runs, n_train


def main():
    _, _, _, _, Xs, ys = load_header_data()
    rows = []
    for v in ("none", "weights", "undersample", "oversample"):
        runs, n = header_runs(v, Xs, ys)
        rows.append(("Header model", v, n, summarise(runs)))
    for v in ("weights", "undersample", "balanced"):
        f = RESULTS_DIR / f"bal1d_{v}" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            n = {"undersample": 4 * 73}.get(v, 669)
            rows.append(("1-D CNN (waveforms)", v, n, summarise(m["test_per_seed"])))

    lines = ["| Model | Class handling | Training ECGs | Runs | Accuracy | Balanced acc. | Macro-F1 "
             "| HF AUC | Recall N / Mi / Mo / S |", "|---|---|---:|---:|---|---|---|---|---|"]
    fmt = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}" if t[1] else f"{t[0]:.3f}"
    for model, v, n, s in rows:
        lines.append(f"| {model} | {v} | {n} | {s['n_runs']} | " + " | ".join(fmt(s[k]) for k in ROW)
                     + " | " + " / ".join(f"{x:.2f}" for x in s["recall"]) + " |")
    table = "\n".join(lines)
    (RESULTS_DIR / "balance_experiment.md").write_text(
        "# Equal number of ECGs per category vs class weights (test n = 210)\n\n"
        "Mean ± SD over runs (random draws for `undersample`; random seeds for the 1-D CNN).\n\n"
        + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
