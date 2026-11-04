"""Evaluation metrics for the 4-class LVEF category and the derived HF screening."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from .config import CLASSES, HF_THRESHOLD


def _binary(y_true_bin, score, thr):
    pred = score >= thr
    tp = int(np.sum(pred & y_true_bin)); tn = int(np.sum(~pred & ~y_true_bin))
    fp = int(np.sum(pred & ~y_true_bin)); fn = int(np.sum(~pred & y_true_bin))
    div = lambda a, b: float(a / b) if b else float("nan")
    try:
        auc = float(roc_auc_score(y_true_bin, score))
    except ValueError:
        auc = float("nan")
    return dict(auc=auc, accuracy=div(tp + tn, tp + tn + fp + fn),
                sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                ppv=div(tp, tp + fp), npv=div(tn, tn + fn), threshold=thr,
                tp=tp, tn=tn, fp=fp, fn=fn)


def pick_hf_threshold(y_true, probs, target_sensitivity: float) -> float:
    """Highest threshold on 1-P(Normal) whose sensitivity on (validation) data is still
    >= target. Screening favours sensitivity: a missed low-EF patient costs more than an
    extra echo."""
    y = np.asarray(y_true) > 0
    score = 1 - np.asarray(probs)[:, 0]
    pos = np.sort(score[y])[::-1]
    if len(pos) == 0:
        return HF_THRESHOLD
    k = int(np.ceil(target_sensitivity * len(pos))) - 1
    return float(pos[min(max(k, 0), len(pos) - 1)])


def fast_macro_f1(y_true, y_pred, n_classes: int = 4) -> float:
    """Same value as sklearn f1_score(average="macro", zero_division=0) (averaged over the
    labels present in y_true or y_pred), ~50x faster - used inside search/bootstrap loops."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return float(f1[present].mean())


def severity_score(probs) -> np.ndarray:
    """Expected class index sum_k k*p_k in [0, 3] (classes are ordered by severity)."""
    probs = np.asarray(probs)
    return probs @ np.arange(probs.shape[1])


def ordinal_decode(probs, cutpoints) -> np.ndarray:
    """Category = number of cut-points the severity score exceeds."""
    return np.searchsorted(np.asarray(cutpoints), severity_score(probs), side="right")


def fit_ordinal_cutpoints(y_true, probs, n_grid: int = 41) -> list[float]:
    """Choose 3 increasing cut-points on the severity score that maximise macro-F1 on
    (validation) data. Argmax rarely picks the middle classes (Mild/Moderate) because
    their probability mass is split between neighbours; ordinal decoding fixes that."""
    y = np.asarray(y_true)
    s = severity_score(probs)
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, n_grid)))
    best, best_c = -1.0, None
    for i, a in enumerate(grid):
        for j in range(i + 1, len(grid)):
            b = grid[j]
            for c in grid[j + 1:]:
                pred = np.searchsorted([a, b, c], s, side="right")
                f = fast_macro_f1(y, pred)
                if f > best:
                    best, best_c = f, [float(a), float(b), float(c)]
    return best_c


def evaluate(y_true, probs, hf_threshold: float = HF_THRESHOLD, cutpoints=None) -> dict:
    """y_true: int labels [N]; probs: softmax probabilities [N, 4].
    cutpoints: if given, the 4-class prediction uses ordinal decoding instead of argmax."""
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    y_pred = probs.argmax(1) if cutpoints is None else ordinal_decode(probs, cutpoints)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)),
                                                 zero_division=0)
    out = dict(
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # ordinal agreement: being off by one category is penalised less than by three
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        within_one_category=float(np.mean(np.abs(y_true - y_pred) <= 1)),
        per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                           support=int(s[i])) for i, c in enumerate(CLASSES)},
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
    )
    # Heart-failure screening: reduced EF (Mild/Moderate/Severe) vs Normal.
    hf_score = 1 - probs[:, 0]
    out["hf_screen"] = _binary(y_true > 0, hf_score, hf_threshold)
    out["hf_screen_at_0.5"] = _binary(y_true > 0, hf_score, 0.5)
    # HFrEF-range detection (EF <= 40%: Moderate or Severe) - the target used by most
    # published AI-ECG low-EF studies, useful for comparison with the literature.
    out["ef_le_40"] = _binary(y_true >= 2, probs[:, 2:].sum(1), 0.5)
    return out
