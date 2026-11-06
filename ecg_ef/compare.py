"""Collect every model's results into one comparison table + chart, and pick the model
used by predict.py.

For each model two 4-class decision rules are reported:
  * argmax   - the most probable class
  * ordinal  - expected severity sum_k k*p_k cut into 4 bands; the 3 cut-points are fit
               on the VALIDATION set (macro-F1). Argmax almost never outputs the middle
               classes (Mild/Moderate) because their probability is shared with the
               neighbouring classes; the ordinal rule respects the Normal<Mild<Moderate<Severe
               ordering. It raises Mild/Moderate recall, but for the CNNs it lowers Severe
               recall a lot, so predict.py uses argmax by default.
Model selection uses the VALIDATION set only (mean of macro-F1, QWK and heart-failure AUC),
so the test numbers stay an unbiased estimate.

    python -m ecg_ef.compare
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .config import RESULTS_DIR, ROOT
from .ensemble import load_predictions
from .metrics import evaluate, fast_macro_f1, fit_ordinal_cutpoints, ordinal_decode
from .train import plot_confusion, selection_score

PRETTY = {"ecgnet": "ECGNet (custom SE-ResNet)", "resnet18": "ResNet-18", "resnet34": "ResNet-34",
          "resnet50": "ResNet-50", "vgg16": "VGG16", "mobilenetv2": "MobileNetV2", "inception_v3": "InceptionV3",
          "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121",
          "mobilenetv3": "MobileNetV3-Large", "convnext_atto": "ConvNeXt-Atto",
          "hog_svm": "HOG + SVM (classical)", "ensemble": "Ensemble (top-3 CNNs by validation)",
          "resnet1d": "1D-ResNet on digitised waveforms", "header_logreg": "Header measurements + statements"}

SHORT = {"header_logreg": "header", "resnet18_filtered": "ResNet-18", "ecgnet_filtered": "ECGNet",
         "resnet1d_x5": "1D-ResNet x5", "resnet1d": "1D-ResNet", "logreg_header": "header (LR)",
         "xgb_waveform": "XGBoost waveform", "fusion_header_resnet18": "header + ResNet-18"}
TABULAR = {"xgb": "XGBoost", "logreg": "Logistic regression"}
TAB_INPUT = {"waveform": "waveform features", "header": "header text",
             "waveform_header": "waveform features + header text"}


def load_metrics(tag):
    return json.loads((RESULTS_DIR / tag / "metrics.json").read_text())


def input_kind(m) -> str:
    """Which part of the ECG PDF a model reads."""
    if m["model"] == "ensemble":
        kinds = sorted({input_kind(load_metrics(t)) for t in m["members"]})
        return " + ".join(kinds)
    if m["variant"] in ("gaussian", "median", "bilateral", "clahe", "adaptive"):
        return f"image ({m['variant']})"
    if m["model"] in TABULAR:
        return TAB_INPUT.get(m["variant"], m["variant"])
    return {"signal": "waveform (vector)", "header": "header text", "raw": "image (grid kept)"}.get(
        m["variant"], "image")


def components(m):
    """What predict.py needs to run this model, or None if it can't."""
    if m["model"] in ("hog_svm", "header_logreg", "resnet1d", *TABULAR):
        return None
    if m["model"] != "ensemble":
        return {"checkpoints": [f"models/{m['tag']}.pt"]}
    members = m["members"]
    if "header_logreg" in members:
        img = [t for t in members if t != "header_logreg"]
        if len(img) != 1 or load_metrics(img[0])["model"] in ("resnet1d", "hog_svm", "ensemble", *TABULAR):
            return None
        w = m["weight"] if members[0] == "header_logreg" else 1 - m["weight"]
        return {"checkpoints": [f"models/{img[0]}.pt"], "header_model": "models/header_logreg.joblib",
                "header_weight": w, "image_tag": img[0]}
    if any(load_metrics(t)["model"] in ("resnet1d", "hog_svm", "ensemble", "header_logreg", *TABULAR)
           for t in members):
        return None
    return {"checkpoints": [f"models/{t}.pt" for t in members]}

INK, MUTED, GRID, BAR, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def model_name(m):
    if m["model"] == "ensemble" and m["tag"] != "ensemble":
        parts = [SHORT.get(t) or (model_name(load_metrics(t)) if (RESULTS_DIR / t / "metrics.json").exists()
                                  else t) for t in m["members"]]
        if m.get("weight") is not None:
            parts = [f"{parts[0]} {m['weight']:.1f}", f"{parts[1]} {1 - m['weight']:.1f}"]
        return "Fusion: " + " + ".join(parts)
    if m["model"] in TABULAR:
        return f"{TABULAR[m['model']]} on {TAB_INPUT.get(m['variant'], m['variant'])}"
    name = PRETTY.get(m["model"], m["model"])
    if m["model"] == "resnet1d":
        name += (" + rhythm strip" if m.get("rhythm") else "") + (
            f", {m['seeds']} seeds" if m.get("seeds", 1) > 1 else "")
        return name
    if m["variant"] == "raw":
        name += ", no grid filter"
    elif m["variant"] in ("gaussian", "median", "bilateral", "clahe", "adaptive"):
        name += f", {m['variant']} filter"
    if m["model"] == "ecgnet" and tuple(m["img_hw"]) != (256, 512):
        name += f", {m['img_hw'][0]}x{m['img_hw'][1]}"
    if m.get("lr") and abs(m["lr"] - 1e-3) > 1e-9 and not m.get("pretrained"):
        name += ", lr " + ("%.0e" % m["lr"]).replace("e-0", "e-")
    if m.get("pretrained"):
        name += ", ImageNet"
    a = m.get("args", {}) or {}
    if a.get("multitask") or "_mtl" in m["tag"]:
        name += ", multi-task"
    if a.get("focal_gamma") or "_focal" in m["tag"]:
        name += ", focal loss"
    return name


def bootstrap_ci(y, probs, cut, n=2000, seed=0):
    """95% bootstrap CIs (resampling test ECGs) for accuracy, ordinal macro-F1, HF AUC."""
    rng = np.random.default_rng(seed)
    am, od, hf = probs.argmax(1), ordinal_decode(probs, cut), 1 - probs[:, 0]
    acc, f1, auc = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        acc.append((am[i] == y[i]).mean())
        f1.append(fast_macro_f1(y[i], od[i]))
        if 0 < (y[i] > 0).sum() < len(i):
            auc.append(roc_auc_score(y[i] > 0, hf[i]))
    q = lambda v: tuple(np.percentile(v, [2.5, 97.5]).round(3))
    return {"accuracy": q(acc), "ord_macro_f1": q(f1), "hf_auc": q(auc)}


def load_rows():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        tag = m["tag"]
        yv, pv, _ = load_predictions(tag, "val")
        yt, pt, _ = load_predictions(tag, "test")
        cut = fit_ordinal_cutpoints(yv, pv)
        ov = evaluate(yv, pv, m["hf_threshold"], cut)
        ot = evaluate(yt, pt, m["hf_threshold"], cut)
        ci = bootstrap_ci(yt, pt, cut)
        (f.parent / "ordinal.json").write_text(json.dumps(
            {"cutpoints": cut, "val": ov, "test": ot, "test_bootstrap_95ci": ci}, indent=2))
        plot_confusion(ot["confusion_matrix"], f"{tag} - test, ordinal decoding "
                       f"(macro-F1 {ot['macro_f1']:.2f})", f.parent / "confusion_matrix_ordinal.png")
        t, v = m["test"], m["val"]
        prec, rec, _, _ = precision_recall_fscore_support(yt, pt.argmax(1), labels=range(4),
                                                          average="macro", zero_division=0)
        ovr = [roc_auc_score(yt == k, pt[:, k]) for k in range(4)]   # one-vs-rest per class
        rows.append({
            "tag": tag, "Model": model_name(m), "Input": input_kind(m),
            "Params (M)": round(m["params"] / 1e6, 1) if m.get("params") else None,
            "Train (min)": m.get("train_minutes"),
            "Val score": selection_score(v),
            "Accuracy": t["accuracy"], "Balanced acc": t["balanced_accuracy"],
            "Macro precision": prec, "Macro recall": rec,
            "Macro-F1": t["macro_f1"], "QWK": t["qwk"],
            "Macro AUROC (OvR)": float(np.mean(ovr)),
            "AUROC Normal": ovr[0], "AUROC Mild": ovr[1], "AUROC Moderate": ovr[2],
            "AUROC Severe": ovr[3],
            "Ord. accuracy": ot["accuracy"], "Ord. balanced acc": ot["balanced_accuracy"],
            "Ord. macro-F1": ot["macro_f1"], "Ord. QWK": ot["qwk"],
            "HF AUC": t["hf_screen"]["auc"], "HF sensitivity": t["hf_screen"]["sensitivity"],
            "HF specificity": t["hf_screen"]["specificity"], "EF<=40% AUC": t["ef_le_40"]["auc"],
            "Accuracy 95% CI": "%.2f-%.2f" % ci["accuracy"],
            "Ord. macro-F1 95% CI": "%.2f-%.2f" % ci["ord_macro_f1"],
            "HF AUC 95% CI": "%.2f-%.2f" % ci["hf_auc"],
            "components": components(m),
            "hf_threshold": m["hf_threshold"], "cutpoints": cut,
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path):
    df = df.sort_values("HF AUC")
    # reference = what a model with no information scores on this test set
    metrics = [("Accuracy", "4-class accuracy", 96 / 210, "always 'Normal'"),
               ("Balanced acc", "4-class balanced accuracy", 0.25, "chance"),
               ("HF AUC", "Heart-failure screen AUC", 0.5, "chance")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.42 * len(df) + 1.5), sharey=True,
                             facecolor=SURFACE)
    for ax, (col, title, ref, ref_label) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
        ax.text(ref, len(df) - 0.35, f" {ref_label}", color=MUTED, fontsize=8, va="bottom")
        ax.barh(df.Model, df[col], color=BAR, height=0.62)
        for y, v in enumerate(df[col]):
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK, zorder=3,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axes[0].tick_params(axis="y", colors=INK, labelsize=9)
    fig.suptitle("Held-out test patients (n = 210); sorted by heart-failure screen AUC",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def to_md(df: pd.DataFrame) -> str:
    md = df.copy()
    for c in md.columns:
        if md[c].dtype == float and c not in ("Params (M)", "Train (min)"):
            md[c] = md[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return md.fillna("").to_markdown(index=False)


def main():
    df = load_rows()
    if df.empty:
        raise SystemExit("no results/*/metrics.json yet")
    df = df.sort_values("HF AUC", ascending=False)
    show = df.drop(columns=["tag", "components", "hf_threshold", "cutpoints"])
    show.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    (RESULTS_DIR / "comparison.md").write_text(to_md(show) + "\n")
    plot(df, RESULTS_DIR / "comparison.png")

    def files_of(comp):
        return comp["checkpoints"] + ([comp["header_model"]] if "header_model" in comp else [])

    # candidates for predict.py: pipelines it can run whose model files are present
    runnable = df[df.components.map(lambda c: c is not None and
                                    all((ROOT / f).exists() for f in files_of(c)))]
    best = runnable.sort_values("Val score", ascending=False).iloc[0]
    comp = dict(best.components)
    files = files_of(comp)
    if True:
        spec = {"tag": best.tag, "model": best.Model, "input": best.Input, **comp,
                "hf_threshold": best.hf_threshold, "decoding": "argmax",
                "selected_by": "validation score = mean(macro-F1, QWK, HF AUC)",
                "val_score": best["Val score"]}
        if "header_model" in comp:     # image-only fallback for PNG/JPG inputs
            img = df[df.tag == comp.pop("image_tag")].iloc[0]
            spec.pop("image_tag")
            spec.update(image_model=img.Model, image_hf_threshold=img.hf_threshold,
                        image_ordinal_cutpoints=img.cutpoints)
        else:
            spec["ordinal_cutpoints"] = best.cutpoints
        (RESULTS_DIR / "best_model.json").write_text(json.dumps(spec, indent=2))
    print(to_md(show))
    print(f"\nselected for predict.py (best validation score): {best.Model}  -> {files}")


if __name__ == "__main__":
    main()
