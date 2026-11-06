"""Features printed in the ECG header (text layer of the PDF), and a model on them.

Read from each PDF (the patient's NAME is never read or stored):
  * age, sex
  * machine measurements: Rate, RR, PR, QRSD, QT, QTcB, QTcF, P/QRS/T axis
  * the machine's interpretation: overall verdict (NORMAL / OTHERWISE NORMAL /
    BORDERLINE / ABNORMAL ECG) and each diagnostic statement (e.g. LEFT BUNDLE BRANCH
    BLOCK, INFERIOR INFARCT OLD, LOW VOLTAGE ...) as a yes/no feature.

These are derived from the same ECG recording (not from the echo), so they are a
legitimate input; sex also sets the Normal/Mild EF cut-off (52 vs 54 %).

    python -m ecg_ef.header                      # -> data/header_features.csv + model
"""
from __future__ import annotations

import json
import re
from multiprocessing import Pool

import joblib

import numpy as np
import pandas as pd
import pymupdf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, DATA_DIR, HF_TARGET_SENSITIVITY, MODELS_DIR, RESULTS_DIR, ROOT
from .metrics import evaluate, pick_hf_threshold
from .train import plot_confusion, save_predictions, selection_score

HEADER_CSV = DATA_DIR / "header_features.csv"
HEADER_MODEL = MODELS_DIR / "header_logreg.joblib"
NUMERIC = ["Rate", "RR", "PR", "QRSD", "QT", "QTcB", "QTcF"]
SUMMARY = ["NORMAL ECG", "OTHERWISE NORMAL ECG", "BORDERLINE ECG", "ABNORMAL ECG"]


def parse_header(pdf_path) -> dict:
    with pymupdf.open(str(pdf_path)) as doc:
        text = doc[0].get_text()
    f: dict = {}
    m = re.search(r"^\s*(\d+)\s*Years", text, re.M)
    f["age"] = float(m.group(1)) if m else np.nan
    f["male"] = 1.0 if re.search(r"^\s*Male\s*$", text, re.M) else (
        0.0 if re.search(r"^\s*Female\s*$", text, re.M) else np.nan)
    for k in NUMERIC:
        m = re.search(rf"^{k}\s*\n\s*(-?\d+)", text, re.M)
        f[k] = float(m.group(1)) if m else np.nan
    m = re.search(r"--AXIS--\s*\nP\s*\n\s*(-?\d+)?\s*\nQRS\s*\n\s*(-?\d+)?\s*\nT\s*\n\s*(-?\d+)?", text)
    for i, k in enumerate(["P_axis", "QRS_axis", "T_axis"]):
        f[k] = float(m.group(i + 1)) if m and m.group(i + 1) else np.nan
    statements = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(". "):
            statements.append(re.split(r"\.{3,}", s[2:])[0].strip())
        elif s.startswith("- ") and s.endswith("-"):
            f["summary"] = s.strip("- ").strip()
    f["statements"] = "|".join(statements)
    return f


def _one(src):
    return parse_header(ROOT / src)


def build_features():
    m = pd.read_csv(DATA_DIR / "manifest.csv")
    with Pool(4) as pool:
        rows = pool.map(_one, m.source, chunksize=16)
    h = pd.DataFrame(rows)
    h.insert(0, "ecg_id", m.ecg_id.values)
    h.to_csv(HEADER_CSV, index=False)
    return h


def design_matrix(h: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    X = h[["age", "male", *NUMERIC, "P_axis", "QRS_axis", "T_axis"]].copy()
    X["PR_missing"] = X.PR.isna().astype(float)            # e.g. atrial fibrillation
    for ax in ["P_axis", "QRS_axis", "T_axis"]:            # axes are angles
        X[ax + "_sin"] = np.sin(np.radians(X[ax])); X[ax + "_cos"] = np.cos(np.radians(X[ax]))
        X = X.drop(columns=ax)
    for s in SUMMARY:
        X["summary=" + s] = (h.summary == s).astype(float)
    st = h.statements.fillna("").str.split("|")
    for v in vocab:
        X["stmt=" + v] = st.map(lambda L: float(v in L))
    return X


def header_probs(pdf_path, bundle: dict) -> np.ndarray:
    """Class probabilities [4] from one ECG PDF's header, with a saved model bundle."""
    h = pd.DataFrame([parse_header(pdf_path)])
    X = design_matrix(h, bundle["vocab"])[bundle["columns"]].fillna(bundle["medians"])
    return bundle["model"].predict_proba(X.values)[0]


def load_header_data():
    """-> (h, X, vocab, medians, Xs, ys): feature matrix per split, imputed with TRAIN medians."""
    h = build_features() if not HEADER_CSV.exists() else pd.read_csv(HEADER_CSV)
    m = pd.read_csv(DATA_DIR / "manifest.csv")
    h = m[["ecg_id", "split", "label_idx", "label"]].merge(h, on="ecg_id")
    tr = h[h.split == "train"]
    # statements seen in >= 5 TRAINING ECGs
    counts = pd.Series([s for L in tr.statements.fillna("").str.split("|") for s in L if s]).value_counts()
    vocab = sorted(counts[counts >= 5].index)
    X = design_matrix(h, vocab)
    med = X[h.split == "train"].median()
    X = X.fillna(med)
    Xs = {s: X[h.split == s].values for s in ("train", "val", "test")}
    ys = {s: h[h.split == s].label_idx.values for s in ("train", "val", "test")}
    return h, X, vocab, med, Xs, ys


def main():
    h, X, vocab, med, Xs, ys = load_header_data()
    print(f"features: {X.shape[1]} ({len(vocab)} machine statements)")

    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                                 max_iter=5000))
        clf.fit(Xs["train"], ys["train"])
        sc = selection_score(evaluate(ys["val"], clf.predict_proba(Xs["val"])))
        print(f"  C={C:<5} val score {sc:.3f}")
        if best is None or sc > best[0]:
            best = (sc, C, clf)
    _, C, clf = best
    pva, pte = clf.predict_proba(Xs["val"]), clf.predict_proba(Xs["test"])
    thr = pick_hf_threshold(ys["val"], pva, HF_TARGET_SENSITIVITY)
    val_m, test_m = evaluate(ys["val"], pva, thr), evaluate(ys["test"], pte, thr)
    tag = "header_logreg"
    out = RESULTS_DIR / tag; out.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(dict(model=clf, vocab=vocab, columns=list(X.columns), medians=med,
                     hf_threshold=thr, C=C), HEADER_MODEL)
    coef = pd.DataFrame(clf[-1].coef_.T, index=X.columns, columns=CLASSES)
    coef.round(3).to_csv(out / "coefficients.csv")
    (out / "metrics.json").write_text(json.dumps(dict(
        tag=tag, model="header_logreg", variant="header", pretrained=False, img_hw=[0, 0],
        params=None, train_minutes=0.0, C=C, hf_threshold=thr, vocab=vocab, val=val_m, test=test_m),
        indent=2))
    plot_confusion(test_m["confusion_matrix"], f"Header features - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    dv, dt = h[h.split == "val"], h[h.split == "test"]
    save_predictions(pva, dv, out / "val_predictions.csv")
    save_predictions(pte, dt, out / "test_predictions.csv")
    hf = test_m["hf_screen"]
    print(f"[header C={C}] TEST acc {test_m['accuracy']:.3f}  bal-acc {test_m['balanced_accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}  HF AUC {hf['auc']:.3f}  "
          f"sens {hf['sensitivity']:.2f} spec {hf['specificity']:.2f}")
    top = (coef["Normal"]).sort_values()
    print("most 'reduced EF' features (lowest Normal coef):", list(top.index[:8]))


if __name__ == "__main__":
    main()
