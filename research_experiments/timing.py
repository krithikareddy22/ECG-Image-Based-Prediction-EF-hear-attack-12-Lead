"""End-to-end CPU inference time per ECG PDF (batch size 1, 4 threads), 30 test PDFs.
Reads the PDFs read-only; no paths/names are printed or stored.
    python -m research_experiments.timing
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from research_experiments.common import RES, ROOT
from research_experiments.ecgfounder.inputs import build
from research_experiments.ecgfounder.model import load_ecgfounder
from ecg_ef.data import Normalizer
from ecg_ef.header import design_matrix, parse_header
from ecg_ef.predict import load_checkpoint
from ecg_ef.preprocess import prepare_for_model
from ecg_ef.signals import extract_signals

torch.set_num_threads(4)
m = pd.read_csv(ROOT / "data" / "manifest.csv")
paths = [ROOT / s for s in m[m.split == "test"].source.values[:30]]
rn, ck = load_checkpoint(ROOT / "models" / "resnet18_filtered.pt"); norm = Normalizer(**ck["normalizer"])
bundle = joblib.load(ROOT / "models" / "header_logreg.joblib")
ef = load_ecgfounder(); ef.eval(); ef.return_features = True
T = {k: [] for k in ("image_preprocess", "resnet18_forward", "header_parse_and_model", "signal_extract_pdf",
                     "ecgfounder_input_build", "ecgfounder_forward")}
with torch.no_grad():
    for i, p in enumerate(paths):
        t = time.perf_counter(); im = prepare_for_model(p, "filtered", (256, 512)); T["image_preprocess"].append(time.perf_counter() - t)
        x = norm(torch.from_numpy(im)[None, None].float() / 255)
        t = time.perf_counter(); rn(x); T["resnet18_forward"].append(time.perf_counter() - t)
        t = time.perf_counter()
        h = pd.DataFrame([parse_header(p)])
        X = design_matrix(h, bundle["vocab"])[bundle["columns"]].fillna(bundle["medians"])
        bundle["model"].predict_proba(X.values); T["header_parse_and_model"].append(time.perf_counter() - t)
        t = time.perf_counter(); L, R = extract_signals(p); T["signal_extract_pdf"].append(time.perf_counter() - t)
        t = time.perf_counter(); xe = build(L, R, "aligned"); T["ecgfounder_input_build"].append(time.perf_counter() - t)
        t = time.perf_counter(); ef(torch.from_numpy(xe)[None]); T["ecgfounder_forward"].append(time.perf_counter() - t)
res = {k: dict(median_ms=1000 * float(np.median(v[2:])), mean_ms=1000 * float(np.mean(v[2:]))) for k, v in T.items()}
res["n_pdfs"] = len(paths) - 2
res["pipelines_median_ms"] = {
    "image CNN (ResNet-18)": res["image_preprocess"]["median_ms"] + res["resnet18_forward"]["median_ms"],
    "header model": res["header_parse_and_model"]["median_ms"],
    "default fusion (header + ResNet-18)": res["image_preprocess"]["median_ms"] + res["resnet18_forward"]["median_ms"] + res["header_parse_and_model"]["median_ms"],
    "ECGFounder (signal extraction + build + forward)": res["signal_extract_pdf"]["median_ms"] + res["ecgfounder_input_build"]["median_ms"] + res["ecgfounder_forward"]["median_ms"],
    "header + ECGFounder fusion": res["header_parse_and_model"]["median_ms"] + res["signal_extract_pdf"]["median_ms"] + res["ecgfounder_input_build"]["median_ms"] + res["ecgfounder_forward"]["median_ms"],
}
(RES / "timing_end_to_end.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
