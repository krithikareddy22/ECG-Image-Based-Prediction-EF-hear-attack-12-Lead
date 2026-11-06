"""Reproducible inference for the phase-2 final model on ONE ECG PDF (no labels needed).

    python -m research_experiments.phase_2.best_model.predict_pdf path/to/ecg.pdf [--rule "prior, protected"]

Pipeline: printed header -> 16 numeric features (age, sex, 14 machine measurements; machine text NOT used)
          PDF vector paths -> 12-lead waveforms -> time-aligned ECGFounder input -> frozen ECGFounder (1,024-d)
          -> block standardisation (development-set statistics) -> 3-seed MLP ensemble -> class probabilities
          -> optional decision-rule calibration chosen on development OOF predictions.
The patient name is never read or printed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_experiments.common import CLASSES
from research_experiments.ecgfounder.inputs import build
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.ecgf_features import load_model, mask_gaps
from research_experiments.phase_2.p2heads import Head
from ecg_ef.header import design_matrix, parse_header
from ecg_ef.signals import extract_signals

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict(pdf, ckpt_name="FINAL_fusion_mlp.pt", rule=None):
    ck = torch.load(HERE / ckpt_name, map_location="cpu", weights_only=False)   # our own checkpoint
    h = pd.DataFrame([parse_header(pdf)])
    for c in ("summary", "statements"):
        if c not in h:
            h[c] = ""
    Xh = design_matrix(h, [])[ck["header_numeric_columns"]].astype(float)
    Xh = Xh.fillna(pd.Series(ck["header_numeric_train_medians"], index=ck["header_numeric_columns"])).values
    leads, rhythm = extract_signals(pdf)
    x = mask_gaps(build(leads, rhythm, "aligned"), geometry(str(Path(pdf).resolve())))
    m = load_model(False); out = m.first_activation(m.first_conv(torch.from_numpy(x[None])))
    for st in m.stage_list:
        out = st(out)
    f6 = out.mean(-1).numpy()
    blocks = []
    for name, B in (("header_numeric", Xh), ("ecgfounder_f6", f6)):
        s = ck["block_stats"][name]
        blocks.append((B - np.array(s["mu"])) / np.array(s["sd"]) / np.sqrt(s["dim"]))
    X = torch.tensor(np.concatenate(blocks, 1), dtype=torch.float32)
    P = []
    for sd in ck["state_dicts"]:
        head = Head(ck["in_dim"], ck["cfg"].get("loss", "ce"), "mlp", ck["cfg"].get("hidden", 64), ck["cfg"].get("dropout", 0.5))
        head.load_state_dict(sd); head.eval(); P.append(head.probs(X).numpy())
    p = np.mean(P, 0)[0]
    pred = int(p.argmax())
    if rule:
        c = ck["calibration"][rule]; prior = np.array(ck["prior"])
        pred = int((p * prior ** c).argmax()) if rule.startswith("prior") else int((np.log(np.clip(p, 1e-9, 1)) + np.array(c)).argmax())
    return dict(probabilities=dict(zip(CLASSES, p.round(4).tolist())), category=CLASSES[pred],
                reduced_ef_likelihood=float(1 - p[0]), decision_rule=rule or "argmax")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--rule", default=None)
    a = ap.parse_args()
    print(json.dumps(predict(a.pdf, rule=a.rule), indent=1))
