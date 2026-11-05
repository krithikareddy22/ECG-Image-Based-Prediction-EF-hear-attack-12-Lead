"""Phase-3 audit of the ECGFounder input masking (development set only; locked test untouched).

A  lead-wise coverage of the 'aligned' input (measured / exactly derived / not recorded), per ECG
B  dose-response: constructions that REMOVE real signal step by step (nothing is ever added or invented);
   frozen ECGFounder -> same linear head as phase 2 -> development CV (repeat 0) + label-free check
C  valid alternatives without fabrication:
     windows_mean / windows_concat : each printed 2.5-s window fed separately with ONLY the leads recorded in it
     multiview                     : aligned 12-lead embedding + 1-lead model embedding of the true 10-s lead II
D  how strongly performance depends on observed coverage (slope; extrapolation is a HYPOTHESIS)
    python -m research_experiments.phase_3.masking_audit
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from research_experiments.common import ROOT, load_meta
from research_experiments.ecgfounder.inputs import LEADS, SEG, _zscore, bandpass, build
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.ecgf_features import load_model, mask_gaps, run
from research_experiments.phase_2 import cv_features as cvf

OUT = ROOT / "research_experiments" / "phase_3"
COL = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]


def construct(leads, rhythm, mode, g):
    full = mask_gaps(build(leads, rhythm, "aligned"), g)           # measured + exactly derived, rest 0
    keep = np.zeros((12, 4 * SEG), bool)
    if mode == "col_only":            # each lead only in its own printed window, no derivation, II only 0-2.5 s
        arr = np.zeros((12, 4 * SEG)); mask = np.zeros_like(arr, bool)
        for i, n in enumerate(LEADS):
            c = COL[n]; arr[i, c * SEG:(c + 1) * SEG] = bandpass(leads[i].astype(np.float64)); mask[i, c * SEG:(c + 1) * SEG] = True
        return _zscore(arr, mask).astype(np.float32)
    if mode == "limb_only":
        keep[[LEADS.index(n) for n in LIMB]] = True
    elif mode == "precordial_plus_II":
        keep[[LEADS.index(n) for n in LEADS if n.startswith("V")] + [1]] = True
    elif mode == "precordial_only":
        keep[[LEADS.index(n) for n in LEADS if n.startswith("V")]] = True
    elif mode == "leadII_2p5":
        keep[1, :SEG] = True
    else:
        raise ValueError(mode)
    raw = np.zeros_like(full); m = (full != 0) & keep
    # re-z-score on the kept samples only (same rule as the aligned construction)
    vals = full[m]; raw[m] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return raw.astype(np.float32)


def windows(leads, rhythm, g):
    """4 inputs of 12 x 1250: window c contains only the leads recorded (or exactly derived) in that window."""
    full = mask_gaps(build(leads, rhythm, "aligned"), g)
    return [full[:, c * SEG:(c + 1) * SEG] for c in range(4)]


def main():
    torch.set_num_threads(1)
    meta = load_meta()
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[meta.ecg_id].source
    with Pool(1) as p:
        G = p.map(geometry, src.values, chunksize=32)
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}; ii = np.array([idx[e] for e in meta.ecg_id])
    L, R = z["leads"][ii], z["rhythm"][ii]
    # A. coverage table of the aligned input
    A = np.stack([mask_gaps(build(l, r, "aligned"), g) != 0 for l, r, g in zip(L, R, G)])
    cov = {n: {"seconds_measured": (10.0 if n == "II" else 2.5), "seconds_exactly_derived": (2.5 if n in ("I", "III", "aVR", "aVL", "aVF") else 0.0),
               "seconds_not_recorded": 10.0 - (10.0 if n == "II" else 2.5) - (2.5 if n in ("I", "III", "aVR", "aVL", "aVF") else 0.0),
               "observed_fraction_median_over_ecgs": float(np.median(A[:, i].mean(1)))} for i, n in enumerate(LEADS)}
    per_ecg = A.reshape(len(A), -1).mean(1)
    rep = {"A_coverage": dict(per_lead=cov, observed_fraction_per_ecg=dict(median=float(np.median(per_ecg)), min=float(per_ecg.min()),
                                                                            max=float(per_ecg.max())))}
    # B / C. features for the new constructions
    m12, m1 = load_model(False), load_model(True)
    ids = np.asarray(meta.ecg_id.tolist(), dtype="U32")
    frac = {}
    for mode in ("col_only", "limb_only", "precordial_plus_II", "precordial_only", "leadII_2p5"):
        X = np.stack([construct(l, r, mode, g) for l, r, g in zip(L, R, G)])
        if (OUT / "models" / f"ecgf_{mode}_features.npz").exists():      # idempotent resume
            frac[mode] = float((X != 0).mean()); continue
        frac[mode] = float((X != 0).mean())
        out = run(m12, X)
        np.savez(OUT / "models" / f"ecgf_{mode}_features.npz", ecg_id=ids, **{k: v for k, v in out.items() if v is not None})
        print(mode, "observed fraction", round(frac[mode], 3), flush=True)
    if (OUT / "models" / "ecgf_windows_features.npz").exists() and (OUT / "results" / "masking_audit_features.json").exists():
        print("masking features already complete"); return
    W = [windows(l, r, g) for l, r, g in zip(L, R, G)]
    feats = []
    for c in range(4):
        Xc = np.stack([w[c] for w in W]).astype(np.float32)
        feats.append(run(m12, Xc)["f6"])
    np.savez(OUT / "models" / "ecgf_windows_features.npz", ecg_id=ids, f6=np.mean(feats, 0), f6cat=np.concatenate(feats, 1))
    rep["observed_fraction"] = dict(frac, aligned=float(A.mean()))
    (OUT / "results" / "masking_audit_features.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["observed_fraction"], indent=1))


if __name__ == "__main__":
    main()
