"""Clinically motivated ECG features (label-free, deterministic, per ECG; no fitting -> no leakage).

From the PRINTED header (machine measurements):
  qrs_t_angle   frontal QRS-T angle |QRS axis - T axis| folded to 0-180 deg (wide angle ~ LV dysfunction)
  jt, jtc       JT = QT - QRSD, JTc = QTcB - QRSD (repolarisation independent of conduction delay)
  qrsd_ge120    QRS duration >= 120 ms (LBBB / IVCD, common in reduced EF)
From the DIGITISED waveforms (mV; each lead's own printed 2.5-s window, lead II rhythm strip 10 s):
  limb_qrs_max_p2p, limb_qrs_sum, limb_low_voltage (<0.5 mV)            low voltage ~ myocardial loss / effusion
  prec_qrs_max_p2p, prec_qrs_sum, prec_low_voltage (<1.0 mV)
  sokolow_lyon  S(V1) + max(R(V5), R(V6));  cornell  R(aVL) + S(V3)      LV hypertrophy voltage criteria
  log_rs_V1..V6, transition_lead                                        R-wave progression (poor progression ~ prior anterior MI)
  rr_sd, rr_rmssd, rr_cv, rr_max_min_ratio, n_beats_10s                 rhythm (AF, ectopy) from the 10-s strip
  t_amp_II, t_amp_V5, t_amp_V6, lateral_t_inversion                      T-wave amplitude 200-360 ms after each R peak
    python -m research_experiments.phase_3.clinical_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from research_experiments.common import ROOT, load_meta
from research_experiments.ecgfounder.inputs import LEADS, bandpass
from research_experiments.phase_3.p3common import MOD3, guard3

FS = 500


def rpeaks(x, min_dist=0.3):
    y = x if x.max() > -x.min() else -x
    pk, _ = find_peaks(y, distance=int(min_dist * FS), prominence=0.5 * np.percentile(np.abs(y), 99))
    return pk


def t_amp(x, pk):
    vals = [np.median(x[p + int(0.2 * FS):p + int(0.36 * FS)]) for p in pk if p + int(0.36 * FS) < len(x)]
    return float(np.median(vals)) if vals else np.nan


def features(leads, rhythm):
    F = {n: bandpass(leads[i].astype(np.float64)) for i, n in enumerate(LEADS)}
    II = bandpass(rhythm.astype(np.float64))
    R = {n: float(np.percentile(F[n], 99.5)) for n in LEADS}; S = {n: float(-np.percentile(F[n], 0.5)) for n in LEADS}
    p2p = {n: R[n] + S[n] for n in LEADS}
    limb = ["I", "II", "III", "aVR", "aVL", "aVF"]; prec = [f"V{k}" for k in range(1, 7)]
    f = dict(limb_qrs_max_p2p=max(p2p[n] for n in limb), limb_qrs_sum=sum(p2p[n] for n in ["I", "II", "III"]),
             prec_qrs_max_p2p=max(p2p[n] for n in prec), prec_qrs_sum=sum(p2p[n] for n in prec),
             sokolow_lyon=S["V1"] + max(R["V5"], R["V6"]), cornell=R["aVL"] + S["V3"])
    f["limb_low_voltage"] = float(f["limb_qrs_max_p2p"] < 0.5); f["prec_low_voltage"] = float(f["prec_qrs_max_p2p"] < 1.0)
    rs = [np.log((R[n] + 1e-3) / (S[n] + 1e-3)) for n in prec]
    for k, v in enumerate(rs, 1):
        f[f"log_rs_V{k}"] = float(v)
    f["transition_lead"] = float(next((k for k, v in enumerate(rs, 1) if v > 0), 7))
    pk = rpeaks(II)
    rr = np.diff(pk) / FS if len(pk) >= 3 else np.array([np.nan])
    f.update(rr_sd=float(np.std(rr)), rr_rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan,
             rr_cv=float(np.std(rr) / np.mean(rr)), rr_max_min_ratio=float(np.max(rr) / np.min(rr)), n_beats_10s=float(len(pk)))
    f["t_amp_II"] = t_amp(II, pk)
    for n in ("V5", "V6"):
        f[f"t_amp_{n}"] = t_amp(F[n], rpeaks(F[n]))
    f["lateral_t_inversion"] = float(np.nanmin([f["t_amp_V5"], f["t_amp_V6"]]) < -0.05) if np.isfinite([f["t_amp_V5"], f["t_amp_V6"]]).any() else np.nan
    return f


def main():
    meta = load_meta()
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    rows = [features(z["leads"][idx[e]], z["rhythm"][idx[e]]) for e in meta.ecg_id]
    C = pd.DataFrame(rows); C.insert(0, "ecg_id", meta.ecg_id.values)
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[meta.ecg_id]
    d = np.abs(H.QRS_axis.values - H.T_axis.values) % 360
    C["qrs_t_angle"] = np.where(d > 180, 360 - d, d)
    C["jt"] = H.QT.values - H.QRSD.values; C["jtc"] = H.QTcB.values - H.QRSD.values
    C["qrsd_ge120"] = (H.QRSD.values >= 120).astype(float)
    C.to_csv(guard3(MOD3 / "clinical_features.csv"), index=False, float_format="%.5f")
    print(C.drop(columns="ecg_id").describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print("missing per feature:", C.isna().sum()[C.isna().sum() > 0].to_dict())


if __name__ == "__main__":
    main()
