"""Phase-2 audit of the PDF -> waveform reconstruction used as ECGFounder input.

Checks on all 1,047 PDFs (geometry only; no text other than the numeric header fields that
ecg_ef.header already parsed is used; nothing identifying is printed or stored):
  1. calibration: minor/major grid spacing (expected 1 mm = 8.208 pt, 5 mm = 41.04 pt at the
     layout's 205.2 pt/s and 25 mm/s), calibration-pulse height (expected 82 pt = 1 mV)
  2. sampling: native vector-point density per lead (points per second) before resampling to 500 Hz
  3. duration: x-coverage of each lead cell (2.5 s) and of the rhythm strip (10 s); fraction of
     samples outside the drawn polyline (np.interp clamping)
  4. lead identity / polarity: dominant QRS polarity per lead, R/S progression V1->V6
  5. external agreement with the ECG machine's own printed measurements:
       heart rate from the 10-s lead-II strip vs printed Rate (time-scale check)
       QRS axis from I and aVF (aVF derived from I, II in 0-2.5 s) vs printed QRS axis
  6. derived-lead validity: Einthoven III = II - I; Goldberger aVR + aVL + aVF = 0;
     cross-column 2(aVF - aVR)/3 = II(2.5-5 s); continuity of the derived I/III at t = 2.5 s
  7. composition of the 'aligned' ECGFounder input (measured / exactly derived / not recorded)

    python -m research_experiments.phase_2.audit_waveform
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pymupdf
from scipy.signal import find_peaks

from research_experiments.common import ROOT, load_meta
from research_experiments.ecgfounder.inputs import bandpass
from ecg_ef.signals import COL_X0, GRID, LEADS, PT_PER_MV, PT_PER_S, ROW_BASE, _polyline

OUT = ROOT / "research_experiments" / "phase_2" / "results"


def geometry(src):
    with pymupdf.open(str(ROOT / src)) as doc:
        dr = doc[0].get_drawings()
    minor_x, major_x, pulses, cells = [], [], [], {}
    for d in dr:
        c = d.get("color")
        if not c:
            continue
        if c[0] > 0.9 and len(d["items"]) == 1 and d["items"][0][0] == "l":   # pink grid line
            p1, p2 = d["items"][0][1], d["items"][0][2]
            if abs(p1.x - p2.x) < 1e-3:                                     # vertical
                (minor_x if c[1] > 0.6 else major_x).append(p1.x)
        elif max(c) <= 0.3 and 3 <= len(d["items"]) <= 20:
            pts = _polyline(d)
            if len(pts):
                pulses.append(float(pts[:, 1].max() - pts[:, 1].min()))
        elif max(c) <= 0.3 and len(d["items"]) > 20:
            pts = _polyline(d)
            row = int(np.argmin(np.abs(ROW_BASE - np.median(pts[:, 1]))))
            col = -1 if row == 3 else int(np.clip(np.searchsorted(COL_X0, np.median(pts[:, 0]), side="right") - 1, 0, 3))
            cells.setdefault((row, col), []).append(pts)
    out = {}
    sp = lambda xs: float(np.median(np.diff(np.unique(np.round(xs, 2))))) if len(xs) > 2 else np.nan
    out["minor_spacing_pt"], out["major_spacing_pt"] = sp(minor_x), sp(major_x)
    out["pulse_height_pt"] = float(np.median(pulses)) if pulses else np.nan
    out["n_pulses"] = len(pulses)
    for (r, c), parts in cells.items():
        P = np.concatenate(parts)
        name = "rhythm" if r == 3 else GRID[r][c]
        x0 = COL_X0[0] if r == 3 else COL_X0[c]
        dur_pt = 4 * 513.0 if r == 3 else 513.0
        out[f"{name}_n_points"] = int(len(P) / 2)          # line segments -> distinct vertices
        out[f"{name}_cover"] = float((min(P[:, 0].max(), x0 + dur_pt) - max(P[:, 0].min(), x0)) / dur_pt)
        out[f"{name}_start_gap_s"] = float(max(0.0, P[:, 0].min() - x0) / PT_PER_S)
        out[f"{name}_end_gap_s"] = float(max(0.0, x0 + dur_pt - P[:, 0].max()) / PT_PER_S)
    out["n_cells"] = len(cells)
    return out


def corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-12)


def main():
    meta = load_meta()
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[meta.ecg_id].source
    with Pool(4) as pool:
        G = pd.DataFrame(pool.map(geometry, src.values, chunksize=16))
    rep = {}
    rep["calibration"] = {
        "minor_grid_spacing_pt_median": float(G.minor_spacing_pt.median()), "expected_1mm_pt": PT_PER_S / 25,
        "major_grid_spacing_pt_median": float(G.major_spacing_pt.median()), "expected_5mm_pt": 5 * PT_PER_S / 25,
        "pulse_height_pt_median": float(G.pulse_height_pt.median()), "pulse_height_pt_range": [float(G.pulse_height_pt.min()), float(G.pulse_height_pt.max())],
        "expected_pulse_pt_(1mV)": PT_PER_MV, "files_with_4_pulses": int((G.n_pulses == 4).sum()),
        "files_with_13_cells": int((G.n_cells == 13).sum()), "n_files": len(G)}
    dens, cov = {}, {}
    for n in LEADS + ["rhythm"]:
        dur = 10.0 if n == "rhythm" else 2.5
        dens[n] = float((G[f"{n}_n_points"] / dur).median())
        cov[n] = {"median_cover": float(G[f"{n}_cover"].median()), "min_cover": float(G[f"{n}_cover"].min()),
                  "max_start_gap_s": float(G[f"{n}_start_gap_s"].max()), "max_end_gap_s": float(G[f"{n}_end_gap_s"].max())}
    rep["native_points_per_second_median"] = dens
    rep["duration_coverage"] = cov

    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    ii = np.array([idx[e] for e in meta.ecg_id])
    L, R = z["leads"][ii].astype(np.float64), z["rhythm"][ii].astype(np.float64)
    F = np.stack([[bandpass(l) for l in ecg] for ecg in L])          # 0.67-40 Hz, baseline removed
    pos = (F.max(-1) > -F.min(-1))
    rep["dominant_positive_QRS_fraction"] = {n: float(pos[:, i].mean()) for i, n in enumerate(LEADS)}
    rs = np.log(F.max(-1) + 1e-3) - np.log(-F.min(-1) + 1e-3)
    rep["median_log_R_over_S_V1_to_V6"] = [float(np.median(rs[:, LEADS.index(f"V{k}")])) for k in range(1, 7)]

    h = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[meta.ecg_id]
    # heart rate from the 10-s rhythm strip
    hr = []
    for r in R:
        x = bandpass(r); x = x if x.max() > -x.min() else -x
        pk, _ = find_peaks(x, distance=int(0.3 * 500), prominence=0.5 * np.percentile(np.abs(x), 99))
        hr.append(60 * (len(pk) - 1) / ((pk[-1] - pk[0]) / 500) if len(pk) >= 3 else np.nan)
    hr = np.array(hr); ok = np.isfinite(hr) & np.isfinite(h.Rate.values)
    rep["heart_rate_vs_printed"] = {"n": int(ok.sum()), "pearson_r": float(np.corrcoef(hr[ok], h.Rate.values[ok])[0, 1]),
                                    "median_abs_diff_bpm": float(np.median(np.abs(hr[ok] - h.Rate.values[ok]))),
                                    "within_5bpm": float(np.mean(np.abs(hr[ok] - h.Rate.values[ok]) <= 5))}
    # QRS axis from net QRS amplitude of I and aVF (aVF = II - I/2, simultaneous 0-2.5 s)
    I0, II0 = F[:, 0], bandpass_matrix(R[:, :1250])
    aVF0 = II0 - I0 / 2
    net = lambda s: s.max(-1) + s.min(-1)
    ax = np.degrees(np.arctan2(net(aVF0), net(I0)))
    pa = h.QRS_axis.values; ok = np.isfinite(pa)
    d = np.abs((ax[ok] - pa[ok] + 180) % 360 - 180)
    rep["qrs_axis_vs_printed"] = {"n": int(ok.sum()), "median_abs_diff_deg": float(np.median(d)),
                                  "within_30deg": float(np.mean(d <= 30)),
                                  "circular_corr": float(np.corrcoef(np.cos(np.radians(ax[ok])), np.cos(np.radians(pa[ok])))[0, 1])}
    # derived-lead identities
    I, II, III, aVR, aVL, aVF = (L[:, k] for k in range(6))
    rel = lambda res, ref: np.sqrt((res ** 2).mean(-1)) / (np.sqrt((ref ** 2).mean(-1)) + 1e-9)
    c0 = lambda s: s - np.median(s, -1, keepdims=True)
    rep["einthoven_col1_III_eq_II_minus_I"] = {"median_r": float(np.median(corr(III, II - I))),
                                               "p5_r": float(np.percentile(corr(III, II - I), 5)),
                                               "median_rel_rms_residual": float(np.median(rel(c0(III) - c0(II - I), c0(III))))}
    rep["goldberger_col2_sum_zero"] = {"median_r_aVR_vs_-(aVL+aVF)": float(np.median(corr(aVR, -(aVL + aVF)))),
                                       "median_rel_rms_residual": float(np.median(rel(c0(aVR + aVL + aVF), c0(aVR))))}
    II1 = R[:, 1250:2500]
    rep["cross_column_2(aVF-aVR)/3_vs_rhythmII_2.5-5s"] = {"median_r": float(np.median(corr(2 * (aVF - aVR) / 3, II1))),
                                                           "p5_r": float(np.percentile(corr(2 * (aVF - aVR) / 3, II1), 5)),
                                                           "median_r_wrong_window": float(np.median(corr(2 * (aVF - aVR) / 3, R[:, :1250])))}
    # continuity of derived I/III at the 2.5-s junction (raw mV): jump vs typical sample step
    I_der, III_der = -2 * aVR - II1, 2 * aVF - II1
    step = np.median(np.abs(np.diff(I, axis=-1)), -1)
    rep["junction_jump_over_median_step"] = {"I": float(np.median(np.abs(I_der[:, 0] - I[:, -1]) / (step + 1e-6))),
                                             "III": float(np.median(np.abs(III_der[:, 0] - III[:, -1]) / (np.median(np.abs(np.diff(III, axis=-1)), -1) + 1e-6))),
                                             "I_jump_mV_median": float(np.median(np.abs(I_der[:, 0] - I[:, -1])))}
    meas, der, zero = 10 + 5 * 2.5 + 6 * 2.5, 5 * 2.5, 0
    rep["aligned_input_composition_lead_seconds_of_120"] = {"measured": 10 + 5 * 2.5 + 6 * 2.5, "exactly_derived": 5 * 2.5,
                                                            "not_recorded_zero": 120 - (10 + 5 * 2.5 + 6 * 2.5) - 5 * 2.5}
    (OUT / "audit_waveform.json").write_text(json.dumps(rep, indent=2))
    G.describe().T.to_csv(OUT / "audit_waveform_geometry_summary.csv")
    print(json.dumps(rep, indent=1))


def bandpass_matrix(X):
    return np.stack([bandpass(x) for x in X])


if __name__ == "__main__":
    main()
