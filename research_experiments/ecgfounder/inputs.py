"""Build ECGFounder inputs (12 x 5000, 500 Hz, 10 s) from the waveforms digitised from the PDFs.

The PDFs use the standard 3x4 + rhythm layout, so each of the 12 leads is only 2.5 s long
and the four columns are SUCCESSIVE 2.5-s windows of one 10-s recording; only lead II is
available for the whole 10 s (rhythm strip). Verified on all 1,047 ECGs
(research_experiments/results/ecgfounder_input_checks.json):
    lead II (col 1) vs rhythm 0-2.5 s                    median r = 0.9998
    2(aVF - aVR)/3 (col 2) vs rhythm 2.5-5 s             median r = 0.997  (vs 0-2.5 s: -0.04)
ECGFounder was pre-trained on simultaneous 10-s 12-lead ECGs, so every construction below is
an approximation (documented domain shift):

  aligned : each lead at its TRUE time position, zero elsewhere; lead II = full 10-s strip;
            limb leads completed over 0-5 s with Einthoven/Goldberger identities
            (0-2.5 s: aVR/aVL/aVF from I, II; 2.5-5 s: I = -2 aVR - II, III = 2 aVF - II).
  tiled   : each lead's 2.5-s segment repeated 4x (lead II = 10-s strip).  Leads are not
            simultaneous and repeats create beat discontinuities every 2.5 s.
  short   : 12 x 1250 (each lead's own 2.5 s, as the existing 1-D CNN uses them).

Pre-processing follows ECGFounder dataset.py / util.filter_bandpass: 50-Hz notch (Q 30),
4th-order Butterworth band-pass 0.67-40 Hz (filtfilt), 0.4-s median baseline removal,
then z-score.  Deviation (documented): filtering is applied per contiguous segment, and the
z-score mean/std are computed over the supported samples only, so zero-filled time is not
counted (otherwise the amplitude scale would be inflated by ~1.5x).
All of this happens in memory; nothing is written to data/.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt

FS = 500
SEG = 1250                       # 2.5 s
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
COL_OF = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1,
          "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
_NB, _NA = iirnotch(50, 30, FS)
_BB, _BA = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=FS)
_K = int(0.4 * FS) + 1           # 201 (odd)


def bandpass(x: np.ndarray) -> np.ndarray:
    """ECGFounder filter_bandpass for one 1-D segment."""
    x = filtfilt(_NB, _NA, x)
    x = filtfilt(_BB, _BA, x)
    return x - medfilt(x, kernel_size=_K)


def _zscore(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = arr[mask]
    out = np.zeros_like(arr)
    out[mask] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def build(leads: np.ndarray, rhythm: np.ndarray, mode: str) -> np.ndarray:
    """leads [12, 1250] mV (ecg_ef.signals order == ECGFounder order), rhythm [5000] -> float32."""
    L = {n: leads[i].astype(np.float64) for i, n in enumerate(LEADS)}
    II = rhythm.astype(np.float64)
    if mode == "short":
        arr = np.stack([bandpass(L[n]) for n in LEADS])
        return _zscore(arr, np.ones_like(arr, bool)).astype(np.float32)
    if mode == "tiled":
        arr = np.stack([bandpass(II) if n == "II" else np.tile(bandpass(L[n]), 4) for n in LEADS])
        return _zscore(arr, np.ones_like(arr, bool)).astype(np.float32)
    if mode != "aligned":
        raise ValueError(mode)
    arr = np.zeros((12, 4 * SEG)); mask = np.zeros_like(arr, bool)
    s0, s1 = slice(0, SEG), slice(SEG, 2 * SEG)
    limb = {
        # 0-2.5 s: I, II, III measured; augmented leads from I and II
        "I": np.r_[L["I"], -2 * L["aVR"] - II[s1]],
        "III": np.r_[L["III"], 2 * L["aVF"] - II[s1]],
        "aVR": np.r_[-(L["I"] + II[s0]) / 2, L["aVR"]],
        "aVL": np.r_[L["I"] - II[s0] / 2, L["aVL"]],
        "aVF": np.r_[II[s0] - L["I"] / 2, L["aVF"]],
    }
    for i, n in enumerate(LEADS):
        if n == "II":
            arr[i] = bandpass(II); mask[i] = True
        elif n in limb:
            arr[i, :2 * SEG] = bandpass(limb[n]); mask[i, :2 * SEG] = True
        else:
            c = COL_OF[n]
            arr[i, c * SEG:(c + 1) * SEG] = bandpass(L[n]); mask[i, c * SEG:(c + 1) * SEG] = True
    return _zscore(arr, mask).astype(np.float32)


def build_all(leads: np.ndarray, rhythm: np.ndarray, mode: str) -> np.ndarray:
    return np.stack([build(l, r, mode) for l, r in zip(leads, rhythm)])
