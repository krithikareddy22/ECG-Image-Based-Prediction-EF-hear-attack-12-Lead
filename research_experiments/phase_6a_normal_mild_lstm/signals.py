"""LSTM inputs from the digitised printout waveforms (development ECGs only; label-free; no clinical data).

Source: data/signals.npz at 500 Hz - 12 leads x 2.5 s, each lead from its own printed column (I/II/III 0-2.5 s,
aVR/aVL/aVF 2.5-5 s, V1-V3 5-7.5 s, V4-V6 7.5-10 s), and the 10-s lead II rhythm strip, which is simultaneous with
all four columns. Processing per segment: ECGFounder band-pass (0.67-40 Hz Butterworth, 50 Hz notch, 0.4-s median
baseline removal), then decimation 500 -> 100 Hz (every 5th sample; the band-pass removes content above 40 Hz, below
the 50 Hz Nyquist limit). Amplitudes stay in mV (no per-ECG normalisation, so voltage information is kept), clipped
at +-5 mV.

No fabrication: on the common 10-s axis (1,000 steps) each lead occupies ONLY its recorded 2.5-s window and is zero
elsewhere; no lead is derived from others or copied into windows where it was not recorded. A mask channel per signal
marks recorded samples (1) vs not recorded / missing (0). Missing = a lead that is >= 90 % flat after digitisation,
or a constant run of >= 20 ms (10 samples at 500 Hz) at the START or END of a segment (undrawn part of the printed
trace). Interior constant runs are kept: vector digitisation draws straight segments (e.g. the baseline) as repeated
values, which are recorded signal, not missing data (plan Amendment 1).
  aligned  (N, 1000, 26): 12 lead channels + rhythm II, then their 13 masks
  rhythm   (N, 1000, 1):  lead II rhythm strip only
Cache (git-ignored): data/private/lstm_inputs.npz; summary (aggregates): results/lstm_inputs.json
    python -m research_experiments.phase_6a_normal_mild_lstm.signals
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_experiments.common import ROOT
from research_experiments.ecgfounder.inputs import COL_OF, LEADS, bandpass
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import IDS, PRIV, RES, guard

FS, DEC, SEG = 500, 5, 250                 # input Hz, decimation factor, steps per 2.5-s column at 100 Hz
T_STEPS = 4 * SEG
CACHE = PRIV / "lstm_inputs.npz"


def edge_mask(x, run=10):
    """0 for a constant run of >= `run` samples (500 Hz) at the start or the end of the segment, else 1."""
    same = np.abs(np.diff(x)) < 1e-9
    m = np.ones(len(x), bool)
    a = int(np.argmax(~same)) if (~same).any() else len(same)          # length of the leading constant run
    if a + 1 >= run:
        m[:a + 1] = False
    b = int(np.argmax(~same[::-1])) if (~same).any() else len(same)    # length of the trailing constant run
    if b + 1 >= run:
        m[len(x) - b - 1:] = False
    return m


def seg(x):
    """band-pass at 500 Hz -> (decimated signal, decimated mask)."""
    if (np.abs(np.diff(x)) < 1e-9).mean() >= 0.90:          # >= 90 % flat: lead missing
        return np.zeros(len(x) // DEC, np.float32), np.zeros(len(x) // DEC, np.float32)
    m = edge_mask(x)
    y = np.clip(bandpass(x.astype(np.float64)), -5, 5)
    md = m.reshape(-1, DEC).min(1)
    return (y[::DEC] * md).astype(np.float32), md.astype(np.float32)


def build():
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[IDS].values      # development rows only
    L, R = z["leads"][pos], z["rhythm"][pos]
    n = len(IDS)
    X = np.zeros((n, T_STEPS, 26), np.float32)
    for i in range(n):
        for j, name in enumerate(LEADS):
            s, m = seg(L[i, j]); c = COL_OF[name]
            X[i, c * SEG:(c + 1) * SEG, j] = s; X[i, c * SEG:(c + 1) * SEG, 13 + j] = m
        s, m = seg(R[i]); X[i, :, 12] = s; X[i, :, 25] = m
    return X


def load(kind):
    z = np.load(CACHE)
    assert (z["ecg_id"] == np.asarray(IDS, dtype="U32")).all()
    X = z["X"]
    return X if kind == "aligned" else X[:, :, 12:13].copy()


def main():
    if CACHE.exists():
        print("exists"); return
    X = build()
    guard(PRIV).mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ecg_id=np.asarray(IDS, dtype="U32"), X=X)
    masks = X[:, :, 13:]
    lead_recorded = masks[:, :, :12].reshape(len(X), 4, SEG, 12).max(2).max(1)          # lead present anywhere
    summ = dict(n_ecgs=len(X), shape=list(X.shape), sampling_hz=FS // DEC, seconds=T_STEPS / (FS // DEC),
                channels=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "rhythm_II"] +
                         [f"mask_{c}" for c in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "rhythm_II"]],
                missing_lead_segments=int((lead_recorded == 0).sum()), lead_segments=int(lead_recorded.size),
                recorded_fraction_within_windows=float(masks[:, :, :12].sum() / (len(X) * SEG * 12)),
                rhythm_recorded_fraction=float(masks[:, :, 12].mean()),
                amplitude_mV_p1_p99=[float(np.percentile(X[:, :, :13][masks > 0], q)) for q in (1, 99)])
    (guard(RES) / "lstm_inputs.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
