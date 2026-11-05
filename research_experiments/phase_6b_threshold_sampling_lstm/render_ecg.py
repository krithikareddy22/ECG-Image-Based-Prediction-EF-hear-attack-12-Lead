"""Render 12-lead ECG waveforms as printout-style images in the project's PDF layout.

Layout (as the hospital PDFs): 3 rows x 4 columns of 2.5-s segments
    I   aVR  V1  V4
    II  aVL  V2  V5
    III aVF  V3  V6
taken from successive 2.5-s windows of one 10-s recording (column c = seconds 2.5c-2.5(c+1)), plus a 10-s lead II
rhythm strip. Paper speed 25 mm/s, gain 10 mm/mV, 1-mm pink minor grid, 5-mm major grid, calibration pulse.

Input: an array (12, n_samples) in mV at `fs` Hz, lead order I, II, III, aVR, aVL, aVF, V1-V6, covering 10 s.
Works for any source (CSV with one column per lead, NumPy, EchoNext arrays after un-normalising). Label-free.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.render_ecg --csv ecg.csv --fs 500 --out ecg.png
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
MM_S, MM_MV = 25.0, 10.0                        # paper speed, gain
ROW_MM = 30.0                                   # vertical space per row (3 mV)
PULSE_MM = 5.0                                  # calibration pulse width before each row


def render(sig, fs, out, dpi=150, title=None):
    sig = np.asarray(sig, float)
    assert sig.shape[0] == 12, "expected 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)"
    n10 = int(round(10 * fs)); sig = sig[:, :n10]
    seg = n10 // 4
    W = PULSE_MM + 10 * MM_S + 2                # mm
    Hh = 4 * ROW_MM + 4
    fig = plt.figure(figsize=(W / 25.4, Hh / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")
    for x in np.arange(0, W + 0.01, 1.0):
        ax.axvline(x, color="#f4b6c2" if x % 5 else "#e8798f", lw=0.25 if x % 5 else 0.6, zorder=0)
    for y in np.arange(0, Hh + 0.01, 1.0):
        ax.axhline(y, color="#f4b6c2" if y % 5 else "#e8798f", lw=0.25 if y % 5 else 0.6, zorder=0)
    idx = {n: i for i, n in enumerate(LEADS)}

    def trace(x0, y0, v):
        t = np.arange(len(v)) / fs * MM_S
        ax.plot(x0 + t, y0 + v * MM_MV, color="black", lw=0.6, zorder=2)

    def pulse(y0):
        ax.plot([0.5, 1, 1, 4, 4, PULSE_MM], [y0, y0, y0 + MM_MV, y0 + MM_MV, y0, y0], color="black", lw=0.6, zorder=2)

    for r, row in enumerate(GRID + [["II"]]):
        y0 = Hh - (r + 0.6) * ROW_MM
        pulse(y0)
        if r < 3:
            for c, name in enumerate(row):
                x0 = PULSE_MM + c * 2.5 * MM_S
                trace(x0, y0, sig[idx[name], c * seg:(c + 1) * seg])
                ax.text(x0 + 1, y0 + 8, name, fontsize=6, zorder=3)
                if c:
                    ax.plot([x0, x0], [y0 - 2, y0 + 2], color="black", lw=0.6, zorder=2)
        else:
            trace(PULSE_MM, y0, sig[idx["II"]])
            ax.text(PULSE_MM + 1, y0 + 8, "II", fontsize=6, zorder=3)
    ax.text(W - 60, 1.5, "25 mm/s   10 mm/mV", fontsize=5, zorder=3)
    if title:
        ax.text(2, Hh - 3, title, fontsize=6, zorder=3)
    fig.savefig(out, dpi=dpi); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with 12 columns (I, II, III, aVR, aVL, aVF, V1-V6) in mV, one row per sample")
    ap.add_argument("--fs", type=float, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(a.csv)
    render(df[LEADS].to_numpy().T, a.fs, a.out)
