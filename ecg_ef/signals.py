"""Digitise the ECG printout: read the exact 12-lead waveforms from the PDF's vector paths.

The ECG PDFs are vector drawings: every lead is stored as a polyline of ~700 line
segments, so the waveform can be recovered exactly (no pixel/resolution loss, no grid).

Layout (identical on all 1,047 files, 25 mm/s, 10 mm/mV):
    row baselines y = 707 / 970 / 1233 pt, column starts x = 75 / 588 / 1102 / 1615 pt,
    each cell = 513 pt = 2.5 s  ->  205.2 pt/s
    calibration pulse = 82 pt = 1 mV
    row 4 (baseline y = 1496): 10-s lead-II rhythm strip
        I   aVR  V1  V4
        II  aVL  V2  V5
        III aVF  V3  V6
        II (rhythm, 10 s)

    python -m ecg_ef.signals          # -> data/signals.npz (all ECGs, 500 Hz)
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
import pymupdf

from .config import DATA_DIR, ROOT

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
ROW_BASE = np.array([707.0, 970.0, 1233.0, 1496.0])
COL_X0 = np.array([75.0, 588.0, 1102.0, 1615.0])
CELL_PT = 513.0
PT_PER_S = CELL_PT / 2.5
PT_PER_MV = 82.0
FS = 500                       # output sampling rate (Hz)
N_LEAD = int(2.5 * FS)         # 1250 samples per 2.5-s lead
N_RHYTHM = int(10 * FS)        # 5000 samples for the 10-s rhythm strip
SIGNALS = DATA_DIR / "signals.npz"


def _polyline(d) -> np.ndarray:
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts.append((it[1].x, it[1].y)); pts.append((it[2].x, it[2].y))
    return np.asarray(pts)


def _resample(pts: np.ndarray, x0: float, n: int, base_y: float) -> np.ndarray:
    """Points in drawing order -> uniform time grid, in mV (positive = up)."""
    x = pts[:, 0] + np.arange(len(pts)) * 1e-7      # strictly increasing for interp
    order = np.argsort(x, kind="stable")
    t = x0 + np.arange(n) * (PT_PER_S / FS)
    y = np.interp(t, x[order], pts[order, 1])
    return ((base_y - y) / PT_PER_MV).astype(np.float32)


def extract_signals(pdf_path) -> tuple[np.ndarray, np.ndarray]:
    """Return (leads [12, 1250] mV at 500 Hz, rhythm lead II [5000])."""
    with pymupdf.open(str(pdf_path)) as doc:
        drawings = doc[0].get_drawings()
    cells: dict[tuple[int, int], list[np.ndarray]] = {}
    for d in drawings:
        c = d.get("color")
        if not c or max(c) > 0.3 or len(d["items"]) <= 20:
            continue                                   # grid, ticks, pulses, text boxes
        pts = _polyline(d)
        row = int(np.argmin(np.abs(ROW_BASE - np.median(pts[:, 1]))))
        col = -1 if row == 3 else int(np.clip(
            np.searchsorted(COL_X0, np.median(pts[:, 0]), side="right") - 1, 0, 3))
        cells.setdefault((row, col), []).append(pts)
    leads = np.zeros((12, N_LEAD), np.float32)
    for r in range(3):
        for c in range(4):
            parts = cells.get((r, c))
            if parts:
                parts.sort(key=lambda p: p[0, 0])
                leads[LEADS.index(GRID[r][c])] = _resample(np.concatenate(parts), COL_X0[c],
                                                            N_LEAD, ROW_BASE[r])
    rhythm = np.zeros(N_RHYTHM, np.float32)
    if (3, -1) in cells:
        parts = sorted(cells[(3, -1)], key=lambda p: p[0, 0])
        rhythm = _resample(np.concatenate(parts), COL_X0[0], N_RHYTHM, ROW_BASE[3])
    return leads, rhythm


def _one(src):
    return extract_signals(ROOT / src)


def main():
    m = pd.read_csv(DATA_DIR / "manifest.csv")
    with Pool(4) as pool:
        out = pool.map(_one, m.source, chunksize=8)
    leads = np.stack([o[0] for o in out]); rhythm = np.stack([o[1] for o in out])
    np.savez_compressed(SIGNALS, ecg_id=m.ecg_id.values, leads=leads, rhythm=rhythm)
    print(f"saved {SIGNALS}: leads {leads.shape}, rhythm {rhythm.shape}")
    print("amplitude range (mV): p1 %.2f  p99 %.2f" % tuple(np.percentile(leads, [1, 99])))


if __name__ == "__main__":
    main()
