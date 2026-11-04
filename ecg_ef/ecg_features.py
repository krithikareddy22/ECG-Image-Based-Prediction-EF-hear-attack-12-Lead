"""Engineered 12-lead ECG waveform features, following Ning et al. (arXiv 2604.25942, 2026):
signal preprocessing + clinically motivated PQRST features + statistical / frequency-
domain time-series descriptors, for use with gradient boosting (ecg_ef/tabular.py).

Input: the waveforms digitised from the PDF vector paths (ecg_ef/signals.py), 500 Hz,
12 leads x 2.5 s (standard 3x4 layout: column c covers seconds 2.5c-2.5(c+1)) plus the
simultaneous 10-s lead-II rhythm strip. Verified on this data: lead II of column 0 equals
the first 2.5 s of the rhythm strip (r = 1.000) and Einthoven's law I + III = II holds
(r = 0.994).

Per ECG (~260 features):
  * preprocessing   0.5-Hz high-pass Butterworth (order 5), as in the paper. Amplitudes are
                    kept in mV (not z-scored) because voltage itself is informative here.
  * rhythm          R peaks on the 10-s strip: RR mean/SD/min/max, RMSSD, heart rate, beats
  * per lead        median beat aligned on those R peaks -> R and S amplitude, QRS
                    peak-to-peak / area / net area, ST level (R+80-120 ms), T max/min/area;
                    signal SD, peak-to-peak, skewness, kurtosis, zero-crossing rate,
                    spectral entropy, dominant frequency, band-power fractions
  * global          Sokolow-Lyon, Cornell voltage, total QRS voltage, low-voltage flags,
                    frontal QRS axis (sin/cos)

    python -m ecg_ef.ecg_features        # -> data/ecg_features.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew

from .config import DATA_DIR
from .signals import FS, GRID, LEADS, N_LEAD, SIGNALS

FEATURES_CSV = DATA_DIR / "ecg_features.csv"
SOS_HP = signal.butter(5, 0.5, "highpass", fs=FS, output="sos")
SOS_QRS = signal.butter(2, [5, 15], "bandpass", fs=FS, output="sos")
LEAD_COL = {lead: c for row in GRID for c, lead in enumerate(row)}
LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
PRE, POST = int(0.25 * FS), int(0.45 * FS)          # beat window around the R peak
ms = lambda t: int(round(t / 1000 * FS))             # milliseconds -> samples


def highpass(x):
    return signal.sosfiltfilt(SOS_HP, x, axis=-1)


def r_peaks(x: np.ndarray) -> np.ndarray:
    """Simple Pan-Tompkins-style detector on the 10-s rhythm strip (sample indices)."""
    b = signal.sosfiltfilt(SOS_QRS, x)
    win = ms(120)
    e = np.convolve(np.diff(b, prepend=b[0]) ** 2, np.ones(win) / win, mode="same")
    peaks, _ = signal.find_peaks(e, distance=ms(300), height=0.3 * np.percentile(e, 99))
    out = []
    for p in peaks:                                   # refine to the largest deflection
        lo = max(0, p - ms(60))
        out.append(lo + int(np.argmax(np.abs(x[lo:p + ms(60)]))))
    return np.unique(np.asarray(out, int))


def template_features(beat: np.ndarray, prefix: str) -> dict:
    """Features of a median beat sampled from R-250 ms to R+450 ms."""
    r = PRE
    seg = lambda a, b: beat[r + ms(a): r + ms(b)]
    base = np.median(seg(-100, -60))                  # PR segment as isoelectric level
    qrs = seg(-60, 80) - base
    f = {
        "r_amp": seg(-60, 60).max() - base,
        "s_amp": seg(-60, 80).min() - base,
        "qrs_area": np.abs(qrs).sum() / FS,
        "qrs_net": qrs.sum() / FS,
        "st": seg(80, 120).mean() - base,
        "t_max": seg(150, 400).max() - base,
        "t_min": seg(150, 400).min() - base,
        "t_area": (seg(150, 400) - base).sum() / FS,
    }
    f["qrs_pp"] = f["r_amp"] - f["s_amp"]
    return {f"{prefix}_{k}": float(v) for k, v in f.items()}


def stat_features(x: np.ndarray, prefix: str) -> dict:
    freqs, psd = signal.welch(x, fs=FS, nperseg=min(512, len(x)))
    p = psd / psd.sum() if psd.sum() > 0 else psd
    band = lambda a, b: float(p[(freqs >= a) & (freqs < b)].sum())
    f = {
        "sd": x.std(), "pp": np.ptp(x), "skew": skew(x), "kurt": kurtosis(x),
        "zcr": np.mean(np.abs(np.diff(np.sign(x))) > 0),
        "spec_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
        "dom_freq": float(freqs[np.argmax(psd)]),
        "band_0_5": band(0.5, 5), "band_5_15": band(5, 15), "band_15_40": band(15, 40),
    }
    return {f"{prefix}_{k}": float(v) for k, v in f.items()}


def beat_template(x: np.ndarray, peaks: np.ndarray) -> np.ndarray | None:
    beats = [x[p - PRE: p + POST] for p in peaks if p - PRE >= 0 and p + POST <= len(x)]
    return np.median(np.stack(beats), 0) if beats else None


def extract(leads: np.ndarray, rhythm: np.ndarray) -> dict:
    leads, rhythm = highpass(leads), highpass(rhythm)
    peaks = r_peaks(rhythm)
    f: dict = {}
    rr = np.diff(peaks) / FS
    if len(rr) >= 2:
        f.update(rr_mean=rr.mean(), rr_sd=rr.std(), rr_min=rr.min(), rr_max=rr.max(),
                 rr_rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2))), hr=60 / rr.mean())
    f["n_beats"] = len(peaks)
    tpl = beat_template(rhythm, peaks)
    if tpl is not None:
        f.update(template_features(tpl, "rhythmII"))
    f.update(stat_features(rhythm, "rhythmII"))
    for k, lead in enumerate(LEADS):
        x = leads[k]
        off = LEAD_COL[lead] * N_LEAD                 # where this lead's window starts
        local = peaks[(peaks >= off) & (peaks < off + N_LEAD)] - off
        tpl = beat_template(x, local)
        if tpl is not None:
            f.update(template_features(tpl, lead))
        f.update(stat_features(x, lead))
    g = lambda lead, k: f.get(f"{lead}_{k}", np.nan)
    f["sokolow_lyon"] = abs(g("V1", "s_amp")) + max(g("V5", "r_amp"), g("V6", "r_amp"))
    f["cornell"] = g("aVL", "r_amp") + abs(g("V3", "s_amp"))
    f["qrs_sum"] = sum(g(l, "qrs_pp") for l in LEADS)
    f["limb_max_pp"] = max(g(l, "qrs_pp") for l in LIMB)
    f["prec_max_pp"] = max(g(l, "qrs_pp") for l in PRECORDIAL)
    f["low_volt_limb"] = float(f["limb_max_pp"] < 0.5)
    f["low_volt_prec"] = float(f["prec_max_pp"] < 1.0)
    axis = np.arctan2(g("aVF", "qrs_net"), g("I", "qrs_net"))
    f["qrs_axis_sin"], f["qrs_axis_cos"] = np.sin(axis), np.cos(axis)
    return {k: float(v) for k, v in f.items()}


def main():
    z = np.load(SIGNALS, allow_pickle=True)
    rows = []
    for i, eid in enumerate(z["ecg_id"]):
        rows.append({"ecg_id": eid, **extract(z["leads"][i].astype(float),
                                              z["rhythm"][i].astype(float))})
    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"saved {FEATURES_CSV}: {len(df)} ECGs x {df.shape[1] - 1} features; "
          f"missing values {df.isna().mean().mean():.1%}")
    print("beats per 10 s: median %d (min %d, max %d)" % (df.n_beats.median(), df.n_beats.min(),
                                                         df.n_beats.max()))


if __name__ == "__main__":
    main()
