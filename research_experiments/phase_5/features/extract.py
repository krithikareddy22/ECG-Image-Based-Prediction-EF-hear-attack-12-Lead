"""5C: ECG-derived feature families (label-free, deterministic, per ECG; development rows only; no fitting -> no leakage).

Input: data/signals.npz (12 leads x 2.5 s at 500 Hz, each lead its own printed window; lead-II rhythm strip 10 s).
Families (git-ignored store phase_5/data/private/features/<family>.csv):
  qrs       per lead from the median beat: R, S, Q amplitudes, QRS area, QRS width estimate, fragmentation (extra slope
            sign changes inside the QRS); + median width, total fragmentation, R-wave progression slope V1->V4
  st        per lead: ST level at J+60 ms vs PR baseline, ST slope J->J+80 ms, T amplitude and T area
  hrv       rhythm strip: mean RR, SDNN, RMSSD, pNN50, CV, min / max RR, beats, ectopy fraction, heart rate
  freq      per lead Welch PSD: relative power 0.5-4 / 4-15 / 15-40 / 40-100 Hz, spectral entropy
  wavelet   per lead db4 5-level DWT: relative energy per level (6) + log total energy
  leadstat  per lead (band-passed): mean, SD, skewness, kurtosis, peak-to-peak, IQR
  sq        signal quality (raw signals): flat-line fraction, clipping fraction, baseline-wander and HF-noise power
            fractions, kurtosis-SQI, amplitude; summarised over leads (mean / min / max), poor-lead count, a composite
            quality score, and the same for the rhythm strip
Validation (results/feature_validation.json, aggregates only): missing / constant rates; agreement of waveform-derived
heart rate, RR and QRS width with the printed machine measurements (correlation, mean difference).
    python -m research_experiments.phase_5.features.extract
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.signal import find_peaks, welch

from research_experiments.common import ROOT
from research_experiments.ecgfounder.inputs import LEADS, bandpass
from research_experiments.phase_5.p5common import FEAT5, IDS, RES5, _H, guard5

FS = 500
MS = lambda ms: int(round(ms * FS / 1000))
FAMILIES = ("qrs", "st", "hrv", "freq", "wavelet", "leadstat", "sq")


def rpeaks(x):
    y = x if x.max() > -x.min() else -x
    pk, _ = find_peaks(y, distance=MS(300), prominence=0.5 * np.percentile(np.abs(y), 99))
    return pk


def template(x, pk, pre=200, post=400):
    segs = [x[p - MS(pre):p + MS(post)] for p in pk if p - MS(pre) >= 0 and p + MS(post) <= len(x)]
    return np.median(np.stack(segs), 0) if segs else None


def beat_features(t):
    """Median-beat measurements (R peak at index c)."""
    c = MS(200); nan = dict.fromkeys(["R", "S", "Q", "area", "width", "frag", "st60", "st_slope", "t_amp", "t_area"], np.nan)
    if t is None:
        return nan
    base = float(np.median(t[c - MS(90):c - MS(50)]))
    v = t - base
    d = np.abs(np.diff(v)); win = slice(c - MS(120), c + MS(120))
    thr = 0.15 * d[win].max() if d[win].max() > 0 else np.inf
    above = np.flatnonzero(d[win] > thr) + (c - MS(120))
    on, off = (int(above[0]), int(above[-1])) if len(above) else (c - MS(40), c + MS(40))
    q = v[max(on - MS(10), 0):c]; s = v[c:min(off + MS(10), len(v))]
    sg = np.sign(np.diff(v[on:off + 1])); sg = sg[sg != 0]
    j = off
    t_win = v[min(j + MS(100), len(v) - 1):min(j + MS(300), len(v))]
    ti = int(np.argmax(np.abs(t_win))) if len(t_win) else 0
    return dict(R=float(v[c - MS(60):c + MS(60)].max()), S=float(-min(s.min(), 0)) if len(s) else np.nan,
                Q=float(-min(q.min(), 0)) if len(q) else np.nan, area=float(np.abs(v[on:off + 1]).sum() / FS * 1000),
                width=float((off - on) / FS * 1000), frag=float(max(int((np.diff(sg) != 0).sum()) - 2, 0)) if len(sg) > 1 else 0.0,
                st60=float(v[min(j + MS(60), len(v) - 1)]), st_slope=float((v[min(j + MS(80), len(v) - 1)] - v[j]) / 0.08),
                t_amp=float(t_win[ti]) if len(t_win) else np.nan, t_area=float(t_win.sum() / FS * 1000) if len(t_win) else np.nan)


def rel_band(x):
    f, p = welch(x - x.mean(), fs=FS, nperseg=min(500, len(x)))
    tot = p[(f >= 0.5) & (f <= 100)].sum() + 1e-12
    bands = [(0.5, 4), (4, 15), (15, 40), (40, 100)]
    rel = [float(p[(f >= a) & (f < b)].sum() / tot) for a, b in bands]
    q = p[(f >= 0.5) & (f <= 100)]; q = q / (q.sum() + 1e-12)
    return rel + [float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q)))]


def wav(x):
    co = pywt.wavedec(x - x.mean(), "db4", level=5)
    e = np.array([float((c ** 2).sum()) for c in co]); tot = e.sum() + 1e-12
    return list(e / tot) + [float(np.log(tot))]


def quality(raw):
    d = np.abs(np.diff(raw))
    f, p = welch(raw - raw.mean(), fs=FS, nperseg=len(raw))      # full length: 0.4 Hz (lead) / 0.1 Hz (rhythm) resolution
    tot = p[f > 0].sum() + 1e-12
    lo, hi = raw.min(), raw.max()
    return dict(flat=float(np.mean(d < 1e-4)), clip=float(np.mean((raw >= hi - 1e-6) | (raw <= lo + 1e-6))),
                bw=float(p[(f > 0) & (f < 0.67)].sum() / tot), hf=float(p[f > 40].sum() / tot),
                ksqi=float(stats.kurtosis(bandpass(raw))), amp=float(hi - lo))


def one(args):
    leads, rhythm = args
    out = {fam: {} for fam in FAMILIES}
    filt = [bandpass(leads[i].astype(np.float64)) for i in range(12)]
    widths, frags, rs = [], [], []
    for i, n in enumerate(LEADS):
        x = filt[i]; bf = beat_features(template(x, rpeaks(x)))
        for k in ("R", "S", "Q", "area", "width", "frag"):
            out["qrs"][f"{k}_{n}"] = bf[k]
        for k in ("st60", "st_slope", "t_amp", "t_area"):
            out["st"][f"{k}_{n}"] = bf[k]
        widths.append(bf["width"]); frags.append(bf["frag"])
        if n in ("V1", "V2", "V3", "V4"):
            rs.append(bf["R"])
        for k, v in zip(("p05_4", "p4_15", "p15_40", "p40_100", "spec_entropy"), rel_band(leads[i].astype(np.float64))):
            out["freq"][f"{k}_{n}"] = v
        for k, v in enumerate(wav(leads[i].astype(np.float64))):
            out["wavelet"][f"{'logE' if k == 6 else f'rel_lvl{k}'}_{n}"] = v
        out["leadstat"].update({f"mean_{n}": float(x.mean()), f"sd_{n}": float(x.std()), f"skew_{n}": float(stats.skew(x)),
                                f"kurt_{n}": float(stats.kurtosis(x)), f"p2p_{n}": float(np.ptp(x)),
                                f"iqr_{n}": float(np.subtract(*np.percentile(x, [75, 25])))})
    out["qrs"].update(width_median=float(np.nanmedian(widths)), frag_total=float(np.nansum(frags)),
                      r_progression_slope=float(np.polyfit(range(len(rs)), rs, 1)[0]) if np.all(np.isfinite(rs)) else np.nan)
    II = bandpass(rhythm.astype(np.float64)); pk = rpeaks(II)
    rr = np.diff(pk) / FS * 1000
    if len(rr) >= 2:
        med = np.median(rr)
        out["hrv"] = dict(mean_rr=float(rr.mean()), sdnn=float(rr.std()), rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan,
                          pnn50=float(np.mean(np.abs(np.diff(rr)) > 50)) if len(rr) > 2 else np.nan, cv=float(rr.std() / rr.mean()),
                          min_rr=float(rr.min()), max_rr=float(rr.max()), n_beats=float(len(pk)),
                          ectopy_frac=float(np.mean(np.abs(rr - med) > 0.2 * med)), hr=float(60000 / rr.mean()))
    else:
        out["hrv"] = dict.fromkeys(["mean_rr", "sdnn", "rmssd", "pnn50", "cv", "min_rr", "max_rr", "n_beats", "ectopy_frac", "hr"], np.nan)
    q = [quality(leads[i].astype(np.float64)) for i in range(12)]
    for k in q[0]:
        v = np.array([x[k] for x in q])
        out["sq"].update({f"{k}_mean": float(v.mean()), f"{k}_min": float(v.min()), f"{k}_max": float(v.max())})
    # v2 (2026-11-05): the flat-line criterion is NOT used for the poor-lead flag - PDF vector digitisation produces
    # piecewise-linear traces, so 'flat' samples are an artefact of digitisation (mean 0.62), not of recording quality.
    poor = [(x["amp"] < 0.15) or (x["ksqi"] < 3) or (x["hf"] > 0.3) or (x["bw"] > 0.5) for x in q]
    out["sq"]["n_poor_leads"] = float(sum(poor))
    qr = quality(rhythm.astype(np.float64))
    out["sq"].update({f"rhythm_{k}": v for k, v in qr.items()})
    out["sq"]["quality_score"] = float(1 - np.mean(poor))      # fraction of acceptable leads (used by E6)
    return out


def main():
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[IDS].values
    leads, rhythm = z["leads"][pos], z["rhythm"][pos]
    with Pool(2) as pool:
        res = pool.map(one, list(zip(leads, rhythm)), chunksize=16)
    guard5(FEAT5).mkdir(parents=True, exist_ok=True)
    val = {"n_ecgs": len(IDS), "families": {}}
    tabs = {}
    for fam in FAMILIES:
        T = pd.DataFrame([r[fam] for r in res], index=pd.Index(IDS, name="ecg_id"))
        T = T.replace([np.inf, -np.inf], np.nan)
        T.to_csv(FEAT5 / f"{fam}.csv"); tabs[fam] = T
        val["families"][fam] = dict(n_features=int(T.shape[1]), missing_share=round(float(T.isna().mean().mean()), 4),
                                    features_with_any_missing=int(T.isna().any().sum()),
                                    constant_features=int((T.nunique() <= 1).sum()))
    H = _H.set_index("ecg_id").loc[IDS]
    def agree(a, b, name):
        m = a.notna() & b.notna()
        return dict(n=int(m.sum()), pearson_r=round(float(np.corrcoef(a[m], b[m])[0, 1]), 3),
                    mean_diff=round(float((a[m] - b[m]).mean()), 2), median_abs_diff=round(float((a[m] - b[m]).abs().median()), 2), measure=name)
    val["consistency_with_machine_measurements"] = {
        "heart_rate_bpm (rhythm strip vs printed Rate)": agree(tabs["hrv"].hr, H.Rate, "bpm"),
        "mean_RR_ms (rhythm strip vs printed RR)": agree(tabs["hrv"].mean_rr, H.RR, "ms"),
        "QRS_width_ms (median-beat estimate vs printed QRSD)": agree(tabs["qrs"].width_median, H.QRSD, "ms")}
    val["version_notes"] = ["sq v2: poor-lead flag = amplitude < 0.15 mV or kurtosis-SQI < 3 or HF-noise power > 30 % or "
                            "baseline-wander power > 50 %; flat-line fraction kept as a descriptive feature only (digitisation "
                            "artefact); quality spectra at full-length resolution. v1 (flat-line criterion, 1-Hz spectra) marked "
                            "98 % of leads poor and was never used by any experiment."]
    val["sq_quality_score"] = {k: round(float(v), 3) for k, v in tabs["sq"].quality_score.describe().items()}
    val["sq_n_poor_leads_share_ecgs_with_any"] = round(float((tabs["sq"].n_poor_leads > 0).mean()), 3)
    (RES5 / "feature_validation.json").write_text(json.dumps(val, indent=1))
    print(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
