"""Image-only clinician ECG features for LVEF class (AMENDMENT_MILDFEAT.md). Label-free, deterministic, per ECG.
No header text, no age, no sex: only the ECG drawing digitised from the printed PDF (data/signals.npz: 12 leads x 2.5 s,
each lead in its own printed column, + the 10-s lead II rhythm strip; 500 Hz; mV). Development ECGs only.

Reading the ECG like a cardiologist, with one technical step the earlier feature families (Phase 3 clin3, Phase 5 5C)
did not use: the rhythm strip is simultaneous with all four printed columns (verified r = 0.9998, ecgfounder/inputs.py),
so its R peaks time-stamp the beats inside every column. Median beats of all 12 leads are aligned to the same rhythm-II
R peak -> a synthesised simultaneous median 12-lead beat (stationary-morphology assumption; non-dominant beats removed).
That allows GLOBAL fiducials (QRS onset / offset from the 12-lead spatial velocity, T end, P wave) and VECTOR measures
(Kors-transform VCG: spatial QRS-T angle, ventricular gradient, frontal axes).

Feature groups (all from the drawing):
  rhythm        heart rate, RR variability / irregularity, premature and non-dominant beat fractions
  intervals     QRS duration, PR, P duration, QT, QTc (Bazett, Fridericia), JT, Tpeak-Tend
  conduction    R-peak time (intrinsicoid deflection) V1 / V5 / V6, LBBB-like / RBBB-like / IVCD patterns
  scar          pathological Q waves and fragmented QRS per territory (inferior, lateral, anterior), poor R-wave
                progression, R-wave amplitudes V1-V4, transition zone
  voltage       Sokolow-Lyon, Cornell voltage, Peguero-Lo Presti, 12-lead QRS voltage sum, low limb / precordial voltage
  repolarisation ST at J+60 and T amplitude per territory, T inversion counts, flat T, lateral strain, T/R ratio V5 / V6
  atrial        P amplitude (II), P-terminal force in V1, P axis
  vector        frontal QRS / T axes and QRS-T angle, spatial QRS-T angle (mean and peak vectors), ventricular gradient,
                QRS and T vector magnitudes
  quality       beats per column, valid leads

Measurement calibration (label-free): the QRS onset / offset threshold is chosen from {0.05, 0.10, 0.15, 0.20} of the
peak spatial velocity by agreement with the printed machine QRS duration. The printed machine measurements are used ONLY
to validate / calibrate the measurements here, never as model input; no label or echo value is used.
Outputs: data/private/mildfeat.csv (per ECG, git-ignored); results/mildfeat_validation.json (aggregates only).
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildfeat
"""
from __future__ import annotations

import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, iirnotch, sosfiltfilt, filtfilt

from research_experiments.common import ROOT
from research_experiments.ecgfounder.inputs import LEADS
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

FS = 500
ms = lambda v: int(round(v * FS / 1000))                    # noqa: E731
PRE, POST = ms(420), ms(520)                               # template window around the rhythm-II R peak
R0 = PRE                                                   # R-peak index inside a template
SEG = 1250
COLS = {0: ("I", "II", "III"), 1: ("aVR", "aVL", "aVF"), 2: ("V1", "V2", "V3"), 3: ("V4", "V5", "V6")}
LI = {n: i for i, n in enumerate(LEADS)}
FRACS = (0.05, 0.10, 0.15, 0.20)
TEMODES = ("median", "p75", "rhythm")         # global T end: median / 75th percentile of lead T ends, or the rhythm-II beat
TEMODE = "median"
PFRAC = 0.15                                  # P onset / offset at this fraction of the P peak amplitude
OUT = B.PRIV / "mildfeat.csv"
_NB, _NA = iirnotch(50, 30, FS)
_SOS = butter(2, [0.5, 40], btype="bandpass", fs=FS, output="sos")
# Kors regression matrix (Kors et al. 1990): rows X, Y, Z; columns I, II, V1..V6
KORS = np.array([[0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
                 [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
                 [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31]])
KORS_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
TERR = dict(inferior=("II", "III", "aVF"), lateral=("I", "aVL", "V5", "V6"), anterior=("V1", "V2", "V3", "V4"))


def filt(x):
    x = filtfilt(_NB, _NA, x.astype(np.float64))
    return sosfiltfilt(_SOS, x)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


# ------------------------------------------------------------------ beats on the rhythm strip
def rpeaks(x):
    """QRS detection on the filtered rhythm strip: envelope of the squared slope, then the largest |deflection| near it."""
    e = smooth(np.gradient(x) ** 2, ms(80))
    pk, _ = find_peaks(e, distance=ms(300), height=0.25 * np.percentile(e, 99))
    out = []
    for p in pk:
        a, b = max(p - ms(60), 0), min(p + ms(60), len(x))
        out.append(a + int(np.argmax(np.abs(x[a:b] - np.median(x[a:b])))))
    return np.unique(np.array(out, int))


def beat_matrix(x, peaks, offset=0):
    """Beats of segment x (time origin `offset` on the rhythm axis) aligned on rhythm R peaks; NaN outside the segment."""
    M = np.full((len(peaks), PRE + POST), np.nan)
    for i, p in enumerate(peaks):
        a = p - offset - PRE
        lo, hi = max(a, 0), min(a + PRE + POST, len(x))
        if hi - lo > 0:
            M[i, lo - a:hi - a] = x[lo:hi]
    return M


def dominant(rh, peaks):
    """Beats whose rhythm-II QRS (R +- 100 ms) correlates >= 0.9 with the median beat and that are not premature."""
    M = beat_matrix(rh, peaks)
    w = slice(R0 - ms(100), R0 + ms(100))
    tpl = np.nanmedian(M[:, w], 0)
    ok = np.array([np.isfinite(m[w]).all() and np.corrcoef(m[w], tpl)[0, 1] >= 0.9 for m in M])
    rr = np.diff(peaks) / FS * 1000
    prem = np.r_[False, rr < 0.85 * np.median(rr)] if len(rr) else np.zeros(len(peaks), bool)
    return ok & ~prem, ok, prem


# ------------------------------------------------------------------ templates and global fiducials
def templates(leads, rh):
    rhf = filt(rh); peaks = rpeaks(rhf)
    if len(peaks) < 3:
        return None
    dom, ok, prem = dominant(rhf, peaks)
    if dom.sum() < 2:
        dom = ok if ok.sum() >= 2 else np.ones(len(peaks), bool)
    T = {}
    nb = {}
    for c, names in COLS.items():
        pk = peaks[dom & (peaks >= c * SEG - POST) & (peaks < (c + 1) * SEG + PRE)]
        for n in names:
            seg = filt(leads[LI[n]])
            Mx = beat_matrix(seg, pk, offset=c * SEG)
            cover = np.isfinite(Mx).sum(0)
            T[n] = np.where(cover >= 1, np.nanmedian(np.where(np.isfinite(Mx), Mx, np.nan), 0), np.nan) if len(pk) else \
                np.full(PRE + POST, np.nan)
            nb[c] = int(((pk - c * SEG >= PRE) & (pk - c * SEG + POST <= SEG)).sum())
    T["rhythm"] = np.nanmedian(beat_matrix(rhf, peaks[dom]), 0)
    return dict(T=T, peaks=peaks, dom=dom, ok=ok, prem=prem, beats_per_column=nb)


def qrs_bounds(T, frac):
    """Global QRS onset / offset from the spatial velocity of the aligned 12-lead median beat."""
    D = np.stack([np.gradient(T[n]) for n in LEADS if np.isfinite(T[n][R0 - ms(160):R0 + ms(220)]).all()])
    if len(D) < 6:
        return np.nan, np.nan
    v = smooth(np.sqrt(np.nansum(D ** 2, 0)), ms(8))
    vmax = np.nanmax(v[R0 - ms(60):R0 + ms(60)]); thr = frac * vmax
    on = np.nan
    run = 0
    for t in range(R0, R0 - ms(160), -1):
        run = run + 1 if v[t] < thr else 0
        if run >= ms(10):
            on = t + run - 1; break
    off = np.nan
    run = 0
    for t in range(R0, R0 + ms(220)):
        run = run + 1 if v[t] < thr else 0
        if run >= ms(16):
            off = t - run + 1; break
    return on, off


def base_of(x, on):
    w = x[int(on) - ms(30):int(on) - ms(8)]
    return float(np.nanmedian(w)) if np.isfinite(w).any() else 0.0


def t_end_lead(y, off, base, rr):
    """Tangent T end of one lead (baseline-corrected y); returns (t_peak, t_end) or NaNs if |T| < 0.1 mV.
    The T-peak search ends before the next beat's P wave (R + min(480 ms, 0.7 RR))."""
    a, b = int(off) + ms(80), min(R0 + min(ms(480), int(0.7 * rr)), len(y) - 2)
    if b - a < ms(40) or not np.isfinite(y[a:b]).all():
        return np.nan, np.nan, np.nan
    seg = y[a:b] - base
    tp = a + int(np.argmax(np.abs(seg))); amp = y[tp] - base
    if abs(amp) < 0.1:
        return np.nan, np.nan, amp
    e = min(tp + ms(160), len(y) - 2)
    d = np.gradient(y[tp:e])
    if not np.isfinite(d).all() or len(d) < 3:
        return np.nan, np.nan, amp
    k = int(np.argmin(d)) if amp > 0 else int(np.argmax(d))
    if d[k] == 0:
        return np.nan, np.nan, amp
    ts = tp + k
    te = ts + (base - y[ts]) / d[k]
    return float(tp), float(min(max(te, ts), len(y) - 1)), amp


# ------------------------------------------------------------------ per-lead QRS description
def qrs_lead(q, dur_ms):
    """q: baseline-corrected QRS samples (onset..offset). Returns R, S, Q, Q duration, R-peak time, net area, extrema."""
    out = dict(R=np.nan, S=np.nan, Q=np.nan, Qdur=np.nan, RPT=np.nan, area=np.nan, p2p=np.nan, ext=np.nan, term=np.nan)
    if len(q) < 4 or not np.isfinite(q).all():
        return out
    qs = smooth(q, 3)
    ir = int(np.argmax(qs)); R = max(float(qs[ir]), 0.0)
    S = max(float(-qs[ir:].min()), 0.0) if R > 0.05 else max(float(-qs.min()), 0.0)
    sig = np.flatnonzero(np.abs(qs) > 0.05)
    Q, Qdur = 0.0, 0.0
    if len(sig) and qs[sig[0]] < 0:                        # initial deflection negative
        pos = np.flatnonzero(qs[sig[0]:] >= 0)
        end = sig[0] + (pos[0] if len(pos) else len(qs) - sig[0])
        Q = float(-qs[:end].min()); Qdur = end / FS * 1000
    d = np.diff(qs)
    sc = np.flatnonzero(np.sign(d[1:]) != np.sign(d[:-1])) + 1
    ext = [0] + [i for i in sc] + [len(qs) - 1]
    keep = [ext[0]]
    for i in ext[1:]:
        if abs(qs[i] - qs[keep[-1]]) >= 0.03:
            keep.append(i)
    n_ext = sum(1 for i in keep[1:-1])
    nz = qs[np.abs(qs) > 0.05]
    out.update(R=R, S=S, Q=Q, Qdur=Qdur, RPT=(ir / FS * 1000) if R > 0.05 else np.nan, area=float(q.sum() / FS * 1000),
               p2p=float(qs.max() - qs.min()), ext=float(n_ext), term=float(np.sign(nz[-1])) if len(nz) else 0.0)
    return out


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def fold180(d):
    d = abs((d + 180) % 360 - 180)
    return d


# ------------------------------------------------------------------ one ECG
def features(leads, rh, frac):
    tp = templates(leads, rh)
    if tp is None:
        return {}
    T = tp["T"]; F = {}
    peaks, dom = tp["peaks"], tp["dom"]
    rr = np.diff(peaks) / FS * 1000
    F["hr"] = 60000 / np.median(rr); F["rr_cv"] = rr.std() / rr.mean()
    F["rr_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else np.nan
    F["rr_irregularity"] = F["rr_rmssd"] / rr.mean()
    F["premature_frac"] = float(tp["prem"].mean()); F["nondominant_frac"] = float((~tp["ok"]).mean())
    F["n_beats"] = len(peaks); F["min_beats_per_column"] = min(tp["beats_per_column"].values())
    on, off = qrs_bounds(T, frac)
    F["valid_leads"] = int(sum(np.isfinite(T[n][R0 - ms(160):R0 + ms(220)]).all() for n in LEADS))
    if not (np.isfinite(on) and np.isfinite(off)):
        return F
    on, off = int(on), int(off)
    qrsd = (off - on) / FS * 1000; F["qrs_dur"] = qrsd
    base = {n: base_of(T[n], on) for n in LEADS}
    Y = {n: T[n] - base[n] for n in LEADS}
    Ld = {n: qrs_lead(Y[n][on:off + 1], qrsd) for n in LEADS}
    # ---- T wave / QT
    tps, tes, tamp = {}, {}, {}
    rrs = int(np.median(np.diff(peaks)))
    for n in LEADS:
        a, e, amp = t_end_lead(T[n], off, base[n], rrs); tps[n], tes[n], tamp[n] = a, e, amp
    vals = np.array([tes[n] for n in LEADS if np.isfinite(tes[n])])
    if TEMODE == "rhythm":
        te = t_end_lead(T["rhythm"], off, base_of(T["rhythm"], on), rrs)[1]
    elif len(vals):
        te = float(np.median(vals) if TEMODE == "median" else np.percentile(vals, 75))
    else:
        te = np.nan
    rr_s = np.median(rr) / 1000
    if np.isfinite(te):
        qt = (te - on) / FS * 1000
        F.update(qt=qt, qtc_bazett=qt / np.sqrt(rr_s), qtc_fridericia=qt / rr_s ** (1 / 3), jt=qt - qrsd,
                 jtc=qt / np.sqrt(rr_s) - qrsd)
        tpe = [(tes[n] - tps[n]) / FS * 1000 for n in ("V2", "V3", "V4", "V5", "V6") if np.isfinite(tes[n])]
        F["tpeak_tend"] = float(np.median(tpe)) if tpe else np.nan
        tend = int(round(te))
    else:
        tend = min(off + ms(320), PRE + POST - 1)
    # ---- per-lead summaries
    for n in LEADS:
        L = Ld[n]
        j60 = off + ms(60)
        F[f"st60_{n}"] = float(Y[n][j60]) if j60 < len(Y[n]) and np.isfinite(Y[n][j60]) else np.nan
        F[f"t_amp_{n}"] = float(tamp[n]) if np.isfinite(tamp[n]) else np.nan
        F[f"qrs_area_{n}"] = L["area"]
        F[f"t_area_{n}"] = float(np.nansum(Y[n][off:tend]) / FS * 1000) if np.isfinite(Y[n][off:tend]).any() else np.nan
    # ---- conduction
    for n in ("V1", "V5", "V6"):
        F[f"rpt_{n}"] = Ld[n]["RPT"]
    noq_lat = all((Ld[n]["Qdur"] or 0) == 0 for n in ("I", "V5", "V6"))
    lbbb = qrsd >= 120 and noq_lat and (Ld["V1"]["S"] > Ld["V1"]["R"]) and (Ld["V6"]["RPT"] or 0) >= 60
    rbbb = qrsd >= 120 and Ld["V1"]["term"] > 0 and Ld["I"]["term"] < 0
    F.update(lbbb_like=float(lbbb), rbbb_like=float(rbbb), ivcd=float(qrsd >= 110 and not lbbb and not rbbb))
    # ---- scar: Q waves, fragmentation, R progression
    def path_q(n):
        L = Ld[n]
        if not np.isfinite(L["Qdur"]):
            return np.nan
        lim = 20 if n in ("V2", "V3") else 40
        qs = L["R"] < 0.05 and L["S"] > 0.1
        return float(L["Qdur"] >= lim or (L["Q"] >= 0.1 and L["Q"] >= 0.25 * max(L["R"], 1e-6)) or (qs and n in ("V2", "V3", "V4")))
    def frag(n):
        L = Ld[n]
        return np.nan if not np.isfinite(L["ext"]) else float(L["ext"] >= (4 if qrsd >= 120 else 3))
    for t, names in TERR.items():
        pq = [path_q(n) for n in names if n not in ("V1",)]
        F[f"path_q_{t}"] = float(np.nansum(pq)); F[f"frag_{t}"] = float(np.nansum([frag(n) for n in names]))
    F["frag_total"] = float(np.nansum([frag(n) for n in LEADS if n != "aVR"]))
    F["notches_total"] = float(np.nansum([Ld[n]["ext"] for n in LEADS if n != "aVR"]))
    for n in ("V1", "V2", "V3", "V4"):
        F[f"r_{n}"] = Ld[n]["R"]
    F["prwp"] = float(Ld["V3"]["R"] <= 0.3) if np.isfinite(Ld["V3"]["R"]) else np.nan
    prec = ["V1", "V2", "V3", "V4", "V5", "V6"]
    tz = [i + 1 for i, n in enumerate(prec) if np.isfinite(Ld[n]["R"]) and Ld[n]["R"] >= Ld[n]["S"]]
    F["transition"] = float(tz[0]) if tz else 7.0
    # ---- voltage
    F["sokolow_lyon"] = Ld["V1"]["S"] + np.nanmax([Ld["V5"]["R"], Ld["V6"]["R"]])
    F["cornell"] = Ld["aVL"]["R"] + Ld["V3"]["S"]
    F["peguero"] = np.nanmax([Ld[n]["S"] for n in LEADS]) + Ld["V4"]["S"]
    F["qrs_voltage_sum"] = float(np.nansum([Ld[n]["p2p"] for n in LEADS]))
    limb = ["I", "II", "III", "aVR", "aVL", "aVF"]
    F["limb_max_p2p"] = float(np.nanmax([Ld[n]["p2p"] for n in limb])); F["prec_max_p2p"] = float(np.nanmax([Ld[n]["p2p"] for n in prec]))
    F["low_voltage_limb"] = float(F["limb_max_p2p"] < 0.5); F["low_voltage_prec"] = float(F["prec_max_p2p"] < 1.0)
    # ---- repolarisation
    tinv = ["I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]
    F["t_inversion_count"] = float(sum(1 for n in tinv if np.isfinite(F[f"t_amp_{n}"]) and F[f"t_amp_{n}"] <= -0.1))
    F["t_inversion_lateral"] = float(sum(1 for n in TERR["lateral"] if np.isfinite(F[f"t_amp_{n}"]) and F[f"t_amp_{n}"] <= -0.1))
    F["t_flat_count"] = float(sum(1 for n in ("I", "II", "V3", "V4", "V5", "V6") if not np.isfinite(F[f"t_amp_{n}"])
                                  or abs(F[f"t_amp_{n}"]) < 0.1))
    for t, names in TERR.items():
        F[f"st_min_{t}"] = float(np.nanmin([F[f"st60_{n}"] for n in names])) if np.isfinite([F[f"st60_{n}"] for n in names]).any() else np.nan
        F[f"t_mean_{t}"] = float(np.nanmean([F[f"t_amp_{n}"] for n in names])) if np.isfinite([F[f"t_amp_{n}"] for n in names]).any() else np.nan
    st56 = np.nanmin([F["st60_V5"], F["st60_V6"]]); t56 = np.nanmin([F["t_amp_V5"], F["t_amp_V6"]])
    F["lateral_strain"] = float(st56 <= -0.05 and t56 <= -0.1) if np.isfinite(st56) and np.isfinite(t56) else np.nan
    for n in ("V5", "V6"):
        F[f"t_r_ratio_{n}"] = F[f"t_amp_{n}"] / Ld[n]["R"] if np.isfinite(F[f"t_amp_{n}"]) and (Ld[n]["R"] or 0) > 0.1 else np.nan
    # ---- atrial (rhythm-II median beat: most beats; V1 aligned on the same R peak)
    rh_t = T["rhythm"] - base_of(T["rhythm"], on)
    prev_t = int(round(te)) - rrs + ms(20) if np.isfinite(te) else 0     # previous beat's T end on the template axis
    p0 = max(on - ms(250), prev_t, 0)
    pw = rh_t[p0:on - ms(30)]
    if len(pw) > ms(60) and np.isfinite(pw).all():
        ip = int(np.argmax(pw)) if pw.max() >= 0.05 else int(np.argmax(np.abs(pw)))
        pa = float(pw[ip]); F["p_amp_II"] = pa
        F["p_present"] = float(abs(pa) >= 0.05)
        if abs(pa) >= 0.05:
            thr = PFRAC * abs(pa); a = ip
            while a > 0 and abs(pw[a]) > thr:
                a -= 1
            b = ip
            while b < len(pw) - 1 and abs(pw[b]) > thr:
                b += 1
            p_on, p_off = p0 + a, p0 + b
            F["p_dur"] = (b - a) / FS * 1000; F["pr"] = (on - p_on) / FS * 1000
            v1 = Y["V1"][p0 + ip:p_off + ms(20)]
            if np.isfinite(v1).all() and len(v1):
                neg = v1 < -0.02
                F["ptf_v1"] = float(neg.sum() / FS * 1000 * max(-v1.min(), 0)) if neg.any() else 0.0
            pI, pF = np.nansum(Y["I"][p_on:p_off]), np.nansum(Y["aVF"][p_on:p_off])
            F["p_axis"] = float(np.degrees(np.arctan2(pF, pI)))
    # ---- vector measures
    F["qrs_axis"] = float(np.degrees(np.arctan2(Ld["aVF"]["area"], Ld["I"]["area"])))
    if np.isfinite(F.get("t_area_I", np.nan)) and np.isfinite(F.get("t_area_aVF", np.nan)):
        F["t_axis"] = float(np.degrees(np.arctan2(F["t_area_aVF"], F["t_area_I"])))
        F["frontal_qrs_t"] = fold180(F["qrs_axis"] - F["t_axis"])
    Kx = np.stack([Y[n] for n in KORS_LEADS])
    if np.isfinite(Kx[:, on:tend]).all():
        V = KORS @ Kx
        qv, tv = V[:, on:off + 1].sum(1) / FS * 1000, V[:, off:tend].sum(1) / FS * 1000
        mq = np.linalg.norm(V[:, on:off + 1], axis=0); mt = np.linalg.norm(V[:, off:tend], axis=0)
        F.update(spatial_qrs_t=angle(qv, tv),
                 spatial_peak_qrs_t=angle(V[:, on + int(np.argmax(mq))], V[:, off + int(np.argmax(mt))]),
                 vg_mag=float(np.linalg.norm(qv + tv)), qrs_vec_max=float(mq.max()), t_vec_max=float(mt.max()),
                 qrs_area_mag=float(np.linalg.norm(qv)), t_area_mag=float(np.linalg.norm(tv)),
                 t_qrs_ratio=float(mt.max() / mq.max()) if mq.max() > 0 else np.nan)
    return F


_SIG = None


def _load():
    global _SIG
    if _SIG is None:
        z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
        pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[B.IDS].values   # development rows only
        _SIG = (z["leads"][pos], z["rhythm"][pos])
    return _SIG


def _one(args):
    global TEMODE, PFRAC
    i, frac, TEMODE, PFRAC = args
    L, Rh = _load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return features(L[i], Rh[i], frac)
        except Exception as e:                                        # noqa: BLE001  (recorded, counted in validation)
            return {"_error": type(e).__name__}


def extract(frac, temode="median", pfrac=0.15, jobs=8, n=None):
    with Pool(jobs) as p:
        rows = p.map(_one, [(i, frac, temode, pfrac) for i in range(B.nD if n is None else n)], chunksize=8)
    return pd.DataFrame(rows)


def machine():
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[B.IDS]
    return H[["Rate", "PR", "QRSD", "QT", "QTcB", "P_axis", "QRS_axis", "T_axis"]]


def agree(a, b, circular=False):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    d = a[m] - b[m]
    if circular:
        d = (d + 180) % 360 - 180
        return dict(n=int(m.sum()), median_abs_diff_deg=float(np.median(np.abs(d))),
                    within_30deg=float(np.mean(np.abs(d) <= 30)))
    return dict(n=int(m.sum()), pearson_r=float(np.corrcoef(a[m], b[m])[0, 1]), mean_diff=float(d.mean()),
                median_abs_diff=float(np.median(np.abs(d))))


PFRACS = (0.15, 0.30, 0.50)


def _r(D, col, ref):
    a = agree(D[col].values.astype(float), ref)
    ok = a.get("n", 0) >= 0.9 * B.nD
    return (a.get("pearson_r", -1) if ok else -1), a


def main():
    """Label-free calibration, one setting at a time, each by Pearson r with the printed machine value (>= 90 % of ECGs
    measured): QRS threshold (vs QRSD), then T-end mode (vs QT), then P threshold (vs PR)."""
    H = machine(); cal = dict(qrs={}, t_end={}, p={})
    for f in FRACS:
        sc, a = _r(extract(f, "median", 0.30), "qrs_dur", H.QRSD.values); cal["qrs"][str(f)] = a
        print("QRS frac", f, a, flush=True)
    frac = max(FRACS, key=lambda f: (cal["qrs"][str(f)].get("pearson_r", -1) if cal["qrs"][str(f)].get("n", 0) >= 0.9 * B.nD else -1))
    for t in TEMODES:
        sc, a = _r(extract(frac, t, 0.30), "qt", H.QT.values); cal["t_end"][t] = a
        print("T end", t, a, flush=True)
    temode = max(TEMODES, key=lambda t: (cal["t_end"][t].get("pearson_r", -1) if cal["t_end"][t].get("n", 0) >= 0.9 * B.nD else -1))
    for pf in PFRACS:
        sc, a = _r(extract(frac, temode, pf), "pr", H.PR.values); cal["p"][str(pf)] = a
        print("P frac", pf, a, flush=True)
    pfrac = max(PFRACS, key=lambda p: (cal["p"][str(p)].get("pearson_r", -1) if cal["p"][str(p)].get("n", 0) >= 0.9 * B.nD else -1))
    D = extract(frac, temode, pfrac)
    D.insert(0, "ecg_id", B.IDS)
    B.guard(B.PRIV).mkdir(parents=True, exist_ok=True)
    D.to_csv(B.guard(OUT), index=False, float_format="%.5g")
    feats = [c for c in D.columns if c not in ("ecg_id", "_error")]
    val = dict(n_ecgs=B.nD, chosen=dict(qrs_threshold=frac, t_end_mode=temode, p_threshold=pfrac), calibration=cal,
               n_features=len(feats), errors=int(D["_error"].notna().sum()) if "_error" in D else 0,
               missing_rate={c: round(float(D[c].isna().mean()), 4) for c in feats},
               constant=[c for c in feats if D[c].nunique(dropna=True) <= 1],
               prevalence_binary={c: round(float(D[c].mean()), 4) for c in feats if set(D[c].dropna().unique()) <= {0.0, 1.0}},
               agreement_with_printed_machine_measurements=dict(
                   heart_rate=agree(D.hr.values, H.Rate.values), pr=agree(D.pr.values, H.PR.values),
                   qrs_duration=agree(D.qrs_dur.values, H.QRSD.values), qt=agree(D.qt.values, H.QT.values),
                   qtc_bazett=agree(D.qtc_bazett.values, H.QTcB.values),
                   p_axis=agree(D.p_axis.values, H.P_axis.values, True),
                   qrs_axis=agree(D.qrs_axis.values, H.QRS_axis.values, True),
                   t_axis=agree(D.t_axis.values, H.T_axis.values, True)))
    (B.guard(B.RES) / "mildfeat_validation.json").write_text(json.dumps(val, indent=1))
    print(json.dumps({k: v for k, v in val.items() if k not in ("missing_rate", "calibration")}, indent=1))


if __name__ == "__main__":
    main()
