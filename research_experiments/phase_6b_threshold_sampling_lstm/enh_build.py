"""ENH feature builder (AMENDMENT_ENH.md P1-P2). Label-free; development ECGs only; image-derived waveform only.
  P1 beat-synchronous 10-s 12-lead reconstruction ("synth") from the 3x4 printout + rhythm strip
  F-syn  ECGFounder (frozen) f6 + 150 logits on the synth input
  F-beat synthesised median beat (12 leads x 100 samples per cycle, mV)
  export of the synth input at 100 Hz for MiniRocket (run by enh_rocket.py in a separate venv)
Outputs (git-ignored): data/private/enh/build.npz; aggregate fidelity: results/enh_build.json
    python -m research_experiments.phase_6b_threshold_sampling_lstm.enh_build
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy.signal import find_peaks

from research_experiments.ecgfounder.inputs import LEADS, bandpass
from research_experiments.phase_2.ecgf_features import load_model, run
from research_experiments.phase_6a_normal_mild_lstm.signals import edge_mask
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

FS, SEG, N10 = 500, 1250, 5000
COL = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
OUTD = B.PRIV / "enh"
XF = 10                                                     # 20-ms cross-fade


def dev_signals():
    z = np.load(B.ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    pos = np.array([idx[e] for e in B.IDS])                 # development rows only
    return z["leads"][pos], z["rhythm"][pos]


def rpeaks(x):
    d = np.diff(x, prepend=x[0]); e = np.convolve(d * d, np.ones(50) / 50, "same")
    pk, _ = find_peaks(e, height=0.3 * np.percentile(e, 99), distance=int(0.3 * FS))
    R = sorted({int(max(0, p - 37) + np.argmax(np.abs(x[max(0, p - 37):p + 38]))) for p in pk})
    out = []
    for r in R:                                             # enforce 0.25-s refractory, keep the larger peak
        if out and r - out[-1] < int(0.25 * FS):
            if abs(x[r]) > abs(x[out[-1]]):
                out[-1] = r
        else:
            out.append(r)
    return np.array(out, int)


def template(s, rec, local_R, M, pre):
    """Median R-to-R cycle (length M, starting `pre` before R) from the recorded samples of one 2.5-s column."""
    cyc = [s[st:st + M] for st in local_R - pre if st >= 0 and st + M <= len(s) and rec[st:st + M].all()]
    if cyc:
        return np.median(np.stack(cyc), 0), True
    best, T = 0, np.zeros(M)                                # partial cycle: the one with the largest recorded overlap
    for st in local_R - pre:
        a, b = max(0, st), min(len(s), st + M)
        cov = int(rec[a:b].sum()) if b > a else 0
        if cov > best:
            best, T = cov, np.zeros(M); T[a - st:b - st] = np.where(rec[a:b], s[a:b], 0)
    return T, False


def warp(T, L, pre):
    M = len(T); k = min(pre + int(0.45 * FS), int(0.85 * L), M)
    tail = T[k:]
    rest = np.full(L - k, T[k - 1]) if len(tail) < 2 else np.interp(np.linspace(0, len(tail) - 1, L - k), np.arange(len(tail)), tail)
    return np.r_[T[:k], rest]


def synth_lead(T, R, pre):
    M = len(T); y = np.zeros(N10)
    st = R - pre
    for j in range(len(R) - 1):
        a, L = st[j], R[j + 1] - R[j]; c = warp(T, L, pre)
        lo, hi = max(0, a), min(N10, a + L)
        if hi > lo:
            y[lo:hi] = c[lo - a:hi - a]
    a0 = st[0]
    if a0 > 0:
        rep = np.tile(T, a0 // M + 2); y[:a0] = rep[len(rep) - a0:]
    a1 = st[-1]
    if a1 < N10:
        rep = np.tile(T, (N10 - a1) // M + 2)
        lo = max(0, a1); y[lo:] = rep[lo - a1:lo - a1 + N10 - lo]
    return y


def build_one(leads, rhythm):
    II = bandpass(rhythm.astype(np.float64))
    R = rpeaks(II)
    ok_R = len(R) >= 3
    if not ok_R:                                            # fallback: one beat per second
        R = np.arange(250, N10, 500)
    M = int(np.median(np.diff(R))); pre = int(min(0.25 * FS, 0.35 * M))
    out = np.zeros((12, N10)); beats = np.zeros((12, 100)); fid = []; full = []
    for i, n in enumerate(LEADS):
        if n == "II":
            out[i] = II
            cyc = [II[s:s + M] for s in R - pre if s >= 0 and s + M <= N10]
            T = np.median(np.stack(cyc), 0) if cyc else II[:M]
        else:
            raw = leads[i].astype(np.float64); o = COL[n] * SEG
            flat = (np.abs(np.diff(raw)) < 1e-9).mean() >= 0.90
            rec = np.zeros(SEG, bool) if flat else edge_mask(raw)
            s = bandpass(raw) if not flat else np.zeros(SEG)
            lr = R[(R >= o - M) & (R < o + SEG + M)] - o
            T, f = template(s, rec, lr, M, pre); full.append(f)
            y = synth_lead(T, R, pre)
            if rec.sum() > 50:
                syn = y[o:o + SEG][rec]; obs = s[rec]
                if syn.std() > 1e-6 and obs.std() > 1e-6:
                    fid.append(float(np.corrcoef(syn, obs)[0, 1]))
                idx = np.flatnonzero(rec); a, b = idx[0], idx[-1]           # recorded span (edge-masked)
                w = rec.astype(float); ramp = (np.arange(XF) + 1) / (XF + 1); n = min(XF, b - a + 1)
                w[a:a + n] = np.minimum(w[a:a + n], ramp[:n])                 # 20-ms cross-fade at both edges
                w[b - n + 1:b + 1] = np.minimum(w[b - n + 1:b + 1], ramp[:n][::-1])
                y[o:o + SEG] = w * s + (1 - w) * y[o:o + SEG]
            out[i] = y
        beats[i] = np.interp(np.linspace(0, len(T) - 1, 100), np.arange(len(T)), T)
    return out, beats, dict(ok_R=ok_R, fid=fid, full=full, hr=60 * FS / M)


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    L, Rh = dev_signals()
    S, BT, info = [], [], []
    for l, r in zip(L, Rh):
        y, b, inf = build_one(l, r); S.append(y.astype(np.float32)); BT.append(b.astype(np.float32)); info.append(inf)
    S = np.stack(S); BT = np.stack(BT)
    Z = (S - S.reshape(len(S), -1).mean(1)[:, None, None]) / (S.reshape(len(S), -1).std(1)[:, None, None] + 1e-8)
    tb = time.time() - t0; t0 = time.time()
    m = load_model(False)
    o = run(m, Z.astype(np.float32))
    tf = time.time() - t0
    rocket_in = np.clip(S, -5, 5)[:, :, ::5].astype(np.float32)          # 100 Hz (band-passed below 40 Hz)
    folds = np.stack([B.fold_vec(r) for r in (0, 1, 2)])
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(B.guard(OUTD / "build.npz"), ecg_id=np.asarray(B.IDS, dtype="U32"), f6=o["f6"], logits=o["logits"], beats=BT)
    np.savez(B.guard(OUTD / "rocket_in.npz"), X=rocket_in, folds=folds)
    fid = np.concatenate([x["fid"] for x in info]); full = np.concatenate([x["full"] for x in info])
    agg = dict(n_ecgs=int(len(S)), rpeak_detection_ok=float(np.mean([x["ok_R"] for x in info])),
               median_heart_rate=float(np.median([x["hr"] for x in info])),
               template_vs_recorded_corr_median=float(np.median(fid)), template_vs_recorded_corr_iqr=[float(np.percentile(fid, 25)), float(np.percentile(fid, 75))],
               share_leads_with_complete_cycle=float(full.mean()), build_seconds=round(tb, 1), ecgfounder_seconds=round(tf, 1))
    (B.guard(B.RES) / "enh_build.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
