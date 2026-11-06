"""Restore the git-ignored, label-free feature inputs that Phase 6 needs - development ECGs ONLY.

  * Phase-5 families (qrs, st, hrv, freq, wavelet, leadstat, sq v2): computed with phase_5.features.extract.one (imported,
    not re-implemented). The validation summary is recomputed exactly as extract.main does and must equal the
    committed phase_5/results/feature_validation.json (otherwise the script stops).
  * 'clin3' (Phase-3 clinical features, 28 columns): the per-ECG feature code of phase_3/clinical_features.py is COPIED
    below because importing that module would import phase_3.p3common, which loads the locked test. Only development
    rows are computed (the Phase-3 script computed all 1,047; no locked-test PDF or waveform is touched here).
    Correctness is verified downstream: the regenerated header+clinical LR baselines must reproduce the committed
    Phase-3 / Phase-5 predictions (phase_6.baselines.regenerate).
Writes phase_6/data/private/features/<family>.csv (git-ignored). With --mirror-phase5 it also copies the verified files
into the git-ignored phase_5/data/private/features/ (needed only so that Phase-5 modules / tests can be imported; no
committed Phase-5 file is touched).
    python -m research_experiments.phase_6.features.restore_inputs --mirror-phase5
"""
from __future__ import annotations

import json
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from research_experiments.common import ROOT
from research_experiments.ecgfounder.inputs import LEADS, bandpass
from research_experiments.phase_5.features.extract import FAMILIES, one
from research_experiments.phase_5.p5common import IDS, _H
from research_experiments.phase_6.p6common import FEAT6, RES6, guard6

FS = 500
FEAT5 = ROOT / "research_experiments" / "phase_5" / "data" / "private" / "features"


# ---------------- copied verbatim from research_experiments/phase_3/clinical_features.py (no import: p3common) -------
def rpeaks(x, min_dist=0.3):
    y = x if x.max() > -x.min() else -x
    pk, _ = find_peaks(y, distance=int(min_dist * FS), prominence=0.5 * np.percentile(np.abs(y), 99))
    return pk


def t_amp(x, pk):
    vals = [np.median(x[p + int(0.2 * FS):p + int(0.36 * FS)]) for p in pk if p + int(0.36 * FS) < len(x)]
    return float(np.median(vals)) if vals else np.nan


def clin_features(leads, rhythm):
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
# ---------------------------------------------------------------------------------------------------------------------


def dev_signals():
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[IDS].values   # development rows only
    return z["leads"][pos], z["rhythm"][pos]


def clin3():
    leads, rhythm = dev_signals()
    C = pd.DataFrame([clin_features(a, b) for a, b in zip(leads, rhythm)]); C.insert(0, "ecg_id", IDS)
    H = _H.set_index("ecg_id").loc[IDS]
    d = np.abs(H.QRS_axis.values - H.T_axis.values) % 360
    C["qrs_t_angle"] = np.where(d > 180, 360 - d, d)
    C["jt"] = H.QT.values - H.QRSD.values; C["jtc"] = H.QTcB.values - H.QRSD.values
    C["qrsd_ge120"] = (H.QRSD.values >= 120).astype(float)
    # Phase-3 CSV precision (float_format %.5f), then the staging step of phase_5.baseline.import_existing
    C = pd.read_csv(pd.io.common.StringIO(C.to_csv(index=False, float_format="%.5f")))
    C = C.set_index("ecg_id").loc[IDS].replace([np.inf, -np.inf], np.nan)
    return C


def phase5_families():
    leads, rhythm = dev_signals()
    with Pool(4) as pool:
        res = pool.map(one, list(zip(leads, rhythm)), chunksize=16)
    val = {"n_ecgs": len(IDS), "families": {}}
    tabs = {}
    for fam in FAMILIES:
        T = pd.DataFrame([r[fam] for r in res], index=pd.Index(IDS, name="ecg_id")).replace([np.inf, -np.inf], np.nan)
        tabs[fam] = T
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
    val["sq_quality_score"] = {k: round(float(v), 3) for k, v in tabs["sq"].quality_score.describe().items()}
    val["sq_n_poor_leads_share_ecgs_with_any"] = round(float((tabs["sq"].n_poor_leads > 0).mean()), 3)
    return tabs, val


def main():
    guard6(FEAT6).mkdir(parents=True, exist_ok=True)
    tabs, val = phase5_families()
    ref = json.loads((ROOT / "research_experiments" / "phase_5" / "results" / "feature_validation.json").read_text())
    ref.pop("version_notes", None)
    same = json.dumps(val, sort_keys=True) == json.dumps(ref, sort_keys=True)
    print("phase-5 feature validation summary identical to committed:", same)
    if not same:
        print(json.dumps(val, indent=1)); sys.exit("STOP: regenerated Phase-5 features differ from the committed validation")
    for fam, T in tabs.items():
        T.to_csv(FEAT6 / f"{fam}.csv")
    C = clin3()
    C.to_csv(FEAT6 / "clin3.csv")
    rep = dict(families=list(tabs) + ["clin3"], n_ecgs=len(IDS), phase5_validation_identical=same,
               clin3=dict(n_features=int(C.shape[1]), missing_share=round(float(C.isna().mean().mean()), 4)))
    (guard6(RES6) / "restore_inputs.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if "--mirror-phase5" in sys.argv:
        FEAT5.mkdir(parents=True, exist_ok=True)
        assert "private" in FEAT5.parts
        for f in list(tabs) + ["clin3"]:
            if not (FEAT5 / f"{f}.csv").exists():
                shutil.copyfile(FEAT6 / f"{f}.csv", FEAT5 / f"{f}.csv")
        print("mirrored to git-ignored", FEAT5.relative_to(ROOT))


if __name__ == "__main__":
    main()
