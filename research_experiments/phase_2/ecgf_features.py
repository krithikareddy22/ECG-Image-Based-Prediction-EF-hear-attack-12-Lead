"""Phase-2 ECGFounder representations for all 1,047 ECGs (label-free; no fitting).

Input constructions (audited in audit_waveform.py; all built in memory):
  aligned        each printed lead at its true time position (3x4 layout = successive 2.5-s windows of
                 ONE simultaneous 10-s acquisition), lead II = full 10-s strip, limb leads completed over
                 0-5 s with the exact Einthoven/Goldberger identities, everything not recorded = 0 (masked).
                 v2 fix: samples outside a lead's drawn polyline (2 PDFs) are masked instead of
                 extrapolated flat.
  aligned_meas   as aligned but WITHOUT the derived limb-lead segments (measured samples only)
  leadII12       12-lead model, only lead II (10 s) given, other leads 0
  onelead_II     1-lead ECGFounder on the 10-s lead-II strip (genuinely continuous; but the 1-lead model
                 was pre-trained on lead I -> lead mismatch)
  tiled          2.5-s segments repeated 4x (FABRICATES repetition; kept only as the phase-1 reference)

Per construction: pooled features of stages 4, 5, 6 (mean) and stage 6 (max), pre-trained 150 logits.
For 'aligned' also the stage-5 output [1024 x 40] and the stage-6 block-2 output [1024 x 20], cached for
partial fine-tuning of the last stage / last block.

    python -m research_experiments.phase_2.ecgf_features
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research_experiments.common import ROOT, load_meta
from research_experiments.ecgfounder.inputs import LEADS, SEG, _zscore, bandpass, build
from research_experiments.ecgfounder.model import ARCH, _state_dict
from research_experiments.ecgfounder.net1d import Net1D
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.p2common import MOD, RES, guard

ONE_LEAD = ROOT / "research_experiments" / "phase_2" / "models" / "pretrained" / "1_lead_ECGFounder.pth"
ONE_LEAD_SHA = "f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603"


def load_model(one_lead=False):
    if not one_lead:
        sd = _state_dict()
        m = Net1D(n_classes=150, **ARCH)
    else:
        import hashlib
        h = hashlib.sha256(open(ONE_LEAD, "rb").read()).hexdigest(); assert h == ONE_LEAD_SHA
        from numpy._core.multiarray import scalar
        allowed = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
        with torch.serialization.safe_globals(allowed):
            sd = torch.load(ONE_LEAD, map_location="cpu", weights_only=True)["state_dict"]
        m = Net1D(n_classes=150, **dict(ARCH, in_channels=1))
    m.load_state_dict(sd, strict=True)
    return m.eval()


@torch.no_grad()
def run(model, X, keep_cache=False, bs=16):
    f4, f5, f6, f6max, logits, c5, c62 = [], [], [], [], [], [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs])
        out = model.first_activation(model.first_conv(x))
        for s in range(model.n_stages):
            st = model.stage_list[s]
            if s == 6:
                if keep_cache:
                    c5.append(out.numpy().astype(np.float32))
                for b, blk in enumerate(st.block_list):
                    out = blk(out)
                    if b == 2 and keep_cache:
                        c62.append(out.numpy().astype(np.float32))
            else:
                out = st(out)
            if s == 4:
                f4.append(out.mean(-1).numpy())
            if s == 5:
                f5.append(out.mean(-1).numpy())
        f6.append(out.mean(-1).numpy()); f6max.append(out.amax(-1).numpy())
        logits.append(model.dense(out.mean(-1)).numpy())
    cat = lambda L: np.concatenate(L) if L else None
    return dict(f4=cat(f4), f5=cat(f5), f6=cat(f6), f6max=cat(f6max), logits=cat(logits), c5=cat(c5), c62=cat(c62))


def mask_gaps(arr, g, lead_names=LEADS):
    """Zero samples outside a lead's drawn polyline (aligned construction, 12 x 5000)."""
    for i, n in enumerate(lead_names):
        if n == "II":
            s, e = g.get("rhythm_start_gap_s", 0), g.get("rhythm_end_gap_s", 0); off = 0
        else:
            s, e = g.get(f"{n}_start_gap_s", 0), g.get(f"{n}_end_gap_s", 0)
            off = {"I": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}[n] * SEG
        if s > 0.02:
            arr[i, off:off + int(s * 500)] = 0
        if e > 0.02:
            end = off + (4 * SEG if n == "II" else SEG)
            arr[i, end - int(e * 500):end] = 0
    return arr


def build_p2(leads, rhythm, mode, g):
    if mode == "aligned":
        return mask_gaps(build(leads, rhythm, "aligned"), g)
    if mode == "tiled":
        return build(leads, rhythm, "tiled")
    if mode == "aligned_meas":
        arr = np.zeros((12, 4 * SEG)); mask = np.zeros_like(arr, bool)
        col = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
        for i, n in enumerate(LEADS):
            if n == "II":
                arr[i] = bandpass(rhythm.astype(np.float64)); mask[i] = True
            else:
                c = col[n]; arr[i, c * SEG:(c + 1) * SEG] = bandpass(leads[i].astype(np.float64)); mask[i, c * SEG:(c + 1) * SEG] = True
        return mask_gaps(_zscore(arr, mask).astype(np.float32), g)
    if mode == "leadII12":
        arr = np.zeros((12, 4 * SEG)); mask = np.zeros_like(arr, bool)
        arr[1] = bandpass(rhythm.astype(np.float64)); mask[1] = True
        return _zscore(arr, mask).astype(np.float32)
    if mode == "onelead_II":
        x = bandpass(rhythm.astype(np.float64))[None]
        return _zscore(x, np.ones_like(x, bool)).astype(np.float32)
    raise ValueError(mode)


def main():
    torch.set_num_threads(1)
    meta = load_meta()
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[meta.ecg_id].source
    with Pool(1) as pool:
        G = pool.map(geometry, src.values, chunksize=32)
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    ii = np.array([idx[e] for e in meta.ecg_id])
    L, R = z["leads"][ii], z["rhythm"][ii]
    m12, m1 = load_model(False), load_model(True)
    info = {}
    for mode in ("aligned", "aligned_meas", "leadII12", "onelead_II", "tiled"):
        t0 = time.time()
        X = np.stack([build_p2(l, r, mode, g) for l, r, g in zip(L, R, G)])
        tb = time.time() - t0; t0 = time.time()
        out = run(m1 if mode == "onelead_II" else m12, X, keep_cache=(mode == "aligned"))
        tf = time.time() - t0
        ids = np.asarray(meta.ecg_id.tolist(), dtype="U32")
        np.savez(guard(MOD / f"ecgf_{mode}_features.npz"), ecg_id=ids, **{k: v for k, v in out.items() if k not in ("c5", "c62") and v is not None})
        if mode == "aligned":
            np.save(guard(MOD / "ecgf_aligned_cache_stage5out.npy"), out["c5"])
            np.save(guard(MOD / "ecgf_aligned_cache_stage6block2out.npy"), out["c62"])
        nz = float((X != 0).mean())
        info[mode] = dict(input_shape=list(X.shape[1:]), nonzero_fraction=nz, build_s=tb, forward_s=tf)
        print(mode, info[mode], flush=True)
    (RES / "ecgf_features_info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
