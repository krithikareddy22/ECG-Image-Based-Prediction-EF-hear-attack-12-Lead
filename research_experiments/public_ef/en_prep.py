"""EchoNext → printout-format ECGFounder embeddings (PUBLIC_EF_PLAN.md). Runs on the owner's laptop only.
Reads EchoNext from ECHONEXT_DIR (outside the repository); writes only the git-ignored data/private/ cache and an
aggregate-only JSON. Never prints EchoNext rows.
    set ECHONEXT_DIR=C:\\krithika_major\\public_data\\echonext
    set ECHONEXT_MINI_DIR=<clone of PierreElias/IntroECG>\\7-EchoNext Minimodel      (for the published normalisation)
    python -m research_experiments.public_ef.en_prep --splits val,test
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from research_experiments.common import ROOT
from research_experiments.ecgfounder.inputs import LEADS, build
from research_experiments.phase_2.ecgf_features import load_model, run
from research_experiments.phase_6a_normal_mild_lstm.data_check import rule
from research_experiments.phase_6b_threshold_sampling_lstm import enh_build as EB

HERE = Path(__file__).resolve().parent
PRIV = HERE / "data" / "private"; RES = HERE / "results"
COL = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
CLS = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
MUSE_UV = 4.88


def source_dir():
    d = Path(os.environ["ECHONEXT_DIR"]).resolve()
    assert ROOT.resolve() not in d.parents and d != ROOT.resolve(), "ECHONEXT_DIR must be outside the repository"
    return d


def norm_params():
    p = Path(os.environ["ECHONEXT_MINI_DIR"]) / "models" / "echonext_multilabel_minimodel" / "waveform_normalization_params.json"
    return json.loads(p.read_text())


def to_mv(w, params, normalised):
    """w: (2500, 12) EchoNext array -> (12, 2500) mV."""
    x = np.asarray(w, float).T
    if normalised:
        x = x * np.array(params["std"])[:, None] + np.array(params["mean"])[:, None]
    return x * MUSE_UV / 1000.0


def printout(x250):
    """(12, 2500) mV at 250 Hz -> hospital printout layout: leads (12, 1250) at 500 Hz + 10-s lead II rhythm (5000)."""
    x = resample_poly(x250, 2, 1, axis=1)[:, :5000]
    leads = np.stack([x[i, COL[n] * 1250:(COL[n] + 1) * 1250] for i, n in enumerate(LEADS)])
    return leads, x[1]


def synth_input(leads, rhythm):
    y, _, _ = EB.build_one(leads, rhythm)
    return ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="val,test"); ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    src = source_dir(); params = norm_params()
    meta = pd.read_csv(src / "echonext_metadata_100k.csv")
    model = load_model(False)
    out = {k: [] for k in ("split", "row", "patient_key", "ef", "male", "y", "f6_aligned", "logits_aligned", "f6_synth", "logits_synth")}
    agg = dict(splits={}, normalised_input_detected={})
    t0 = time.time()
    for sp in a.splits.split(","):
        W = np.load(src / f"EchoNext_{sp}_waveforms.npy", mmap_mode="r")
        M = meta[meta.split == sp].reset_index(drop=True)
        assert len(W) == len(M), f"{sp}: waveforms {len(W)} != metadata rows {len(M)}"
        sd = float(np.asarray(W[: min(200, len(W))], float).std())
        normalised = sd < 5                                   # z-scored inputs have SD ~1; raw MUSE units ~30-60
        agg["normalised_input_detected"][sp] = dict(normalised=normalised, sample_sd=round(sd, 3))
        keep = M.lvef_value.notna() & M.sex.isin(["male", "female"])
        idx = np.flatnonzero(keep.values)
        y = np.array([CLS[rule(e, "M" if s == "male" else "F")] for e, s in zip(M.lvef_value.values[idx], M.sex.values[idx])])
        agg["splits"][sp] = dict(rows=int(len(M)), with_lvef_and_sex=int(len(idx)), class_counts={c: int((y == i).sum()) for c, i in CLS.items()},
                                 patients=int(M.patient_key.iloc[idx].nunique()))
        for s in range(0, len(idx), a.chunk):
            sel = idx[s:s + a.chunk]
            A, S = [], []
            for i in sel:
                leads, rhythm = printout(to_mv(W[i, 0], params, normalised))
                A.append(build(leads, rhythm, "aligned")); S.append(synth_input(leads, rhythm))
            oa, os_ = run(model, np.stack(A).astype(np.float32)), run(model, np.stack(S).astype(np.float32))
            out["f6_aligned"].append(oa["f6"]); out["logits_aligned"].append(oa["logits"])
            out["f6_synth"].append(os_["f6"]); out["logits_synth"].append(os_["logits"])
            print(f"{sp}: {min(s + a.chunk, len(idx))}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
        out["split"].append(np.asarray([sp] * len(idx), dtype="U8")); out["row"].append(idx)
        out["patient_key"].append(np.asarray([str(v) for v in M.patient_key.to_numpy()[idx]], dtype="U64"))
        out["ef"].append(M.lvef_value.to_numpy(dtype=float)[idx])
        out["male"].append(np.asarray(M.sex.to_numpy()[idx] == "male", dtype=bool)); out["y"].append(y)
    PRIV.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    np.savez(PRIV / "en_cache.npz", **{k: np.concatenate(v) for k, v in out.items()})
    agg["total_ecgs"] = int(sum(v["with_lvef_and_sex"] for v in agg["splits"].values())); agg["seconds"] = round(time.time() - t0)
    (RES / "en_prep_summary.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
