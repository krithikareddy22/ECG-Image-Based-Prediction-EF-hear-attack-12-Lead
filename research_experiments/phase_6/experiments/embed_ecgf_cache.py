"""Experiment A prerequisite: frozen ECGFounder activations for the 837 DEVELOPMENT ECGs only (label-free).

Input = the Phase-2 "aligned" construction (phase_2.ecgf_features.build_p2 + PDF layout gaps from
phase_2.audit_waveform.geometry; file paths stay in memory). The frozen network is run once per ECG and caches:
  stage4  output of stage 4 (input of stage 5)  -> A2 (stages 5-6 trainable)
  stage5  output of stage 5 (input of stage 6)  -> A1 (stage 6 trainable)
  f6      mean-pooled stage-6 output             -> must reproduce the committed Phase-2 f6 (max |diff| < 1e-3)
Because Net1D has no batch-norm and no dropout, training on these caches is exactly training the full network with
the lower stages frozen. Output (git-ignored): phase_6/data/private/embeddings/ecgf_cache.npz
Validation (aggregate): phase_6/results/ecgf_cache_validation.json
    python -m research_experiments.phase_6.experiments.embed_ecgf_cache
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from research_experiments.common import ROOT
from research_experiments.ecgfounder.model import verify_checkpoint
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.ecgf_features import build_p2, load_model
from research_experiments.phase_6.p6common import EMB6, IDS, RES6, ecgf_f6, guard6

OUT = EMB6 / "ecgf_cache.npz"


def dev_aligned_inputs(ids=IDS):
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[ids].source.values
    G = [geometry(s) for s in src]
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[ids].values
    return np.stack([build_p2(l, r, "aligned", g) for l, r, g in zip(z["leads"][pos], z["rhythm"][pos], G)]), G


@torch.no_grad()
def run_cache(model, X, bs=16):
    s4, s5, f6 = [], [], []
    for i in range(0, len(X), bs):
        out = model.first_activation(model.first_conv(torch.from_numpy(X[i:i + bs])))
        for s in range(model.n_stages):
            out = model.stage_list[s](out)
            if s == 4:
                s4.append(out.numpy().astype(np.float32))
            if s == 5:
                s5.append(out.numpy().astype(np.float32))
        f6.append(out.mean(-1).numpy())
    return np.concatenate(s4), np.concatenate(s5), np.concatenate(f6)


def load_cache():
    z = np.load(OUT)
    assert (z["ecg_id"] == np.asarray(IDS, dtype="U32")).all()
    return z["stage4"], z["stage5"], z["f6"]


def main():
    if OUT.exists():
        print("exists:", OUT.relative_to(ROOT)); return
    torch.set_num_threads(4); verify_checkpoint()
    t0 = time.time(); X, _ = dev_aligned_inputs(); tb = time.time() - t0
    m = load_model(False)
    t0 = time.time(); s4, s5, f6 = run_cache(m, X); tf = time.time() - t0
    ref = ecgf_f6()
    val = dict(n_ecgs=len(IDS), stage4_shape=list(s4.shape[1:]), stage5_shape=list(s5.shape[1:]),
               f6_reproduction_max_abs_diff=float(np.abs(f6 - ref).max()), f6_mean_abs=float(np.abs(ref).mean()),
               build_minutes=tb / 60, forward_minutes=tf / 60)
    print(json.dumps(val, indent=1))
    assert val["f6_reproduction_max_abs_diff"] < 1e-3, "aligned pipeline does not reproduce the committed f6"
    guard6(EMB6).mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ecg_id=np.asarray(IDS, dtype="U32"), stage4=s4, stage5=s5, f6=f6.astype(np.float32))
    (guard6(RES6) / "ecgf_cache_validation.json").write_text(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
