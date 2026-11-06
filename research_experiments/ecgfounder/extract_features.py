"""A1 step 1: frozen ECGFounder features for all 1,047 ECGs, for each input construction.

    python -m research_experiments.ecgfounder.extract_features
-> research_experiments/models/features_ecgfounder_<mode>.npz (git-ignored)
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from research_experiments.common import MOD, RES, ROOT, load_meta, seed_all
from research_experiments.ecgfounder.inputs import build_all
from research_experiments.ecgfounder.model import features, load_ecgfounder, verify_checkpoint


def main():
    torch.set_num_threads(4); seed_all()
    verify_checkpoint()
    meta = load_meta()
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)   # read-only
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    ii = np.array([idx[e] for e in meta.ecg_id])
    leads, rhythm = z["leads"][ii], z["rhythm"][ii]
    model = load_ecgfounder()
    log = {}
    for mode in ("aligned", "tiled", "short"):
        t0 = time.time(); x = build_all(leads, rhythm, mode); tb = time.time() - t0
        assert np.isfinite(x).all()
        t0 = time.time(); f, lg = features(model, x); tf = time.time() - t0
        np.savez(MOD / f"features_ecgfounder_{mode}.npz", ecg_id=np.asarray(meta.ecg_id.tolist(), dtype="U32"), feat=f, logits=lg)
        log[mode] = dict(input_shape=list(x.shape[1:]), build_s_per_ecg=tb / len(x), forward_s_per_ecg=tf / len(x),
                         feat_mean_abs=float(np.abs(f).mean()), n_nonfinite=int((~np.isfinite(f)).sum()))
        print(mode, log[mode], flush=True)
    (RES / "ecgfounder_feature_extraction.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
