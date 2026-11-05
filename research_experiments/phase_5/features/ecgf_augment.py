"""E7 support: physiologically plausible waveform augmentation, re-embedded with the frozen ECGFounder (aligned input,
exactly the phase-2 construction: phase_2.ecgf_features.build_p2 + run). Development ECGs only; label-free.
Augmentations (per ECG, seeded): global amplitude scale U(0.85, 1.15) (electrode / body-habitus gain), per-lead gain
jitter U(0.95, 1.05), additive white noise sigma 0.015 mV, baseline wander (one sinusoid 0.05-0.5 Hz, amplitude <= 0.1 mV).
No time warping (the aligned layout fixes lead timing). Step 0 re-embeds the UN-augmented signals and must reproduce
the committed phase-2 f6 features (validation). Output (git-ignored): phase_5/data/private/features/ecgf_aug{k}.npz
    python -m research_experiments.phase_5.features.ecgf_augment
"""
import json

import numpy as np
import pandas as pd
import torch

from research_experiments.common import ROOT
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.ecgf_features import build_p2, load_model, run
from research_experiments.phase_5.p5common import FEAT5, IDS, RES5, ecgf_f6, guard5

K_AUG = 2


def augment(leads, rhythm, rng):
    s = rng.uniform(0.85, 1.15); g = rng.uniform(0.95, 1.05, size=(12, 1))
    t_l, t_r = np.arange(leads.shape[1]) / 500, np.arange(rhythm.shape[0]) / 500
    f, a, ph = rng.uniform(0.05, 0.5), rng.uniform(0, 0.1), rng.uniform(0, 2 * np.pi)
    L = leads * s * g + rng.normal(0, 0.015, leads.shape) + a * np.sin(2 * np.pi * f * t_l + ph)[None]
    R = rhythm * s * g[1, 0] + rng.normal(0, 0.015, rhythm.shape) + a * np.sin(2 * np.pi * f * t_r + ph)
    return L.astype(np.float32), R.astype(np.float32)


def main():
    torch.set_num_threads(2)
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[IDS].source.values
    G = [geometry(s) for s in src]                     # page layout gaps; file paths stay in memory
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[IDS].values
    L, R = z["leads"][pos], z["rhythm"][pos]
    m = load_model(False)
    X0 = np.stack([build_p2(l, r, "aligned", g) for l, r, g in zip(L, R, G)])
    f6 = run(m, X0)["f6"]
    ref = ecgf_f6()
    val = dict(reproduction_max_abs_diff=float(np.abs(f6 - ref).max()), reproduction_mean_abs=float(np.abs(ref).mean()))
    print("reproduction of committed f6:", val, flush=True)
    assert val["reproduction_max_abs_diff"] < 1e-3, "aligned pipeline does not reproduce the committed features"
    guard5(FEAT5).mkdir(parents=True, exist_ok=True)
    for k in range(1, K_AUG + 1):
        out = FEAT5 / f"ecgf_aug{k}.npz"
        if out.exists():
            continue
        rng = np.random.default_rng(5000 + k)
        X = np.stack([build_p2(*augment(l, r, rng), "aligned", g) for l, r, g in zip(L, R, G)])
        fa = run(m, X)["f6"]
        np.savez(out, f6=fa.astype(np.float32))
        val[f"aug{k}_mean_abs_shift_vs_original"] = float(np.abs(fa - ref).mean())
        print(f"aug {k} done", flush=True)
    (RES5 / "ecgf_augmentation_validation.json").write_text(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
