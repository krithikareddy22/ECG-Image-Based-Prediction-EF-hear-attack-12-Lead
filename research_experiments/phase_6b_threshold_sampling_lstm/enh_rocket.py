"""ENH F-rocket (AMENDMENT_ENH.md): label-free MiniRocket features of the synth input (12 x 1000, 100 Hz).
Standalone on purpose: runs in a separate virtual environment with aeon 1.6.0 (aeon needs an older numpy than the
project). It reads and writes only the git-ignored ENH folder; no labels are read. One transform is fitted per outer
training fold (5 folds x 3 repeats) and applied to all 837 development ECGs.
    <venv>/bin/python research_experiments/phase_6b_threshold_sampling_lstm/enh_rocket.py <private enh folder>
"""
import sys
from pathlib import Path

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket

d = Path(sys.argv[1]).resolve()
assert d.parts[-3:] == ("data", "private", "enh"), d
z = np.load(d / "rocket_in.npz")
X, folds = z["X"].astype(np.float32), z["folds"]
for r in range(folds.shape[0]):
    for k in range(5):
        out = d / f"rocket_r{r}_k{k}.npy"
        if out.exists():
            continue
        tr = folds[r] != k
        m = MiniRocket(n_kernels=10000, random_state=0, n_jobs=4).fit(X[tr])
        np.save(out, m.transform(X).astype(np.float16))
        print("done", out.name, flush=True)
