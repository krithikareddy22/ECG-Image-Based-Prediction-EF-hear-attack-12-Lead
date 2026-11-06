"""Load the pre-trained 12-lead ECGFounder (Net1D) safely and expose features.

Checkpoint: https://huggingface.co/PKUDigitalHealth/ECGFounder (revision
d9b1793951b2342f5f7e84f1ac03cd37f8a08724), file 12_lead_ECGFounder.pth,
SHA-256 ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997 (MIT licence).
Loaded with torch.load(weights_only=True); only numpy's scalar/dtype types are allow-listed
(the file stores a numpy scalar besides the tensors) - no arbitrary unpickling.
Architecture = finetune_model.ft_12lead_ECGFounder: use_bn=False, use_do=False.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .net1d import Net1D

CKPT = Path(__file__).resolve().parent.parent / "models" / "pretrained" / "12_lead_ECGFounder.pth"
SHA256 = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"
ARCH = dict(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16,
            verbose=False, use_bn=False, use_do=False)


def verify_checkpoint() -> str:
    h = hashlib.sha256()
    with open(CKPT, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    assert h.hexdigest() == SHA256, "ECGFounder checkpoint hash mismatch"
    return h.hexdigest()


def _state_dict():
    from numpy._core.multiarray import scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType,
               np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    return ck["state_dict"]


def load_ecgfounder(n_classes: int | None = None) -> Net1D:
    """n_classes=None: keep the pre-trained 150-label head; else a new randomly initialised head."""
    sd = _state_dict()
    model = Net1D(n_classes=150, **ARCH)
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    if n_classes is not None:
        model.dense = nn.Linear(model.dense.in_features, n_classes)
    return model


@torch.no_grad()
def features(model: Net1D, x: np.ndarray, bs: int = 32):
    """-> (deep features [N,1024] = mean-pooled last stage, pre-trained 150 logits [N,150])."""
    model.eval(); model.return_features = True
    f, lg = [], []
    for i in range(0, len(x), bs):
        out, deep = model(torch.from_numpy(x[i:i + bs]))
        f.append(deep.numpy()); lg.append(out.numpy())
    model.return_features = False
    return np.concatenate(f), np.concatenate(lg)
