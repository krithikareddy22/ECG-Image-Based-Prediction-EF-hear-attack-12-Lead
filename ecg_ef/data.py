"""In-memory dataset + ECG-safe augmentation."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .preprocess import CACHE_DIR, MANIFEST


def load_split(split: str, variant: str, img_hw: tuple[int, int]):
    """Return (images uint8 [N,1,H,W], labels int64 [N], dataframe) for one split."""
    df = pd.read_csv(MANIFEST)
    df = df[df.split == split].reset_index(drop=True)
    h, w = img_hw
    imgs = np.empty((len(df), 1, h, w), np.uint8)
    for i, ecg_id in enumerate(df.ecg_id):
        im = cv2.imread(str(CACHE_DIR / variant / f"{ecg_id}.png"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"no cached '{variant}' image for {ecg_id}; run "
                                    f"python -m ecg_ef.preprocess --variants {variant}")
        if im.shape != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        imgs[i, 0] = im
    return torch.from_numpy(imgs), torch.tensor(df.label_idx.values), df


def augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Batched, label-preserving augmentation for ECG images (float, [B,1,H,W]).

    * small rotation (paper not perfectly straight on a scan)       +-2 deg
    * independent x/y scaling (paper speed / gain differences)       +-6% / +-10%
    * translation                                                    +-3%
    * random time-segment erasing (lead-off / artefact)
    No flips: a horizontal flip reverses time and a vertical flip inverts the
    polarity of every lead, both of which change the diagnosis.
    """
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(2) * strength
    sx = 1 + (torch.rand(b, device=dev) * 2 - 1) * 0.06 * strength
    sy = 1 + (torch.rand(b, device=dev) * 2 - 1) * 0.10 * strength
    tx = (torch.rand(b, device=dev) * 2 - 1) * 0.03 * strength
    ty = (torch.rand(b, device=dev) * 2 - 1) * 0.03 * strength
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([cos / sx, -sin / sx, tx], 1),
        torch.stack([sin / sy, cos / sy, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # random erasing of one narrow vertical band in ~30% of images
    h, w = x.shape[-2:]
    for i in torch.nonzero(torch.rand(b) < 0.3 * strength).flatten().tolist():
        bw = int(w * float(torch.empty(1).uniform_(0.02, 0.06)))
        x0 = int(torch.randint(0, w - bw, (1,)))
        x[i, :, :, x0:x0 + bw] = 0
    return x


class Normalizer:
    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit(cls, imgs_u8: torch.Tensor) -> "Normalizer":
        x = imgs_u8.float() / 255
        return cls(x.mean().item(), x.std().item())

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: float tensor already scaled to [0, 1]."""
        return (x01 - self.mean) / self.std

    def state(self):
        return {"mean": self.mean, "std": self.std}
