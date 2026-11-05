"""IMG laptop step 4: fidelity of the drawn images (IMG_PLAN.md). Redraws n development ECGs from their digitised traces
(data/signals.npz) with img_render.drawn_image and compares each with img_render.hospital_image of the same PDF.
Pixel correlation (Pearson over all pixels) and trace-mask IoU (pixels > 127). Development ECGs only; aggregates only.
    python -m research_experiments.public_ef.img_fidelity [--n 30]
"""
import argparse
import json

import numpy as np
import pandas as pd

from research_experiments.common import ROOT
from research_experiments.public_ef import img_render as IR
from research_experiments.public_ef.en_prep import RES
from research_experiments.public_ef.img_prep import HW, dev_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    dev = dev_manifest()
    pick = dev.iloc[np.sort(np.random.default_rng(0).choice(len(dev), a.n, replace=False))]
    IR.set_template(dev.source.iloc[0])
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"])
    cor, iou = [], []
    for _, row in pick.iterrows():
        i = pos[row.ecg_id]
        d = IR.drawn_image(z["leads"][i], z["rhythm"][i], fs=500, out_hw=HW).astype(float)
        h = IR.hospital_image(ROOT / row.source, HW).astype(float)
        cor.append(float(np.corrcoef(d.ravel(), h.ravel())[0, 1]))
        md, mh = d > 127, h > 127
        iou.append(float((md & mh).sum() / max((md | mh).sum(), 1)))
    out = dict(n=a.n, pixel_correlation=dict(median=float(np.median(cor)), min=float(np.min(cor))),
               trace_mask_iou=dict(median=float(np.median(iou)), min=float(np.min(iou))),
               reference_cloud=dict(pixel_correlation_median=0.993, trace_mask_iou_median=0.967))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "img_fidelity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
