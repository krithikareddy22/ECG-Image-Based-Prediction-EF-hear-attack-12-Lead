"""Build the image caches for the image-only route (IMG_PLAN.md). Laptop only for EchoNext.
    python -m research_experiments.public_ef.img_prep --hospital                         (837 development ECG images)
    set ECHONEXT_DIR=...  &  set ECHONEXT_MINI_DIR=...
    python -m research_experiments.public_ef.img_prep --echonext --splits val,test       (EchoNext drawn as printouts)
Outputs (git-ignored data/private/): img_hospital.npy + img_hospital_ids.npy, img_echonext.npy (memmap) +
img_echonext_meta.npz; aggregates: results/img_prep_summary.json.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research_experiments.common import ROOT
from research_experiments.public_ef import img_render as IR
from research_experiments.public_ef.en_prep import CLS, PRIV, RES, norm_params, source_dir, to_mv

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
COL = {"I": 0, "II": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1, "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3}
HW = (320, 640)


def dev_manifest():
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    return man[man.split != "test"].reset_index(drop=True)                 # development ECGs only


def _init(template_pdf):
    IR.set_template(template_pdf)


def _hosp(path):
    return IR.hospital_image(path, HW)


def _drawn(x250):
    """(12, 2500) mV at 250 Hz -> printout layout at native 250 Hz -> image."""
    leads = np.stack([x250[i, COL[n] * 625:(COL[n] + 1) * 625] for i, n in enumerate(LEADS)])
    return IR.drawn_image(leads, x250[1], fs=250, out_hw=HW)


def hospital(jobs, limit=None):
    dev = dev_manifest()
    if limit:
        dev = dev.iloc[:limit]
    t0 = time.time()
    with Pool(jobs, initializer=_init, initargs=(dev.source.iloc[0],)) as p:
        imgs = p.map(_hosp, dev.source.tolist(), chunksize=8)
    PRIV.mkdir(parents=True, exist_ok=True)
    np.save(PRIV / "img_hospital.npy", np.stack(imgs).astype(np.uint8))
    np.save(PRIV / "img_hospital_ids.npy", np.asarray(dev.ecg_id.tolist(), dtype="U32"))
    return dict(hospital_images=len(imgs), shape=list(HW), seconds=round(time.time() - t0))


def echonext(splits, jobs, limit=None):
    src = source_dir(); params = norm_params(); dev = dev_manifest()
    meta = pd.read_csv(src / "echonext_metadata_100k.csv")
    rows = []
    for sp in splits:
        M = meta[meta.split == sp].reset_index(drop=True)
        keep = np.flatnonzero((M.lvef_value.notna() & M.sex.isin(["male", "female"])).values)
        rows += [(sp, i, M.lvef_value.iloc[i], M.sex.iloc[i] == "male", str(M.patient_key.iloc[i])) for i in keep]
    if limit:
        rows = rows[:limit]
    n = len(rows)
    out = np.lib.format.open_memmap(PRIV / "img_echonext.npy", mode="w+", dtype=np.uint8, shape=(n, *HW))
    t0 = time.time(); pos = 0
    with Pool(jobs, initializer=_init, initargs=(dev.source.iloc[0],)) as p:
        for sp in splits:
            W = np.load(src / f"EchoNext_{sp}_waveforms.npy", mmap_mode="r")
            normalised = float(np.asarray(W[: min(200, len(W))], float).std()) < 5
            idx = [r[1] for r in rows if r[0] == sp]
            for s in range(0, len(idx), 512):
                chunk = [to_mv(W[i, 0], params, normalised) for i in idx[s:s + 512]]
                imgs = p.map(_drawn, chunk, chunksize=16)
                out[pos:pos + len(imgs)] = np.stack(imgs); pos += len(imgs)
                print(f"{sp}: {pos}/{n}  {time.time() - t0:.0f}s", flush=True)
    out.flush()
    ef = np.array([r[2] for r in rows], float); male = np.array([r[3] for r in rows], bool)
    from research_experiments.phase_6a_normal_mild_lstm.data_check import rule
    y = np.array([CLS[rule(e, "M" if m else "F")] for e, m in zip(ef, male)])
    np.savez(PRIV / "img_echonext_meta.npz", split=np.asarray([r[0] for r in rows], dtype="U8"), y=y, ef=ef, male=male,
             patient_key=np.asarray([r[4] for r in rows], dtype="U64"))
    return dict(echonext_images=n, splits=splits, class_counts={c: int((y == i).sum()) for c, i in CLS.items()},
                patients=int(len(set(r[4] for r in rows))), seconds=round(time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospital", action="store_true"); ap.add_argument("--echonext", action="store_true")
    ap.add_argument("--splits", default="val,test"); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    PRIV.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    f = RES / "img_prep_summary.json"
    summ = json.loads(f.read_text()) if f.exists() else {}
    if a.hospital:
        summ["hospital"] = hospital(a.jobs, a.limit)
    if a.echonext:
        summ["echonext"] = echonext(a.splits.split(","), a.jobs, a.limit)
    f.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
