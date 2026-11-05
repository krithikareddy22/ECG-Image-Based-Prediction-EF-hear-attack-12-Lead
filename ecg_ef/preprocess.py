"""ECG image preprocessing: PDF/image -> cropped, grid-removed, normalised trace image.

Pipeline (applied identically at training and prediction time):

1. Render   - rasterise the ECG PDF (vector) at 1 px per PDF point, or load a PNG/JPG.
2. Crop     - keep only the 12-lead waveform area. This drops the header that contains
              the patient's name/age/sex and the ECG machine's own text interpretation,
              so the CNN can only learn from the waveforms themselves.
3. Grid filter - the trace is black, the grid is pink/red. In HSV, any coloured grid has a
              high Value (max of R,G,B) while the black trace has a low Value, so
              `darkness = 255 - max(R,G,B)` keeps the trace and suppresses the grid.
4. Threshold - Otsu threshold (clamped) on `darkness` -> binary trace mask.
5. Denoise  - drop tiny connected components (scanner speckle / grid intersections).
6. Thicken  - 3x3 dilation so 1-px lines survive down-sampling.
7. Resize   - area interpolation (anti-aliased) to the cache size.

Output: uint8 image, white trace on black background.

A "raw" variant (inverted grayscale, grid kept, no filtering) is also produced so the
effect of the filtering can be measured.

Run as a script to build the cache, manifest and patient-grouped split:

    python -m ecg_ef.preprocess
"""
from __future__ import annotations

import argparse
import hashlib
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf
from sklearn.model_selection import StratifiedGroupKFold

from .config import (CACHE_H, CACHE_W, CLASS_TO_IDX, CLASSES, DATA_DIR, PAGE_SIZE_PT,
                     RESULTS_DIR, ROOT, WAVEFORM_CLIP_PT)

CACHE_DIR = DATA_DIR / "cache"
MANIFEST = DATA_DIR / "manifest.csv"
SPLIT_CSV = RESULTS_DIR / "split.csv"


# --------------------------------------------------------------------------- loading
def render_pdf_waveform(pdf_path: str | Path, px_per_pt: float = 1.0) -> np.ndarray:
    """Rasterise only the waveform area of page 1 of an ECG PDF. Returns RGB uint8."""
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[0]
        clip = pymupdf.Rect(*WAVEFORM_CLIP_PT)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(px_per_pt, px_per_pt), clip=clip,
                              alpha=False, colorspace=pymupdf.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def load_image_waveform(img_path: str | Path) -> np.ndarray:
    """Load a PNG/JPG ECG. If it looks like a full page of the same layout as the PDFs
    (aspect ~4:3), crop the waveform area proportionally; otherwise assume it is
    already cropped to the waveforms. Returns RGB uint8."""
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    page_aspect = PAGE_SIZE_PT[0] / PAGE_SIZE_PT[1]
    if abs(w / h - page_aspect) < 0.08:
        x0, y0, x1, y1 = WAVEFORM_CLIP_PT
        sx, sy = w / PAGE_SIZE_PT[0], h / PAGE_SIZE_PT[1]
        rgb = rgb[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
    return rgb


def load_waveform(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return render_pdf_waveform(path)
    return load_image_waveform(path)


# --------------------------------------------------------------------------- filters
def filter_trace(rgb: np.ndarray, out_hw: tuple[int, int] = (CACHE_H, CACHE_W)) -> np.ndarray:
    """Grid removal + denoising + thickening + anti-aliased resize. White trace on black."""
    darkness = 255 - rgb.max(axis=2)                     # HSV value -> grid suppressed
    otsu, _ = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu, 60, 160))
    mask = (darkness > thr).astype(np.uint8)

    # remove specks smaller than a few pixels (keeps thin but long trace segments)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6
    mask = keep[labels]

    # scale-aware thickening so the line is ~2-3 source px wide before downsampling
    k = max(2, int(round(rgb.shape[1] / 1000)))
    mask = cv2.dilate(mask * 255, np.ones((k, k), np.uint8))
    out = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    # stretch contrast so the trace is full white after area-averaging
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out


def raw_gray(rgb: np.ndarray, out_hw: tuple[int, int] = (CACHE_H, CACHE_W)) -> np.ndarray:
    """No filtering: inverted grayscale with the grid still visible (for comparison)."""
    gray = 255 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)


# --- classic image filters, for the preprocessing comparison --------------------------
# Each works on the inverted grayscale image (trace bright, background dark, grid kept
# faint) at full resolution, before area-resizing - so only the filter differs.
def _inv_gray(rgb):
    return 255 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize(img, out_hw):
    return cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)


def gaussian_filter(rgb, out_hw=(CACHE_H, CACHE_W)):
    """Gaussian blur (5x5, sigma 1): removes high-frequency noise, slightly softens edges."""
    return _resize(cv2.GaussianBlur(_inv_gray(rgb), (5, 5), 1.0), out_hw)


def median_filter(rgb, out_hw=(CACHE_H, CACHE_W)):
    """3x3 median: removes salt-and-pepper noise (can thin 1-px trace lines)."""
    return _resize(cv2.medianBlur(_inv_gray(rgb), 3), out_hw)


def bilateral_filter(rgb, out_hw=(CACHE_H, CACHE_W)):
    """Edge-preserving smoothing: flattens background/grid texture, keeps trace edges."""
    return _resize(cv2.bilateralFilter(_inv_gray(rgb), 7, 50, 7), out_hw)


def clahe_filter(rgb, out_hw=(CACHE_H, CACHE_W)):
    """CLAHE (clip 2.0, 8x8 tiles): local contrast enhancement of the trace."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _resize(clahe.apply(_inv_gray(rgb)), out_hw)


def adaptive_threshold(rgb, out_hw=(CACHE_H, CACHE_W)):
    """Adaptive (local Gaussian-mean) threshold: separates the dark trace from the
    lighter grid/background without using colour. Offset C=60: the bold grid lines are
    ~80 grey levels below the background, so a smaller C keeps them as 'trace'."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 60)
    return cv2.normalize(_resize(binary, out_hw), None, 0, 255, cv2.NORM_MINMAX)


VARIANTS = {
    "filtered": filter_trace,        # colour grid removal + Otsu + denoise + thicken (default)
    "raw": raw_gray,                 # grayscale only, no filter
    "gaussian": gaussian_filter,
    "median": median_filter,
    "bilateral": bilateral_filter,
    "clahe": clahe_filter,
    "adaptive": adaptive_threshold,
}


def preprocess(path: str | Path, variant: str = "filtered",
               out_hw: tuple[int, int] = (CACHE_H, CACHE_W)) -> np.ndarray:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {list(VARIANTS)}")
    return VARIANTS[variant](load_waveform(path), out_hw)


def prepare_for_model(path: str | Path, variant: str, img_hw: tuple[int, int]) -> np.ndarray:
    """Exactly the transform used for training: filter at cache size, then area-resize
    to the network input size. Returns uint8 [H, W]."""
    im = preprocess(path, variant, (CACHE_H, CACHE_W))
    if im.shape != tuple(img_hw):
        im = cv2.resize(im, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
    return im


# --------------------------------------------------------------------------- dataset
_FNAME = re.compile(r"EK__\[(\d+)\]_(.*?)_*\[\]_")


def parse_filename(path: Path) -> tuple[str, str]:
    """Return (timestamp, patient_key). The patient key is a hash of the normalised
    name, used only for grouping so the same person is never in both train and test."""
    m = _FNAME.match(path.name)
    if not m:
        raise ValueError(f"Unexpected file name: {path.name}")
    ts, name = m.group(1), m.group(2)
    norm = re.sub(r"[^A-Z]", "", name.upper())
    return ts, hashlib.sha1(norm.encode()).hexdigest()[:12]


def build_manifest() -> pd.DataFrame:
    rows = []
    for pdf in sorted(ROOT.glob("LV_ECG_Dataset_*/*/*.pdf")):
        label = pdf.parent.name
        if label not in CLASS_TO_IDX:
            continue
        year = pdf.parent.parent.name.rsplit("_", 1)[-1]
        ts, patient = parse_filename(pdf)
        rows.append(dict(ecg_id=f"{year}_{ts}", year=int(year), label=label,
                         label_idx=CLASS_TO_IDX[label], patient=patient,
                         source=str(pdf.relative_to(ROOT))))
    df = pd.DataFrame(rows)
    assert df.ecg_id.is_unique
    return df


def make_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Patient-grouped, class-stratified split: ~64% train / ~16% val / 20% test."""
    split = pd.Series("train", index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df.label_idx, df.patient))
    split.iloc[test_idx] = "test"
    rest = df.index[split == "train"]
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    _, val_pos = next(sgkf2.split(df.loc[rest], df.loc[rest, "label_idx"],
                                  df.loc[rest, "patient"]))
    split.loc[rest[val_pos]] = "val"
    return split


def _cache_one(args):
    src, ecg_id, variants = args
    todo = [(v, VARIANTS[v]) for v in variants
            if not (CACHE_DIR / v / f"{ecg_id}.png").exists()]
    if todo:
        rgb = render_pdf_waveform(ROOT / src)
        for variant, fn in todo:
            cv2.imwrite(str(CACHE_DIR / variant / f"{ecg_id}.png"), fn(rgb))
    return ecg_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["filtered", "raw"],
                    help=f"image variants to cache: {list(VARIANTS)} or 'all'")
    ap.add_argument("--resplit", action="store_true",
                    help="make a new split instead of reusing results/split.csv")
    args = ap.parse_args()

    df = build_manifest()
    if SPLIT_CSV.exists() and not args.resplit:
        # reuse the committed split so results are comparable across machines
        saved = pd.read_csv(SPLIT_CSV).set_index("ecg_id").split
        missing = set(df.ecg_id) - set(saved.index)
        if missing:
            raise SystemExit(f"{len(missing)} ECGs are not in {SPLIT_CSV}; run with --resplit")
        df["split"] = df.ecg_id.map(saved)
        print(f"using the saved split in {SPLIT_CSV}")
    else:
        df["split"] = make_split(df)
    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    for variant in variants:
        (CACHE_DIR / variant).mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    jobs = [(s, e, variants) for s, e in zip(df.source, df.ecg_id)]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, jobs, chunksize=8), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"  cached {i}/{len(jobs)}", flush=True)

    df.to_csv(MANIFEST, index=False)                       # git-ignored (has file paths)
    df[["ecg_id", "year", "label", "split"]].to_csv(SPLIT_CSV, index=False)  # no names

    # sanity: no patient in more than one split
    leak = df.groupby("patient").split.nunique()
    assert (leak == 1).all(), "patient leakage across splits"
    print(pd.crosstab(df.label, df.split).reindex(CLASSES))
    print(f"patients: {df.patient.nunique()}  ECGs: {len(df)}")


if __name__ == "__main__":
    main()
