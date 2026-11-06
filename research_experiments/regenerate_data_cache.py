"""Rebuild the project's git-ignored derived cache `data/` in a FRESH CLONE, safely.

Unlike `python -m ecg_ef.preprocess` (rewrites results/split.csv) and `python -m ecg_ef.header` (re-trains and
overwrites models/header_logreg.joblib + results/header_logreg/), this script writes ONLY into data/ and never
touches results/, models/ or the dataset folders. The committed split results/split.csv is read, not written.
    python -m research_experiments.regenerate_data_cache            # manifest + header features + signals
    python -m research_experiments.regenerate_data_cache --images   # + filtered image cache (only for the image CNN)
"""
import argparse
import hashlib
from multiprocessing import Pool

import pandas as pd

from ecg_ef.config import DATA_DIR
from ecg_ef.preprocess import CACHE_DIR, MANIFEST, SPLIT_CSV, _cache_one, build_manifest

EXPECTED = {"manifest.csv": "2463bd35b02013751474c418c195ef7a7c073699e8bfb5c113bd1848200fc4a8",
            "header_features.csv": "a9a379035729d806159185af53b1408644e63a53c820ec0cac95c394b1a4c638"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", action="store_true"); a = ap.parse_args()
    split_before = sha(SPLIT_CSV)
    DATA_DIR.mkdir(exist_ok=True)
    if not MANIFEST.exists():
        df = build_manifest()
        saved = pd.read_csv(SPLIT_CSV).set_index("ecg_id").split          # read-only
        assert set(df.ecg_id) == set(saved.index), "dataset and results/split.csv disagree"
        df["split"] = df.ecg_id.map(saved)
        assert (df.groupby("patient").split.nunique() == 1).all(), "patient leakage"
        df.to_csv(MANIFEST, index=False)
    print("manifest sha256 OK" if sha(MANIFEST) == EXPECTED["manifest.csv"] else "WARNING: manifest differs from the recorded version")
    if not (DATA_DIR / "header_features.csv").exists():
        from ecg_ef.header import build_features
        build_features()
    print("header_features sha256 OK" if sha(DATA_DIR / "header_features.csv") == EXPECTED["header_features.csv"] else "WARNING: header_features differs")
    if not (DATA_DIR / "signals.npz").exists():
        from ecg_ef import signals
        signals.main()
    if a.images:
        df = pd.read_csv(MANIFEST)
        (CACHE_DIR / "filtered").mkdir(parents=True, exist_ok=True)
        with Pool(4) as pool:
            list(pool.imap_unordered(_cache_one, [(s, e, ["filtered"]) for s, e in zip(df.source, df.ecg_id)], chunksize=8))
    assert sha(SPLIT_CSV) == split_before, "results/split.csv must not change"
    print("data/ cache ready; results/split.csv unchanged")


if __name__ == "__main__":
    main()
