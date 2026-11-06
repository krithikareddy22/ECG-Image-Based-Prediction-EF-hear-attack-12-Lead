"""Phase 0b - retrain the baseline ResNet-18 with the ORIGINAL training code (ecg_ef.train.main,
unchanged) and the original arguments (read from models/resnet18_filtered.pt), only redirecting
its output directories into research_experiments/ (the original models/ and results/ are not
touched). Measures training reproducibility (same seed) on CPU.

    python -m research_experiments.reproduce_resnet18 [--seed 42]
"""
import argparse
import sys

import ecg_ef.train as T
from research_experiments.common import MOD, RES, _guard

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
T.MODELS_DIR = _guard(MOD / "repro"); T.RESULTS_DIR = _guard(RES / "repro")
T.MODELS_DIR.mkdir(parents=True, exist_ok=True); T.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.argv = ["train", "--model", "resnet18", "--variant", "filtered", "--epochs", "30", "--patience", "10",
            "--min-epochs", "8", "--batch-size", "16", "--wd", "0.05", "--label-smoothing", "0.05",
            "--class-weight", "balanced", "--ema", "0.99", "--img-size", "256", "512", "--seed", str(a.seed),
            "--threads", "4", "--tag", f"R0_resnet18_filtered_retrain_seed{a.seed}"]
T.main()
