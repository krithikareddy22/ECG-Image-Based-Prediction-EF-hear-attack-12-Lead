"""E1: feature-enhanced MLP fusion. Hypothesis: adding one validated engineered ECG feature family to the phase-2 MLP
fusion (header numeric || ECGFounder f6; MLPF config, inner-val selection) raises four-class accuracy.
Baseline: the identical MLP fusion without the family (P5A-import-mlpfusion-v1; same folds and seeds).
Header + ECGF + phase-3 clinical features is the completed phase-3 3-way fusion (P5A-import-mlp3way-v1), re-used.
    python -m research_experiments.phase_5.experiments.e1_feature_mlp --jobs 3 --part 0
    python -m research_experiments.phase_5.experiments.e1_feature_mlp --table
"""
import argparse

import torch

from research_experiments.phase_5.experiments.nested import MLPF
from research_experiments.phase_5.experiments.runner import cv_run
from research_experiments.phase_5.p5common import REPEATS, RES5, SEEDS
from research_experiments.phase_5.registry_ext import E1_FAMILIES, NEW_FAMILIES
from research_experiments.phase_5.validation.stats import family_table

CMD = "python -m research_experiments.phase_5.experiments.e1_feature_mlp"


def blocks(f):
    return ["hdr", "ecgf", "feat:" + ("+".join(NEW_FAMILIES) if f == "allnew" else f)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--table", action="store_true"); a = ap.parse_args()
    if a.table:
        T = family_table([f"P5D-E1-mlp-{f}-v1" for f in E1_FAMILIES] + ["P5A-import-mlp3way-v1"], "P5A-import-mlpfusion-v1")
        T.to_csv(RES5 / "e1_feature_mlp.csv", index=False, float_format="%.4f")
        print(T[["model", "accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "min_class_recall_mean", "d_accuracy",
                 "p_accuracy", "p_holm_accuracy", "same_sign_accuracy"]].round(3).to_string(index=False))
        return
    torch.set_num_threads(1)
    J = [(f, s, r) for f in E1_FAMILIES for r in REPEATS for s in SEEDS][a.part::a.jobs]
    for f, s, r in J:
        cv_run(f"P5D-E1-mlp-{f}-v1", blocks(f), "p2heads", s, r, cfg=MLPF, command=f"{CMD} --jobs {a.jobs} --part {a.part}",
               extra_cfg=dict(family=f, baseline="P5A-import-mlpfusion-v1"))


if __name__ == "__main__":
    main()
