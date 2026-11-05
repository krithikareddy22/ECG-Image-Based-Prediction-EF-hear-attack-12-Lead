"""5C feature-family comparison: multinomial LR (class-balanced, C chosen on inner-val; the recipe of the header-only
baseline) on each family alone and on header numeric + family; repeats 0-2, identical folds.
References: header numeric LR (P5A-import-hdrlr-v1), header + phase-3 clinical LR (P5A-import-hdrclinlr-v1), ECGF linear.
    python -m research_experiments.phase_5.features.compare            # run missing runs (skips completed)
    python -m research_experiments.phase_5.features.compare --table    # write results/feature_comparison.csv
"""
import json
import sys

from research_experiments.phase_5.experiments.runner import cv_run
from research_experiments.phase_5.p5common import RES5
from research_experiments.phase_5.registry_ext import FEATURE_SETS
from research_experiments.phase_5.validation.stats import family_table

CMD = "python -m research_experiments.phase_5.features.compare"


def run_all():
    for name, blocks in FEATURE_SETS.items():
        for r in (0, 1, 2):
            cv_run(f"P5C-lr-{name}-v1", blocks, "sklr", 0, r, command=CMD, extra_cfg=dict(feature_set=name))


def table():
    alone = [f"P5C-lr-{n}-v1" for n in FEATURE_SETS if not n.startswith("hdr")]
    added = [f"P5C-lr-{n}-v1" for n in FEATURE_SETS if n.startswith("hdr")]
    A = family_table(alone + ["P5A-import-ecgflin-v1", "P5A-import-hdrclinlr-v1"], "P5A-import-hdrlr-v1")
    B = family_table(added, "P5A-import-hdrlr-v1")
    A["comparison"] = "family alone vs header LR"; B["comparison"] = "header + family vs header LR"
    import pandas as pd
    T = pd.concat([B, A]); T.to_csv(RES5 / "feature_comparison.csv", index=False, float_format="%.4f")
    (RES5 / "feature_comparison.json").write_text(json.dumps(T.to_dict("records"), indent=1, default=float))
    pd.set_option("display.width", 250)
    print(T[["model", "comparison", "accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "min_class_recall_mean",
             "d_accuracy", "p_accuracy", "p_holm_accuracy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    table() if "--table" in sys.argv else run_all()
