"""Append-only registry extensions for milestones 5C onward (imported by registry.py). Never remove entries."""
from research_experiments.phase_5.registry import M, REPEATS, SEEDS, register, runs

# ---------------- 5C feature families (LR, deterministic; 3 repeats)
NEW_FAMILIES = ("qrs", "st", "hrv", "freq", "wavelet", "leadstat", "sq")
FEATURE_SETS = {**{f: [f"feat:{f}"] for f in NEW_FAMILIES + ("clin3",)},
                **{f"hdr+{f}": ["hdr", f"feat:{f}"] for f in NEW_FAMILIES},
                "allnew": ["feat:" + "+".join(NEW_FAMILIES)], "hdr+allnew": ["hdr", "feat:" + "+".join(NEW_FAMILIES)],
                "hdr+clin3+allnew": ["hdr", "feat:clin3+" + "+".join(NEW_FAMILIES)]}
register("P5C-features-extract", "5C features", M + "features.extract",
         artifacts=["results/feature_validation.json"])
for name in FEATURE_SETS:
    register(f"P5C-lr-{name}-v1", "5C features", M + "features.compare", runs(f"P5C-lr-{name}-v1", (0,), REPEATS))
register("P5C-feature-comparison", "5C features", M + "features.compare --table", artifacts=["results/feature_comparison.csv"])
_ = SEEDS

# ---------------- E1 feature-enhanced MLP fusion (header || ECGF || family), 3 seeds x 3 repeats
E1_FAMILIES = NEW_FAMILIES + ("allnew",)
for f in E1_FAMILIES:
    register(f"P5D-E1-mlp-{f}-v1", "5D-E1 feature MLP", M + "experiments.e1_feature_mlp", runs(f"P5D-E1-mlp-{f}-v1", SEEDS, REPEATS))
register("P5D-E1-table", "5D-E1 feature MLP", M + "experiments.e1_feature_mlp --table", artifacts=["results/e1_feature_mlp.csv"])

# ---------------- E2 gradient boosting (seed 42, 3 repeats)
E2_SETS = {"eng": ["hdr", "feat:clin3+" + "+".join(NEW_FAMILIES)], "compact": ["hdr", "feat:clin3+qrs"],
           "eng+ecgfpca": ["hdr", "feat:clin3+" + "+".join(NEW_FAMILIES), "ecgfpca:32"]}
for mdl in ("xgb", "lgbm"):
    for st in E2_SETS:
        register(f"P5D-E2-{mdl}-{st}-v1", "5D-E2 boosting", M + "experiments.e2_boosting", runs(f"P5D-E2-{mdl}-{st}-v1", (42,), REPEATS))
# ---------------- E3 nested stacking (deterministic, 3 repeats)
for v in ("lr-balanced", "lr-unweighted", "wavg-acc"):
    register(f"P5D-E3-stack-{v}-v1", "5D-E3 stacking", M + "experiments.e3_stacking", runs(f"P5D-E3-stack-{v}-v1", (0,), REPEATS))
# ---------------- E4 hierarchical (3 seeds x 3 repeats)
register("P5D-E4-hier-v1", "5D-E4 hierarchical", M + "experiments.e4_hierarchical", runs("P5D-E4-hier-v1", SEEDS, REPEATS))
# ---------------- E5 nested decision optimisation (no training)
for cand in ("vote", "mlpfusion"):
    for rule in ("prior", "bias"):
        for obj in ("accuracy", "floor", "protected"):
            if cand == "mlpfusion" and rule == "prior":
                continue                                 # = P5A-calmlp-* (done in 5A)
            register(f"P5D-E5-{cand}-{rule}-{obj}-v1", "5D-E5 decision optimisation", M + "experiments.e5_decision",
                     runs(f"P5D-E5-{cand}-{rule}-{obj}-v1", (0,), REPEATS))
# ---------------- E6 signal-quality-aware
register("P5D-E6-qfilter-v1", "5D-E6 signal quality", M + "experiments.e6_quality", runs("P5D-E6-qfilter-v1", SEEDS, REPEATS))
register("P5D-E6-qweight-v1", "5D-E6 signal quality", M + "experiments.e6_quality", runs("P5D-E6-qweight-v1", SEEDS, REPEATS))
register("P5D-E6-qgate-v1", "5D-E6 signal quality", M + "experiments.e6_quality", runs("P5D-E6-qgate-v1", (0,), REPEATS))
# ---------------- E7 training strategies (MLP fusion; one change at a time)
E7_VARIANTS = {"focal": dict(loss="focal", gamma=2.0), "smooth": dict(smooth=0.1), "mixup": dict(mixup=0.2),
               "cosine": dict(sched="cosine"), "selacc": dict(select="accuracy")}
for v in E7_VARIANTS:
    register(f"P5D-E7-{v}-v1", "5D-E7 training strategies", M + "experiments.e7_training", runs(f"P5D-E7-{v}-v1", SEEDS, REPEATS))
E7_PHASE2 = {"unweighted": "ff_mlp_unweighted", "balsampler": "ff_mlp_balsampler", "do0.3": "ff_mlp_do0.3", "do0.7": "ff_mlp_do0.7",
             "h32": "ff_mlp_h32", "h128": "ff_mlp_h128"}
for v in E7_PHASE2:
    register(f"P5D-E7-p2-{v}-v1", "5D-E7 training strategies", M + "experiments.e7_training --import-phase2",
             runs(f"P5D-E7-p2-{v}-v1", SEEDS, (0,)))
register("P5D-E7-augment-prep", "5D-E7 training strategies", M + "features.ecgf_augment",
         artifacts=["results/ecgf_augmentation_validation.json"])
register("P5D-E7-augment-v1", "5D-E7 training strategies", M + "experiments.e7_training --augment",
         runs("P5D-E7-augment-v1", SEEDS, REPEATS))
register("P5E-final-statistics", "5E statistics", M + "validation.final_stats", artifacts=["results/phase5_all_models.csv"])
# E7 extension: class-unweighted CE (phase-2 run existed for repeat 0 only) evaluated on all three repeats:
# repeat 0 re-uses the completed phase-2 runs (copied, not retrained); repeats 1-2 are new runs (mlp5 weight='none').
register("P5D-E7-unweighted-v1", "5D-E7 training strategies", M + "experiments.e7_training --unweighted",
         runs("P5D-E7-unweighted-v1", SEEDS, REPEATS))
register("P5F-selection-and-final-report", "5F selection + final report", M + "validation.final_stats",
         artifacts=["results/phase5_selection.json", "reports/PHASE5_FINAL_REPORT.md", "PHASE5_HANDOFF.md"])
