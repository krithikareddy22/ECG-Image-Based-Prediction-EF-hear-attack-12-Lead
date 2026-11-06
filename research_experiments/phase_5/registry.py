"""Phase-5 experiment registry (append-only). Every experiment has a deterministic id, a milestone, the exact command
that (re)creates it, and the run ids it produces (run_id = <exp_id>__s<seed>_r<repeat>). resume5.py derives the status
of every run / experiment / milestone from the files on disk and writes checkpoint.json + PHASE5_STATUS.md.
Never edit or delete an entry that has completed runs; add a new version (-v2) instead."""
from __future__ import annotations

REPEATS, SEEDS = (0, 1, 2), (42, 43, 44)
M = "python -m research_experiments.phase_5."


def runs(exp_id, seeds=(0,), repeats=REPEATS):
    return [f"{exp_id}__s{s}_r{r}" for r in repeats for s in seeds]


MILESTONES = ["M0 audit + foundation", "5A baseline", "5B error analysis", "5C features", "5D-E1 feature MLP", "5D-E2 boosting",
              "5D-E3 stacking", "5D-E4 hierarchical", "5D-E5 decision optimisation", "5D-E6 signal quality",
              "5D-E7 training strategies", "5E statistics", "5F selection + final report"]

# --- 5A: existing predictions re-evaluated (no training). source pattern: {s} seed, {sfx} '' / '_r1' / '_r2'
IMPORTS = {
    "P5A-import-vote-v1": ("phase_3/results/ens_mlp2+ecgflin+hdrclin__vote{sfx}", (0,)),
    "P5A-import-stack-v1": ("phase_3/results/ens_mlp2+ecgflin+hdrclin__stack{sfx}", (0,)),
    "P5A-import-wavg-v1": ("phase_3/results/ens_mlp2+ecgflin+hdrclin__wavg{sfx}", (0,)),
    "P5A-import-mlpfusion-v1": ("phase_2/results/ff_hdrnum+ecgf_mlp{seed}{sfx}", SEEDS),
    "P5A-import-mlp3way-v1": ("phase_3/results/clin_mlpfusion_hdr+clin+ecgf{seed}{sfx}", SEEDS),
    "P5A-import-ecgflin-v1": ("phase_2/results/ecgf_aligned_f6_lin{sfx}", (0,)),
    "P5A-import-ecgflindist-v1": ("phase_2/results/ecgf_f6_lin_cedist1{sfx}", (0,)),
    "P5A-import-hdrlr-v1": ("phase_2/results/hdr_numeric_skLR{sfx}", (0,)),
    "P5A-import-hdrclinlr-v1": ("phase_3/results/clin_hdrnum+clin_skLR{sfx}", (0,)),
    "P5A-import-p4-fusvotemlpreg-v1": ("phase_4/data/private/runs/fus_vote+mlpregA_r{r}", (0,)),
    "P5A-import-p4-regmlpCclin-v1": ("phase_4/data/private/runs/reg_mlp_hdr+ecgf_C_s{s}_r{r}__clin", SEEDS),
    "P5A-import-p4-mtsel-v1": ("phase_4/data/private/runs/mt_sel_s{s}_r{r}", SEEDS),
}
IMPORTS_R0_ONLY = {
    "P5A-import-imgcnn-v1": "phase_2/results/img_resnet18_baseline_recipe",
    "P5A-import-hdrimg-v1": "phase_2/results/R_hdrall+img_wavg",
}

EXPERIMENTS = {}
for eid, (_, seeds) in IMPORTS.items():
    EXPERIMENTS[eid] = dict(milestone="5A baseline", command=M + "baseline.import_existing", runs=runs(eid, seeds))
for eid in IMPORTS_R0_ONLY:
    EXPERIMENTS[eid] = dict(milestone="5A baseline", command=M + "baseline.import_existing", runs=runs(eid, (0,), (0,)))
for obj in ("protected", "accuracy", "floor"):
    eid = f"P5A-calmlp-{obj}-v1"
    EXPERIMENTS[eid] = dict(milestone="5A baseline", command=M + "baseline.calibrated_mlp", runs=runs(eid))
EXPERIMENTS["P5A-baseline-report"] = dict(milestone="5A baseline", command=M + "baseline.report",
                                          artifacts=["results/phase5_baseline_results.csv", "results/phase5_baseline_results.json"])
EXPERIMENTS["P5B-error-analysis"] = dict(milestone="5B error analysis", command=M + "error_analysis.analyse",
                                         artifacts=["results/error_analysis.json"])


def register(eid, milestone, command, run_ids=None, artifacts=None):
    """Later milestones append entries (call at import time of their module or in registry_ext)."""
    if eid in EXPERIMENTS:
        return
    EXPERIMENTS[eid] = dict(milestone=milestone, command=command, **({"runs": run_ids} if run_ids else {}),
                            **({"artifacts": artifacts} if artifacts else {}))


# extensions written by later milestones (append-only file)
try:
    from research_experiments.phase_5 import registry_ext  # noqa: F401
except ImportError:
    pass
