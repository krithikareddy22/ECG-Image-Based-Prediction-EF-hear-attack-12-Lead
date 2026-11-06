"""Phase-6 experiment registry (append-only). Every experiment has a deterministic id, a milestone, the exact command
that (re)creates it, and the run ids it produces (run_id = <exp_id>__s<seed>_r<repeat>) or the artefacts it writes.
resume6.py derives the status of every run / experiment / milestone from the files on disk.
Never edit or delete an entry that has completed runs; add a new version (-v2) instead. Stage-2 runs are appended
only for candidates that passed the pre-registered Stage-1 screen (PHASE6_PLAN.md §3)."""
from __future__ import annotations

REPEATS, SEEDS = (0, 1, 2), (42, 43, 44)
M = "python -m research_experiments.phase_6."


def runs(exp_id, seeds=(0,), repeats=REPEATS):
    return [f"{exp_id}__s{s}_r{r}" for r in repeats for s in seeds]


MILESTONES = ["M0 setup + pre-registration", "M1 timing", "A stage 1 (fine-tuning screen)", "B stage 1 (encoder screen)",
              "D input fidelity", "Stage 2 (full evaluation)", "E diverse ensembling", "N name-pair sensitivity",
              "F statistics + final report"]

EXPERIMENTS: dict = {}


def register(eid, milestone, command, run_ids=None, artifacts=None):
    if eid in EXPERIMENTS:
        return
    EXPERIMENTS[eid] = dict(milestone=milestone, command=command, **({"runs": run_ids} if run_ids else {}),
                            **({"artifacts": artifacts} if artifacts else {}))


# ---------------- M0: environment restore (git-ignored inputs, regenerated baselines) and the pre-registration
register("P6-restore-inputs", "M0 setup + pre-registration", M + "features.restore_inputs --mirror-phase5",
         artifacts=["results/restore_inputs.json"])
register("P6-baseline-regeneration", "M0 setup + pre-registration", M + "baselines",
         artifacts=["results/baseline_regeneration.json"])
register("P6-plan", "M0 setup + pre-registration", "(written by hand; sha256 in configs/PLAN.sha256)",
         artifacts=["PHASE6_PLAN.md", "configs/PLAN.sha256"])
# ---------------- M1: timing of experiment A (forward / backward cost only; no metric)
register("P6-timing-A", "M1 timing", M + "experiments.timing_a", artifacts=["results/timing_A.json"])
# ---------------- A stage 1: reference (frozen) and fine-tuning screens, seed 42 x repeats 0-2
register("P6R0-joint-frozen-v1", "A stage 1 (fine-tuning screen)", M + "experiments.finetune --exp R0",
         runs("P6R0-joint-frozen-v1", SEEDS, REPEATS))
register("P6A1-ft-s6-joint-v1", "A stage 1 (fine-tuning screen)", M + "experiments.finetune --exp A1",
         runs("P6A1-ft-s6-joint-v1", (42,), REPEATS))
register("P6A2-ft-s56-joint-v1", "A stage 1 (fine-tuning screen)", M + "experiments.finetune --exp A2",
         runs("P6A2-ft-s56-joint-v1", (42,), REPEATS))

# extensions written by later milestones (append-only)
try:
    from research_experiments.phase_6 import registry6_ext  # noqa: F401
except ImportError:
    pass
