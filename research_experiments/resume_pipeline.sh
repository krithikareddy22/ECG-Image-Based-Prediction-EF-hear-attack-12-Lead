#!/usr/bin/env bash
# Idempotent resume of the remaining Phase-2/3 pipeline (finished steps are skipped automatically).
# Run from the repository root:  bash research_experiments/resume_pipeline.sh
# Locked test is never scored here. Nothing is committed or pushed.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
L=research_experiments/phase_3/logs; mkdir -p $L
step() { echo "=== $(date +%T) $*"; }
# guard: never start a second copy of a job that is still running
if ps -eo args | grep -E "[p]ython -m research_experiments\.(phase_2|phase_3)\." >/dev/null; then
  echo "research jobs are still running (ps -eo pid,args | grep research_experiments); wait for them, then re-run."; exit 1
fi

# lane A (3 threads): image-CNN CV (skips completed folds)
step "image CNN CV"; OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -m research_experiments.phase_2.cv_image_cnn --threads 3 >> $L/resume_cnn.log 2>&1 &
CNN=$!

# lane B (1 thread): everything that does not need the CNN
step "phase-2 robust seeds"; M="python -m research_experiments.phase_2.cv_features"
$M --group mlp_hp_seeds; $M --group robust_r0_extra; $M --group robust_seeds --repeat 1; $M --group robust_seeds --repeat 2
step "phase-3 clinical/gated (seed 42)"; python -m research_experiments.phase_3.exp_p3 --group clinical gated
step "phase-3 matched 3-way vs 2-way"; bash research_experiments/phase_3/run_matched.sh
step "phase-3 gated-dist seeds x repeats"
E="python -m research_experiments.phase_3.exp_p3 --group gated --only gate_scalar_hdr+clin+ecgf_dist"
$E --seeds 43 44; $E --repeat 1 --seeds 42 43 44; $E --repeat 2 --seeds 42 43 44
step "phase-3 masking"; python -m research_experiments.phase_3.masking_audit && python -m research_experiments.phase_3.masking_cv
step "nested calibration"; for r in 0 1 2; do
  [ -f research_experiments/phase_2/results/nested_calibration_fusion_mlp_r$r.json ] || \
    python -m research_experiments.phase_2.nested_calibration --model fusion_mlp --repeat $r; done

# after the CNN: header + image fusions (incl. reference R), robust tables
step "waiting for CNN"; wait $CNN
step "fusion combos"; python -m research_experiments.phase_2.run_fusion --stage combos
step "ensembles"; python -m research_experiments.phase_3.p3_ensembles
step "nested calibration (vote)"; for r in 0 1 2; do [ -f research_experiments/phase_3/results/nested_calibration_vote_r$r.json ] || python -m research_experiments.phase_3.nested_calib_vote --repeat $r; done
step "robust tables"; python -m research_experiments.phase_3.p3_robust
step "calibration summary + confusion matrices"; python -m research_experiments.phase_3.calibration_summary; python -m research_experiments.phase_3.confusion_collection
step "status"; python -m research_experiments.status
