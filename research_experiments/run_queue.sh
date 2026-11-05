#!/usr/bin/env bash
# Training queue for Experiments A2 / B (CPU, sequential). Run from the repository root:
#   bash research_experiments/run_queue.sh
set -u
L=research_experiments/logs
run() { tag=$1; shift; echo "=== $(date +%T) start $tag"; python -m research_experiments.train_deep "$@" --tag "$tag" > "$L/$tag.log" 2>&1; echo "=== $(date +%T) done $tag (exit $?)"; }
for s in 42 43 44; do
  run A2_ecgf_ft56_ce_seed$s      --arch ecgfounder --head ce      --unfreeze 5,6 --seed $s
  run A2_ecgf_ft56_cumlink_seed$s --arch ecgfounder --head cumlink --unfreeze 5,6 --seed $s
done
run A2_ecgf_ft6_ce_seed42 --arch ecgfounder --head ce --unfreeze 6 --seed 42
run B_resnet18_image_cumlink_seed42 --arch resnet18 --head cumlink --seed 42
echo "=== $(date +%T) start R0 retrain"; python -m research_experiments.reproduce_resnet18 --seed 42 > $L/R0_resnet18_filtered_retrain_seed42.log 2>&1; echo "=== $(date +%T) done R0 (exit $?)"
