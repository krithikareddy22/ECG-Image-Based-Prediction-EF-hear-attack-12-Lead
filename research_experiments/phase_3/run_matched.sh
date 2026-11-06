#!/usr/bin/env bash
# Matched seeds x repeats for the 3-way vs 2-way comparison (the 2-way runs come from phase 2 with identical
# folds / seeds / config) and the header vs header+clinical linear comparison. 1 thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
E="python -m research_experiments.phase_3.exp_p3 --group clinical"
$E --only clin_mlpfusion_hdr+clin+ecgf --seeds 43 44
for r in 1 2; do
  $E --repeat $r --only clin_mlpfusion_hdr+clin+ecgf --seeds 42 43 44
  $E --repeat $r --only clin_hdrnum+clin_skLR --seeds 42
done
