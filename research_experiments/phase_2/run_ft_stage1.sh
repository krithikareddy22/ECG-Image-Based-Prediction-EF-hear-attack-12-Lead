#!/usr/bin/env bash
# ECGFounder partial fine-tuning, staged search, stage 1 (last block; learning rates). 1 thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
F="python -m research_experiments.phase_2.cv_finetune --threads 1 --depth block3"
$F --tag ft_b3_base
$F --tag ft_b3_lrb3e-4 --lr-b 3e-4
$F --tag ft_b3_lrb3e-5 --lr-b 3e-5
