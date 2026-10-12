#!/bin/bash
# Run INSIGHT over a cohort with a Slurm array job: each task takes every n-th patient (patients never split).
# Submit with:  sbatch --array=0-7 examples/slurm_array.sh
#SBATCH --job-name=insight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/insight_%A_%a.out

set -euo pipefail
N=${SLURM_ARRAY_TASK_COUNT:-8}
insight run \
  --slides /path/to/slides \
  --patient-regex '^(TCGA-\w\w-\w{4})' \
  --output /path/to/insight_runs --experiment my_cohort --run 1 \
  --device cuda --batch-size 128 --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --shard "${SLURM_ARRAY_TASK_ID}/${N}"

# After all tasks finish, collect the report and predictions once (fast; everything is cached):
#   insight run --slides /path/to/slides --output /path/to/insight_runs --experiment my_cohort --run 1 --stages predict
