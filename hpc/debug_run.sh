#!/bin/bash
# Interactive helper meant to be launched inside an srun allocation on cscc-gpu-p.
# Recommended general queue (longer interactive sessions):
#   srun -p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:1 --cpus-per-task=8 \
#        --time=02:00:00 --pty bash hpc/debug_run.sh
# Debug queue for short tests (≤3h, ≤4 GPUs total per user):
#   srun -p cscc-gpu-p -q gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 \
#        --time=00:30:00 --pty bash hpc/debug_run.sh

set -euo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
  PROJECT_ROOT="/l/users/${USER}/xarel"
fi
ENV_PATH="/l/users/${USER}/envs/surrol-rl"

source /apps/local/conda_init.sh
conda activate "${ENV_PATH}"

cd "${PROJECT_ROOT}/SurRoL/rl"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python train_rl.py \
