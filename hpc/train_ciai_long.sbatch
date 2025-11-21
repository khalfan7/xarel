#!/bin/bash
#SBATCH --job-name=surrol-train
#SBATCH -p cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/l/users/%u/xarel/logs/%x-%j.out
#SBATCH --error=/l/users/%u/xarel/logs/%x-%j.err

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load openmpi4.1.2rc2
source /apps/local/conda_init.sh
conda activate "/l/users/${USER}/envs/arel"

PROJECT_ROOT="/l/users/${USER}/xarel"
cd "${PROJECT_ROOT}/SurRoL/rl"
mkdir -p "${PROJECT_ROOT}/logs"

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128
export NUMEXPR_NUM_THREADS=128
export HYDRA_FULL_ERROR=1

TASK_OVERRIDE=${TASK:-GauzeRetrieveRL-v0}
SEED_OVERRIDE=${SEED:-1}

python train_rl.py \
  task=${TASK_OVERRIDE} \
  seed=${SEED_OVERRIDE}
