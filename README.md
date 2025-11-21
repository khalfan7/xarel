# xarel

SAC+HER training for SurRoL surgical robotics tasks with demonstration data.

## Structure

- `SurRoL/rl/` – state-based RL training code (SAC, DDPG, HER buffer)
- `gcdt_ckpt/success_demo/` – demonstration datasets (.npz files, 100 trajectories each)
- `hpc/` – cluster job scripts for distributed training
- `SETUP_README.md` – detailed setup and troubleshooting guide

## Quick Start

```bash
# Install dependencies (Python 3.10-3.12, venv recommended)
cd SurRoL/rl
pip install -r requirements.txt
cd ../Benchmark/state_based
pip install -e .

# Run training
cd rl
python train_rl.py
```

Edit `configs/train.yaml` to change task, demo path, or hyperparameters.

## HPC Cluster Usage

The `hpc/` directory contains SLURM scripts for distributed training:

1. **Setup environment** (modify paths in scripts for your cluster):
   ```bash
   # Create conda environment
   conda create -n arel python=3.10 -y
   conda activate arel
   pip install -r SurRoL/rl/requirements.txt
   ```

2. **Interactive testing**:
   ```bash
   # Short debug job to verify setup
   srun -p <gpu-partition> --gres=gpu:1 --cpus-per-task=8 \
        --time=00:30:00 --pty bash hpc/debug_run.sh
   ```

3. **Submit training job**:
   ```bash
   # Long multi-GPU training
   sbatch hpc/train_long.sbatch
   ```

Override task/seed via environment variables: `TASK=PegTransferRL-v0 sbatch hpc/train_long.sbatch`

## Key Features

- **Algorithms**: SAC, DDPG, DDPG+BC, DEX (all with HER)
- **Tasks**: 10 surgical manipulation tasks (NeedlePick, GauzeRetrieve, PegTransfer, etc.)
- **Demo integration**: Pre-fills replay buffer and initializes normalizers
- **Sparse rewards**: HER enables learning from failure trajectories

See `SETUP_README.md` for Gym API fixes, dependency troubleshooting, and config parameter explanations.
