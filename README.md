# xarel

SAC+HER training for SurRoL surgical robotics tasks with demonstration data.

## Structure

```
xarel/
├── SurRoL/
│   ├── rl/                    # RL training code
│   │   ├── train_rl.py        # Main training script
│   │   ├── configs/           # Hydra config files (train.yaml, eval.yaml)
│   │   ├── agents/            # RL algorithms (SAC, DDPG, DDPG+BC, DEX)
│   │   ├── trainers/          # RLTrainer and base trainer classes
│   │   ├── modules/           # Replay buffer, HER, samplers
│   │   ├── components/        # Logger, checkpointer, WandB integration
│   │   ├── utils/             # RL utilities, MPI helpers
│   │   └── requirements.txt   # Python dependencies
│   ├── surrol/                # SurRoL environment package
│   └── setup.py               # Package installation
├── success_demo/              # Demo datasets (100 trajectories each)
│   ├── data_GauzeRetrieveRL-v0_random_100.npz
│   ├── data_NeedlePickRL-v0_random_100.npz
│   ├── data_PegTransferRL-v0_random_100.npz
│   └── ... (7 more task demos)
├── hpc/                       # Cluster job scripts
│   ├── train_ciai_long.sbatch # Long training job
│   └── debug_run.sh           # Interactive testing
├── README.md                  # This file
└── SETUP_README.md            # Detailed setup guide
```

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
