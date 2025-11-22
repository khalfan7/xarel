# xarel

SAC+HER training for SurRoL surgical robotics tasks with Stable-Baselines3.

## Overview

SurRoL provides surgical robotics manipulation tasks with sparse rewards and goal-based learning. This implementation uses:
- **SAC** (Soft Actor-Critic) for continuous control
- **HER** (Hindsight Experience Replay) for sparse reward learning
- **Optimized hyperparameters** based on SB3 best practices
- **Custom TimeLimit wrapper** to maintain compute_reward compatibility

### Example: Gauze Retrieval Task
A 5-waypoint sequential manipulation task:
1. Navigate to gauze location
2. Precisely align gripper above object
3. Execute timed grasp for stable grip
4. Lift object without dropping
5. Transport to goal (0.005m precision required)

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n arel python=3.12 -y
conda activate arel

# Install SurRoL environment
cd SurRoL
pip install -e .
cd ..

# Install training dependencies
pip install stable-baselines3[extra]
pip install hydra-core
pip install tensorboard
```

### 2. Train a Model

```bash
cd sb3

# Basic training (uses train_sb3.yaml config)
python train_sb3.py

# Different task
python train_sb3.py task=NeedlePickRL-v0

# Different seed
python train_sb3.py seed=123

# Adjust training duration
python train_sb3.py total_timesteps=2_000_000
```

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir sb3/outputs

# View training curves at http://localhost:6006
```

## Configuration

All training parameters are in `sb3/configs/train_sb3.yaml`. Key optimized settings:

### Core Parameters

```yaml
# Task and resources
task: GauzeRetrieveRL-v0
n_envs: 8
device: cuda

# Training
total_timesteps: 1_000_000
learning_starts: 10_000
batch_size: 256
buffer_size: 1_500_000

# SAC hyperparameters (optimized for sparse rewards)
learning_rate: 3e-4
gamma: 0.95              # Lower for sparse rewards
tau: 0.005               # Stable target updates
gradient_steps: -1       # Sample efficient

# Policy network
policy_kwargs:
  net_arch: [256, 256]   # Balanced capacity

# HER configuration
her:
  strategy: 'future'     # Best for most tasks
  n_sampled_goal: 6      # Virtual transitions per real one
  handle_timeout_termination: true  # Critical for time limits

# Normalization (critical for RL)
normalize: true
norm_obs: true
norm_reward: false       # Never normalize sparse rewards!
clip_obs: 200.0

## HPC Cluster Usage

### Submit Training Job

```bash
# Submit to SLURM
sbatch sb3/hpc/train_sb3.sbatch

# Check job status
squeue -u $USER

# Monitor training
tail -f logs/surrol-sb3-<JOBID>.out

# Check for errors
tail -f logs/surrol-sb3-<JOBID>.err
```

### SLURM Configuration

Edit `sb3/hpc/train_sb3.sbatch` to adjust resources:

```bash
#SBATCH --gres=gpu:1        # Number of GPUs
#SBATCH --cpus-per-task=8   # CPU cores
#SBATCH --mem=32G           # Memory
#SBATCH --time=24:00:00     # Time limit
```

## Implementation Details

### Custom TimeLimit Wrapper

SB3's `check_env()` requires goal-based environments to expose `compute_reward()`. Gymnasium's TimeLimit wrapper doesn't forward this method, so we implemented a custom wrapper:

```python
class TimeLimitWithComputeReward(gymnasium.Wrapper):
    """TimeLimit wrapper that forwards compute_reward for HER compatibility."""
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Recursively unwrap to find the base environment
        env = self.env
        while hasattr(env, 'env') and not hasattr(env, 'compute_reward'):
            env = env.env
        return env.compute_reward(achieved_goal, desired_goal, info)
```

This wrapper is automatically applied in `make_env()` to replace Gymnasium's default TimeLimit.

## File Structure

```
xarel/
├── README.md                          # This file
├── sb3/
│   ├── train_sb3.py                   # Main training script (with custom wrapper)
│   ├── configs/
│   │   └── train_sb3.yaml             # Optimized configuration
│   ├── hpc/
│   │   └── train_sb3.sbatch           # SLURM job script
│   └── outputs/                       # Training outputs
│       └── <task>/seed_<N>/
│           ├── tensorboard/           # TensorBoard logs
│           ├── checkpoints/           # Periodic saves (every 50k steps)
│           ├── best_model/            # Best evaluation model
│           ├── eval_logs/             # Evaluation metrics
│           ├── final_model.zip        # Complete model
│           └── final_policy.pth       # Policy-only (inference)
├── SurRoL/                            # SurRoL surgical robotics
│   └── surrol/
│       ├── tasks/                     # Task implementations
│       ├── robots/                    # PSM robot models
│       └── gym/                       # Gymnasium wrappers
└── logs/                              # SLURM logs
    ├── surrol-sb3-<JOBID>.out
    └── surrol-sb3-<JOBID>.err
```