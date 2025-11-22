# xarel

SAC+HER training for SurRoL surgical robotics tasks with demonstration data.

## Task Overview

SurRoL provides surgical robotics manipulation tasks with sparse rewards and goal-based learning. Tasks like **GauzeRetrieveRL-v0** require complex multi-stage manipulation:

### Task Complexity Example: Gauze Retrieval
The agent must learn a 5-waypoint sequential manipulation task:
1. **Navigate to object** (~10 steps) - Move gripper to gauze location
2. **Precisely align** (~5 steps) - Position gripper above object
3. **Grasp timing** - Close gripper at the right moment for stable grasp
4. **Lift object** (~10 steps) - Elevate gauze without dropping
5. **Transport to goal** (~15 steps) - Move to target location
6. **Precision requirement**: Final position must be within **0.005m (5mm)** of goal

### Training Expectations
- **Success threshold**: 0.005m (5mm precision)
- **Initial distance**: ~0.25m (50x the threshold)
- **Episode length**: 50 steps (tight constraint for 5-stage task)
- **Expected timeline**: 
  - 0-200k steps: Pure exploration, success_rate = 0%
  - 200k-500k steps: First successes appear (HER learning kicks in)
  - 500k-1M steps: Success rate climbs to 20-60%

**Note**: Success rate of 0% before 200k-300k steps is **completely normal** for sparse reward manipulation tasks. The agent is learning foundational skills during this period.

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
   # Long multi-GPU training (recommended: 1M steps minimum)
   sbatch hpc/train_long.sbatch
   ```

Override task/seed via environment variables: `TASK=PegTransferRL-v0 sbatch hpc/train_long.sbatch`

## Training Diagnostics

Monitor training progress with the diagnostic tool:

```bash
cd sb3
source /apps/local/conda_init.sh
conda activate /l/users/${USER}/envs/arel
python diagnose_training.py
```

This analyzes:
- Task difficulty (initial distance vs success threshold)
- Random policy baseline performance
- Current training progress (actor/critic losses, entropy)
- Resource utilization predictions

### Key Metrics to Watch
- **Actor loss**: Should decrease over time (policy improving)
- **Goal distance**: Should trend downward even if success_rate=0
- **Entropy coefficient**: Controls exploration (0.1-1.0 is healthy early on)
- **Buffer size**: Should fill up to capacity over first 100k-200k steps