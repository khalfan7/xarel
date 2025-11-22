# Output Directories Reference

## Directory Structure

```
xarel/
├── logs/                          # SLURM job logs (stdout/stderr)
│   ├── surrol-sb3-{jobid}.out
│   └── surrol-sb3-{jobid}.err
│
├── sb3/
│   ├── results/                   # Main training results (NEW)
│   │   └── {task}/
│   │       └── seed_{N}/
│   │           ├── tensorboard/   # TensorBoard logs
│   │           ├── checkpoints/   # Periodic model saves
│   │           ├── best_model/    # Best performing model
│   │           ├── eval_logs/     # Evaluation metrics
│   │           ├── final_model.zip
│   │           └── vec_normalize.pkl
│   │
│   ├── hydra_outputs/             # Hydra config management (NEW)
│   │   └── YYYY-MM-DD/
│   │       └── HH-MM-SS/
│   │           └── .hydra/
│   │               ├── config.yaml
│   │               ├── hydra.yaml
│   │               └── overrides.yaml
│   │
│   ├── exp_sb3/                   # OLD location (deprecated)
│   └── outputs/                   # OLD Hydra location (deprecated)
```

## Output Locations Explained

### 1. **SLURM Logs** (`logs/`)
- **What**: stdout and stderr from SLURM jobs
- **Created by**: SLURM scheduler (via `#SBATCH --output/--error` directives)
- **When**: Every job submission
- **Size**: Small (text logs)
- **Keep or Delete**: Can delete old logs after reviewing
- **Location in code**: `sb3/hpc/train_sb3.sbatch` lines 11-12

### 2. **Training Results** (`sb3/results/`)
- **What**: All training outputs (models, metrics, logs)
- **Created by**: `train_sb3.py` line 84
- **When**: Training starts
- **Size**: Large (models can be 10-100MB+)
- **Keep or Delete**: 
  - Keep: `best_model/`, `final_model.zip` (for inference)
  - Delete: Old `checkpoints/`, old `tensorboard/`
- **Location in code**: `train_sb3.py` line 84, config `train_sb3.yaml` line 57

**Subdirectories:**
- `tensorboard/` - Training metrics for visualization
- `checkpoints/` - Periodic saves every 20k steps
- `best_model/` - Best model based on evaluation
- `eval_logs/` - Evaluation episode returns
- `final_model.zip` - Model after full training
- `vec_normalize.pkl` - Observation normalization stats

### 3. **Hydra Outputs** (`sb3/hydra_outputs/`)
- **What**: Hydra configuration management logs
- **Created by**: Hydra framework automatically
- **When**: Every `python train_sb3.py` run
- **Size**: Tiny (KB)
- **Keep or Delete**: Can delete (just config snapshots)
- **Location in code**: `train_sb3.yaml` lines 58-60

### 4. **Deprecated Directories**
- `sb3/exp_sb3/` - Old training results location
- `sb3/outputs/` - Old Hydra location
- Can be safely deleted

## Cleanup Commands

### Delete all generated outputs:
```bash
cd /l/users/khalfan.hableel/xarel
rm -rf logs/
rm -rf sb3/results/
rm -rf sb3/exp_sb3/
rm -rf sb3/hydra_outputs/
rm -rf sb3/outputs/
```

### Keep only the best models:
```bash
cd /l/users/khalfan.hableel/xarel/sb3/results
# For each task/seed, delete checkpoints and tensorboard but keep best_model
find . -type d -name "checkpoints" -exec rm -rf {} + 2>/dev/null
find . -type d -name "tensorboard" -exec rm -rf {} + 2>/dev/null
# Keep best_model/ and final_model.zip
```

### Delete old SLURM logs (keep last 10):
```bash
cd /l/users/khalfan.hableel/xarel/logs
ls -t | tail -n +21 | xargs rm -f  # Keep last 10 jobs (out+err = 20 files)
```

## What's in .gitignore

The following are excluded from git (won't be committed):
- `logs/` - SLURM logs
- `sb3/results/` - Training outputs
- `sb3/exp_sb3/` - Old outputs
- `sb3/hydra_outputs/` - Hydra logs
- `sb3/outputs/` - Old Hydra logs
- `*.zip`, `*.pkl`, `*.npz`, `*.pth` - Model files
- `events.out.tfevents.*` - TensorBoard logs

## Disk Space Estimates

Per training run (1M steps):
- TensorBoard logs: ~50-200MB
- Checkpoints (50 saves): ~500MB-2GB
- Best model: ~10-50MB
- Final model: ~10-50MB
- Replay buffer (if saved): ~5-20GB
- SLURM logs: ~1-10MB

**Total per run: ~0.5-3GB** (without replay buffer)
**With 10 parallel runs: ~5-30GB**
