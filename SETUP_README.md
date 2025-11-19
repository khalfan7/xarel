# SurRoL SAC+HER Training Setup Guide

This document describes the setup and modifications required to train SAC+HER on SurRoL tasks (specifically GauzeRetrieveRL-v0).

## Environment Information

- **Python Environment**: `arel` (virtualenv, not conda)
- **Python Version**: 3.12.3
- **Environment Path**: `/home/gak/Documents/edoardo/arel`
- **GPU**: NVIDIA RTX 4060 Laptop (12GB VRAM)
- **CUDA Version**: 11.8
- **PyTorch Version**: 2.7.1+cu118

## Required Packages

### Core Dependencies

```bash
# Activate the virtual environment
source /home/gak/Documents/edoardo/arel/bin/activate

# Install key packages
pip install 'numpy<2'  # CRITICAL: NumPy 2.x is incompatible with some dependencies
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
pip install gym==0.26.2
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118
pip install pybullet==3.2.7
pip install wandb==0.22.3
```

### Full Package List

The complete list of installed packages in the `arel` environment:

```
annotated-types==0.7.0
ansitable==0.11.4
antlr4-python3-runtime==4.9.3
certifi==2025.10.5
cfgv==3.4.0
charset-normalizer==3.4.4
click==8.3.0
cloudpickle==3.1.2
colored==2.3.1
colorlog==6.10.1
contourpy==1.3.3
cycler==0.12.1
distlib==0.4.0
docutils==0.22.3
filelock==3.20.0
filetype==1.2.0
fonttools==4.60.1
fsspec==2025.10.0
gitdb==4.0.12
GitPython==3.1.45
gym==0.26.2
gym-notices==0.1.0
hf-xet==1.2.0
huggingface-hub==0.36.0
hydra-core==1.3.2
identify==2.6.15
idna==3.11
ImageIO==2.37.2
imageio-ffmpeg==0.6.0
Jinja2==3.1.6
joblib==1.5.2
Kivy==2.3.1
Kivy-Garden==0.1.5
kivymd==1.2.0
kiwisolver==1.4.9
MarkupSafe==2.1.5
matplotlib==3.10.7
mpi4py==4.1.1
mpmath==1.3.0
networkx==3.5
nodeenv==1.9.1
numpy==1.26.4  # IMPORTANT: Downgraded from 2.x
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==9.1.0.70
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.21.5
nvidia-nvtx-cu11==11.8.86
omegaconf==2.3.0
opencv-python==4.12.0.88
packaging==25.0
Panda3D==1.10.14
pandas==2.3.3
pgraph-python==0.6.3
pillow==12.0.0
pip==25.3
platformdirs==4.5.0
pre_commit==4.3.0
progress==1.6.1
protobuf==6.33.0
pybullet==3.2.7
pydantic==2.12.4
pydantic_core==2.41.5
Pygments==2.19.2
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.5
roboticstoolbox-python==1.1.1
rtb-data==1.0.1
sacremoses==0.1.1
safetensors==0.6.2
scipy==1.16.3
sentry-sdk==2.43.0
setuptools==80.9.0
six==1.17.0
smmap==5.0.2
spatialgeometry==1.1.0
spatialmath-python==1.1.15
surrol==0.2.0
swift-sim==1.1.0
sympy==1.14.0
termcolor==3.2.0
tokenizers==0.15.0
torch==2.7.1+cu118
torchaudio==2.7.1+cu118
torchvision==0.22.1+cu118
tqdm==4.67.1
transformers==4.36.0
trimesh==4.9.0
triton==3.3.1
typing_extensions==4.15.0
typing-inspection==0.4.2
tzdata==2025.2
urllib3==2.5.0
virtualenv==20.35.4
wandb==0.22.3
websockets==15.0.1
wheel==0.45.1
```

### Install SurRoL Package

```bash
cd /home/gak/Documents/edoardo/SurRoL/Benchmark/state_based
pip install -e .
```

## Code Modifications Required

### 1. Environment Registration (`rl/train_rl.py`)

**File**: `/home/gak/Documents/edoardo/SurRoL/Benchmark/state_based/rl/train_rl.py`

**Change**: Add import to register SurRoL environments

```python
# Add this line after other imports (line 2)
import surrol.gym
```

**Reason**: Environments must be registered with Gym before they can be instantiated via `gym.make()`.

---

### 2. Gym API Compatibility (`surrol/gym/surrol_env.py`)

**File**: `/home/gak/Documents/edoardo/SurRoL/Benchmark/state_based/surrol/gym/surrol_env.py`

#### Change A: Update metadata (line 31)

```python
# OLD:
metadata = {'render.modes': ['human', 'rgb_array', 'img_array']}

# NEW:
metadata = {'render_modes': ['human', 'rgb_array', 'img_array']}
```

**Reason**: Gym 0.26+ uses `'render_modes'` instead of `'render.modes'`.

#### Change B: Update step() return values (lines 128-134)

```python
# OLD (returned 4 values):
return obs, reward, done, info

# NEW (return 5 values):
terminated = done
truncated = False
return obs, reward, terminated, truncated, info
```

**Reason**: Gym 0.26+ API requires 5 return values. The TimeLimit wrapper expects `(obs, reward, terminated, truncated, info)`.
- `terminated`: Episode ended due to task completion/failure
- `truncated`: Episode ended due to time limit (handled by wrapper)

---

### 3. Sampler Compatibility (`rl/modules/samplers.py`)

**File**: `/home/gak/Documents/edoardo/SurRoL/Benchmark/state_based/rl/modules/samplers.py`

**Change**: Update step() unpacking (lines 33-34)

```python
# OLD:
obs, reward, done, info = self._env.step(action)

# NEW:
obs, reward, terminated, truncated, info = self._env.step(action)
done = terminated or truncated
```

**Reason**: Handle the new 5-value return format from environment step().

---

### 4. Observation Dtype Fix (`surrol/tasks/psm_env_RL.py`)

**File**: `/home/gak/Documents/edoardo/SurRoL/Benchmark/state_based/surrol/tasks/psm_env_RL.py`

**Change**: Add dtype conversion in `_get_obs()` (lines 181-186)

```python
# OLD:
obs = {
    'observation': observation.copy(),
    'achieved_goal': achieved_goal.copy(),
    'desired_goal': self.goal.copy()
}

# NEW:
obs = {
    'observation': observation.copy().astype(np.float32),
    'achieved_goal': achieved_goal.copy().astype(np.float32),
    'desired_goal': self.goal.copy().astype(np.float32)
}
```

**Reason**: Eliminate dtype warnings. Observation space expects float32, but internal calculations may produce float64.

---

### 5. Training Configuration (`rl/configs/train.yaml`)

**File**: `/home/gak/Documents/edoardo/SurRoL/Benchmark/state_based/rl/configs/train.yaml`

**Key Changes**:

```yaml
# Agent configuration
agent: sac  # Changed from ddpgbc

# Task configuration
task: GauzeRetrieveRL-v0

# Demo configuration
num_demo: 100  # Changed from 200 to match available demo data
demo_path: /home/gak/Documents/edoardo/gcdt_ckpt/success_demo/data_GauzeRetrieveRL-v0_random_100.npz

# Training parameters
n_train_steps: 1_000_000  # Reduced from 100_000_001 for faster training (~6 hours)
n_eval: 100    # Evaluate every ~10,000 steps (200 episodes)
n_save: 50     # Save checkpoint every ~5,000 steps (100 episodes)
n_log: 1000    # Log metrics every ~1,000 steps (20 episodes)
replay_buffer_capacity: 500_000
batch_size: 512
device: cuda:0

# Logging
project_name: surrol_gauzeretriever_sac
entity_name: edoardo
```

**IMPORTANT**: When changing `n_train_steps`, you MUST proportionally adjust `n_eval`, `n_save`, and `n_log` (see explanation below).

---

## Critical Issues Resolved

### Issue 1: NumPy 2.x Incompatibility
- **Error**: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"
- **Solution**: Downgrade to `numpy==1.26.4`
- **Command**: `pip install 'numpy<2'`

### Issue 2: Gym API Mismatch
- **Error**: `ValueError: not enough values to unpack (expected 5, got 4)`
- **Root Cause**: Gym 0.26 TimeLimit wrapper expects 5 return values, but environment returned 4
- **Solution**: Update `step()` method to return `(obs, reward, terminated, truncated, info)`

### Issue 3: Missing Hydra
- **Error**: `ModuleNotFoundError: No module named 'hydra'`
- **Solution**: Install `hydra-core` and `omegaconf`

### Issue 4: Environment Not Registered
- **Error**: `gym.error.NameNotFound: Environment GauzeRetrieveRL doesn't exist`
- **Solution**: Add `import surrol.gym` to training script

### Issue 5: Demo Count Mismatch
- **Error**: `IndexError: index 100 is out of bounds for axis 0 with size 100`
- **Solution**: Change `num_demo` from 200 to 100 in config

---

## Understanding Confusing Parameters: n_eval, n_save, n_log

### The Confusion


### How They Actually Work

```python
# Step 1: Calculate total episodes
n_train_episodes = n_train_steps / max_timesteps
# For 1M steps: 1,000,000 / 50 = 20,000 episodes

# Step 2: Calculate frequency (in episodes!)
n_eval_episodes = n_train_episodes / n_eval
# If n_eval = 100: 20,000 / 100 = 200 episodes between evaluations

# Step 3: Convert to steps
eval_every_steps = n_eval_episodes * max_timesteps
# 200 episodes * 50 steps = 10,000 steps between evaluations
```
### Critical Rule: Scale Parameters with n_train_steps

When you reduce `n_train_steps`, you **MUST** proportionally reduce these parameters:

```yaml
# Original (100M steps):
n_train_steps: 100_000_001
n_eval: 10000    # → eval every ~200,000 steps
n_save: 5000     # → save every ~100,000 steps  
n_log: 100000    # → log every ~1,000 steps

# Reduced by 100x (1M steps):
n_train_steps: 1_000_000
n_eval: 100      # → eval every ~10,000 steps (divide by 100)
n_save: 50       # → save every ~5,000 steps (divide by 100)
n_log: 1000      # → log every ~1,000 steps (divide by 100)
```

### Why This Matters

**Problem**: If you change only `n_train_steps` without adjusting the others:
- Training will be **extremely slow** (constant evaluation)
- Disk will fill with **thousands of checkpoints**
- You'll see **no training logs** (they trigger too late)

**Solution**: Always adjust all four parameters together proportionally.

### Formula to Calculate Desired Values

```python
# Desired evaluation frequency in steps
desired_eval_steps = 10000

# Calculate n_eval
n_train_episodes = n_train_steps / 50  # 50 steps per episode
n_eval_episodes_desired = desired_eval_steps / 50
n_eval = n_train_episodes / n_eval_episodes_desired

# Example: For 1M steps, evaluate every 10k steps
n_eval = (1_000_000 / 50) / (10_000 / 50) = 20_000 / 200 = 100
```

---

## Running Training

```bash
# Activate environment
source /home/gak/Documents/edoardo/arel/bin/activate

# Navigate to workspace
cd /home/gak/Documents/edoardo/SurRoL/Benchmark/state_based

# Run training
python rl/train_rl.py
```

### Training Output Interpretation

```
| train | F: 2000 | S: 2000 | E: 40 | L: 50 | R: -50.0000 | SR: 0.0000 | BS: 0 | FPS: 45.79 | T: 0:00:41 | ETA: 25 days, 6:35:05
```

- **F**: Frames (environment steps)
- **S**: Total steps
- **E**: Episodes completed
- **L**: Episode length
- **R**: Average reward
- **SR**: Success rate
- **BS**: Batch size (0 = still in seed phase, 512 = active learning)
- **FPS**: Frames per second
- **T**: Time elapsed
- **ETA**: Estimated time to completion

---

## Demo Data

- **Location**: `/home/gak/Documents/edoardo/gcdt_ckpt/success_demo/`
- **File**: `data_GauzeRetrieveRL-v0_random_100.npz`
- **Format**: 100 successful trajectories × 50 steps = 5000 transitions
- **Usage**: 
  - Pre-fills replay buffer at initialization
  - Initializes normalizer statistics
  - Provides training data throughout training
  - NOT used during seed phase (seed phase uses random actions)

---

## Task Details: GauzeRetrieveRL-v0

- **Objective**: Grasp and retrieve gauze to target position
- **Reward**: Sparse (-1 until success, 0 at success)
- **Success Threshold**: 5mm distance from goal
- **Episode Length**: 50 steps
- **Action Space**: 5D (dx, dy, dz, dyaw, gripper)
- **Observation Space**: Dict with 'observation', 'achieved_goal', 'desired_goal'

---
---

## File Structure

```
SurRoL/Benchmark/state_based/
├── rl/
│   ├── train_rl.py          # Training entry point (MODIFIED)
│   ├── configs/
│   │   └── train.yaml       # Training configuration (MODIFIED)
│   ├── modules/
│   │   └── samplers.py      # Rollout sampler (MODIFIED)
│   └── trainers/
│       └── rl_trainer.py
├── surrol/
│   ├── gym/
│   │   ├── surrol_env.py    # Base environment (MODIFIED)
│   │   └── surrol_goalenv.py
│   └── tasks/
│       └── psm_env_RL.py    # RL task base class (MODIFIED)
└── setup.py
```

✅ PyTorch (CUDA): Using NVIDIA RTX 4060 (shown in nvidia-smi)
✅ PyBullet (OpenGL): Using NVIDIA RTX 4060 (shown in glxinfo)

When render=False, PyBullet never allocates those GPU render buffers in the first place.