# SurRoL SAC+HER Training Setup

Setup guide for training SAC+HER on SurRoL surgical robotics tasks.

## Environment Requirements

- **Python**: 3.10-3.12
- **CUDA**: 11.8+ (for GPU training)
- **Key Packages**:
  ```bash
  pip install 'numpy<2'           # Critical: NumPy 2.x incompatible
  pip install hydra-core==1.3.2
  pip install gym==0.26.2
  pip install torch torchvision torchaudio
  pip install pybullet wandb
  ```

## Installation

```bash
# Create virtual environment
python -m venv arel
source arel/bin/activate  # Linux/Mac
# or: arel\Scripts\activate  # Windows

# Install dependencies
cd SurRoL/rl
pip install -r requirements.txt

# Install SurRoL package
cd ../Benchmark/state_based
pip install -e .
```

## Required Code Modifications

### 1. Environment Registration (`rl/train_rl.py`)
Add import to register environments:
```python
import surrol.gym
```

### 2. Gym 0.26 API Compatibility

#### `surrol/gym/surrol_env.py` (line 31):
```python
# OLD: metadata = {'render.modes': [...]}
metadata = {'render_modes': ['human', 'rgb_array', 'img_array']}
```

#### `surrol/gym/surrol_env.py` (lines 128-134):
```python
# OLD: return obs, reward, done, info
terminated = done
truncated = False
return obs, reward, terminated, truncated, info
```

### 3. Sampler Update (`rl/modules/samplers.py`, lines 33-34):
```python
# OLD: obs, reward, done, info = self._env.step(action)
obs, reward, terminated, truncated, info = self._env.step(action)
done = terminated or truncated
```

### 4. Observation Dtype (`surrol/tasks/psm_env_RL.py`, lines 181-186):
```python
obs = {
    'observation': observation.copy().astype(np.float32),
    'achieved_goal': achieved_goal.copy().astype(np.float32),
    'desired_goal': self.goal.copy().astype(np.float32)
}
```

## Configuration (`rl/configs/train.yaml`)

Key parameters:
```yaml
agent: sac                    # Algorithm: sac, ddpg, ddpgbc, dex
task: GauzeRetrieveRL-v0     # Task name
num_demo: 100                # Number of demo trajectories
demo_path: /path/to/data_GauzeRetrieveRL-v0_random_100.npz
use_wb: False                # Enable WandB logging

n_train_steps: 1_000_000     # Total training steps
n_eval: 100                  # Eval frequency (episodes between evals)
n_save: 50                   # Save frequency
n_log: 1000                  # Log frequency

replay_buffer_capacity: 500_000
batch_size: 512
device: cuda:0
```

### Important: Scaling Training Parameters

When changing `n_train_steps`, scale these proportionally:

```yaml
# 100M steps (original)
n_train_steps: 100_000_001
n_eval: 10000   # eval every ~200k steps
n_save: 5000    # save every ~100k steps
n_log: 100000   # log every ~1k steps

# 1M steps (100x reduction)
n_train_steps: 1_000_000
n_eval: 100     # divide by 100
n_save: 50      # divide by 100
n_log: 1000     # divide by 100
```

**Why?** These parameters are episode-based internally:
```python
eval_every_episodes = (n_train_steps / 50) / n_eval
eval_every_steps = eval_every_episodes * 50
```

## Running Training

```bash
source arel/bin/activate
cd SurRoL/rl
python train_rl.py
```

### Training Output
```
| train | F: 2000 | S: 2000 | E: 40 | L: 50 | R: -50.00 | SR: 0.00 | BS: 512 | FPS: 45.79 | T: 0:00:41
```
- **F**: Frames (env steps)
- **S**: Total steps
- **E**: Episodes
- **L**: Episode length
- **R**: Avg reward
- **SR**: Success rate
- **BS**: Batch size (0 = seed phase, 512 = learning)
- **FPS**: Frames/second

## Troubleshooting

### NumPy 2.x Incompatibility
**Error**: "module compiled using NumPy 1.x cannot run in NumPy 2.x"
**Fix**: `pip install 'numpy<2'`

### Gym API Mismatch
**Error**: `ValueError: not enough values to unpack (expected 5, got 4)`
**Fix**: Update `step()` methods to return 5 values (see modifications above)

### Environment Not Registered
**Error**: `gym.error.NameNotFound: Environment X doesn't exist`
**Fix**: Add `import surrol.gym` to training script

### Demo Count Mismatch
**Error**: `IndexError: index 100 is out of bounds`
**Fix**: Set `num_demo` to match available demos (usually 100)

### Missing use_wb
**Error**: `AttributeError: 'DictConfig' object has no attribute 'use_wb'`
**Fix**: Add `use_wb: False` to `configs/train.yaml`

## Demo Data

Demo files contain 100 successful trajectories (5000 transitions):
- Location: `gcdt_ckpt/success_demo/`
- Format: `data_<TaskName>_random_100.npz`
- Usage: Pre-fills replay buffer and initializes normalizers

## Available Tasks

All tasks have sparse rewards (-1 until success, 0 at goal):
- `NeedlePickRL-v0`, `NeedleReachRL-v0`, `NeedleRegraspRL-v0`
- `GauzeRetrieveRL-v0`
- `PegTransferRL-v0`, `BiPegTransferRL-v0`
- `PickAndPlaceRL-v0`
- `MatchBoardRL-v0`, `MatchBoardPanelRL-v0`, `BiPegBoardRL-v0`

Episode length: 50 steps | Success threshold: 5mm from goal
