# xarel

This repository mirrors the current working state of the SurRoL benchmark codebase together with the `gcdt_ckpt` experimentation artifacts and the original `SETUP_README.md` instructions that describe how to prepare the environment.

## Layout

- `SurRoL/` – full research codebase cloned from the working tree located under `/home/gak/Documents/edoardo/SurRoL` (state-based RL lives under `SurRoL/rl`).
- `gcdt_ckpt/` – success demonstration datasets and checkpoints.
- `SETUP_README.md` – original environment setup notes from the workspace root.

## Usage

1. Consult `SETUP_README.md` for the exact dependency stack that was used locally.
2. Install the Python requirements under `SurRoL/rl/requirements.txt` (a virtual environment is recommended).
3. Run training scripts from `SurRoL/rl`, for example:

   ```bash
   cd SurRoL/rl
   python train_rl.py --help
   ```

Large binary artifacts (e.g., checkpoints) are kept as-is; consider migrating them to Git LFS if you plan to make the public repository lightweight.

## CIAI HPC quickstart

Inside the CIAI cluster you can keep the project under `/l/users/<user>/xarel` and use the helper scripts in `hpc/`.

1. **Create/activate the Conda environment (run once on CIAI):**

   ```bash
   source /apps/local/conda_init.sh
   conda create -p /l/users/$USER/envs/surrol-rl python=3.10 -y
   conda activate /l/users/$USER/envs/surrol-rl
   pip install -r /l/users/$USER/xarel/SurRoL/rl/requirements.txt
   ```

2. **Interactive run on `cscc-gpu-p` (pick the right QoS):**

   - **General work (longer interactive sessions):**

   ```bash
   cd /l/users/$USER/xarel
   chmod +x hpc/debug_run.sh
   srun -p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:1 --cpus-per-task=8 \
      --time=02:00:00 --pty bash hpc/debug_run.sh
   ```

   - **Debug queue (≤3 h, ≤4 GPUs per user total):**

   ```bash
   cd /l/users/$USER/xarel
   chmod +x hpc/debug_run.sh
   srun -p cscc-gpu-p -q gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 \
      --time=00:30:00 --pty bash hpc/debug_run.sh
   ```

   Both commands run the short `train_rl.py` session bundled in `hpc/debug_run.sh` to quickly verify dependencies and configs before submitting a long job.

3. **Queued training job on the CIAI long partition:**

   ```bash
   cd /l/users/$USER/xarel
   chmod +x hpc/train_ciai_long.sbatch
   sbatch hpc/train_ciai_long.sbatch
   ```

   The script requests the policy-required `long` partition resources (`--gres=gpu:4`, `--qos=gpu-12`) and launches `SurRoL/rl/train_rl.py`. Logs show up under `/l/users/$USER/xarel/logs/`.

Feel free to override task/seed within the batch script by exporting `TASK` or `SEED` variables before calling `sbatch` (e.g., `TASK=PegTransferRL-v0 sbatch hpc/train_ciai_long.sbatch`).
