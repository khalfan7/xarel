# xarel

This repository mirrors the current working state of the SurRoL benchmark codebase together with the `gcdt_ckpt` experimentation artifacts and the original `SETUP_README.md` instructions that describe how to prepare the environment.

## Layout

- `SurRoL/` – full research codebase cloned from the working tree located under `/home/gak/Documents/edoardo/SurRoL`.
- `gcdt_ckpt/` – success demonstration datasets and checkpoints.
- `SETUP_README.md` – original environment setup notes from the workspace root.

## Usage

1. Consult `SETUP_README.md` for the exact dependency stack that was used locally.
2. Install the Python requirements under `SurRoL/Benchmark/state_based/rl/requirements.txt` (a virtual environment is recommended).
3. Run training scripts from `SurRoL/Benchmark/state_based/rl`, for example:

   ```bash
   cd SurRoL/Benchmark/state_based/rl
   python train_rl.py --help
   ```

Large binary artifacts (e.g., checkpoints) are kept as-is; consider migrating them to Git LFS if you plan to make the public repository lightweight.
