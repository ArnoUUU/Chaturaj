# cluster — GPU runs on Tillicum (Slurm + conda)

## Recommended environment: Miniforge
**Use Miniforge** (conda-forge's installer) for the conda environment, and install PyTorch
from its official CUDA wheels with pip inside that env.
- **Free to use:** Miniforge defaults to the free conda-forge channel and has no
  Anaconda licensing terms, which can apply to university use of Anaconda/Miniconda.
- **Fast:** it ships with the fast libmamba solver.
- **PyTorch from pip:** PyTorch no longer publishes conda packages. The CUDA runtime comes
  inside the wheel, so the GPU node only needs an NVIDIA driver, with no CUDA module to load.
- **Alternatives:**
  - If Tillicum provides its own conda/Miniforge module or recommends Apptainer
    containers, follow its documentation.
  - `uv` + a plain venv also works and is lighter, but you asked for conda, and conda
    handles non-Python packages better if you need them later.

## One-time setup (login node)
```bash
git clone <this repo> && cd Chaturaj
bash cluster/setup_env.sh                                  # Miniforge in ~/miniforge3, env "chaturaji"
ENV_PREFIX=/path/with/more/quota/envs/chaturaji bash cluster/setup_env.sh   # optional: env outside $HOME
```
Home directories on clusters are usually small, so if Tillicum gives you a project or
scratch directory, point `ENV_PREFIX` (and `CONDA_PKGS_DIRS`) there.

Then edit `#SBATCH --account=CHANGE_ME` and `#SBATCH --partition=CHANGE_ME` in
`gpu_test.slurm` and `train.slurm` to your Tillicum account and GPU partition. `sinfo`
and Tillicum's documentation list them. If Tillicum uses `--gres=gpu:1` instead of
`--gpus=1`, change that line too.

## Run
```bash
mkdir -p logs/slurm
sbatch cluster/gpu_test.slurm          # tests + GPU check + short GPU training run
squeue -u $USER                        # watch the queue
tail -f logs/slurm/chaturaji-gputest-<jobid>.out
sbatch --time=08:00:00 --export=ALL,TAG=mlp-ucb1,SELECTION=ucb1,ITERS=100 cluster/train.slurm
```
Interactive debugging on a GPU node (same account/partition):
```bash
salloc --account=CHANGE_ME --partition=CHANGE_ME --gpus=1 --cpus-per-task=8 --time=00:30:00
source cluster/activate.sh && python -m bot.gpu_check --require-gpu
```

## What a run produces
| Path | Contents |
|---|---|
| `models/<label>/model.pt` | weights, optimizer, model kind and config (resumable) |
| `models/<label>/meta.json` | run arguments, GPU/torch/git info, Slurm job id, per-iteration losses, evaluation vs random |
| `models/index.jsonl` | one line per finished run |
| `logs/<label>/log_0000.jsonl …` | the run's log, split into chunks of `--log-chunk` lines |
| `logs/<label>/games/` | sample self-play games (open with `python -m chaturaji.viewer`) |
| `logs/gpu_check/` | GPU check reports (device, TFLOPS, positions/s, sims/s) |
| `logs/slurm/` | raw Slurm stdout/stderr |

Labels say what was run, e.g. `2026-11-21_1530_gputest_mlp-e512-w1024x512x256_ucb1-s32_it2_gpu`:
date, tag, model and size, selection rule and simulations, iterations, and device.

## Notes
- Self-play search runs on the CPU one position at a time, so the GPU is mostly
  idle during self-play until batched leaf evaluation is added (bot/PLAN.md §5.6). It
  already speeds up training steps. Ask for enough CPUs (`--cpus-per-task`).
- `bot.gpu_check --require-gpu` makes a job fail early if PyTorch can't see the GPU.
