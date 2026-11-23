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

## Copy the repo to Tillicum (from Windows PowerShell)
The archive leaves out the scraper's browser profiles (they hold your chess.com login),
scraped data, logs and caches. Replace `NETID` with your UW NetID.
```powershell
cd C:\Users\Arnav\Downloads
tar -czf chaturaj.tgz --exclude=Chaturaj/scraper/.firefox-profile --exclude=Chaturaj/scraper/.selenium-profile --exclude=Chaturaj/scraper/data --exclude=Chaturaj/logs --exclude=Chaturaj/gui/.cache --exclude=__pycache__ --exclude=.pytest_cache --exclude=Chaturaj/.claude Chaturaj
scp chaturaj.tgz NETID@tillicum.hyak.uw.edu:~/
ssh NETID@tillicum.hyak.uw.edu "tar -xzf chaturaj.tgz && rm chaturaj.tgz && mkdir -p Chaturaj/logs/slurm"
```

## Copy results back (no git)
On Tillicum, `ls -t ~/Chaturaj/models | head` shows the newest run label. Then on Windows:
```powershell
cd C:\Users\Arnav\Downloads\Chaturaj
scp -r NETID@tillicum.hyak.uw.edu:~/Chaturaj/models/LABEL models/
scp -r NETID@tillicum.hyak.uw.edu:~/Chaturaj/logs/LABEL logs/
scp -r NETID@tillicum.hyak.uw.edu:~/Chaturaj/logs/slurm logs/
```
`models/index.jsonl` isn't copied so your local list isn't overwritten; the run's
`meta.json` has the same information.

## One-time setup (login node)
```bash
cd ~/Chaturaj
bash cluster/setup_env.sh      # Miniforge + env in /gpfs/scrubbed/$USER (falls back to ~/miniforge3)
```
The env with CUDA PyTorch is about 6 GB, too much for the 10 GB home quota, so it goes to
`/gpfs/scrubbed/$USER` and the package caches are emptied. Scrubbed storage can be purged
when unused; if `source cluster/activate.sh` fails later, rerun `setup_env.sh`. The chosen
location is saved in `cluster/site.env` so jobs find it.

The scripts follow Tillicum's Slurm guide: `--qos` (no partition), 1 GPU with its
8 CPUs and 200 GB RAM, `debug` QOS (30 min) for `gpu_test.slurm` and `normal` (24 h max)
for training. No account line is needed; Slurm uses your default (uncomment
`##SBATCH --account=` to pick one). Cost is $0.90 per GPU-hour: a full 24 h
`selftrain.slurm` job is about $21.60, the GPU test about $0.45.

## Run
```bash
mkdir -p logs/slurm             # Slurm needs this before it can write the .out file
sbatch cluster/gpu_test.slurm          # tests + GPU check + short GPU training run
squeue -u $USER                        # watch the queue
tail -f logs/slurm/chaturaji-gputest_<jobid>.out
sbatch --time=08:00:00 --export=ALL,TAG=mlp-ucb1,SELECTION=ucb1,ITERS=100 cluster/train.slurm
```
### Long self-training with hourly snapshots
`selftrain.slurm` self-plays and trains until 20 minutes before the job's time limit, then
saves and evaluates. Every hour it keeps a copy in `models/<label>/snapshots/`.
```bash
sbatch --time=24:00:00 cluster/selftrain.slurm
sbatch --export=ALL,INIT=models/<supervised-label> cluster/selftrain.slurm    # start from a model
sbatch --export=ALL,RESUME=models/<label> cluster/selftrain.slurm             # next job of the same run
sbatch --export=ALL,SNAPSHOT_MIN=30,SIMS=400 cluster/selftrain.slurm          # other settings
```
Settings (pass with `--export=ALL,NAME=value`): `TAG`, `MODEL`, `SELECTION`, `GAMES`, `SIMS`,
`STEPS`, `BATCH`, `EVAL_GAMES`, `SNAPSHOT_MIN` (default 60), `MARGIN_MIN` (default 20), `HOURS`
(override the budget), `KEEP_SNAPSHOTS` (default 6), `PRINT_EVERY` (default 10),
`SAVE_GAMES_EVERY` (default 50).

Logging is kept small for the 10 GB quota: the Slurm `.out` file gets one short line every
`PRINT_EVERY` iterations, the full per-iteration record goes to `logs/<label>/log_*.jsonl`
(a few hundred bytes per iteration), one sample game is saved every `SAVE_GAMES_EVERY`
iterations, and snapshots are weights only (~12 MB each, newest 6 kept). A 24 h run uses
roughly 150 MB. The replay buffer isn't saved, so a resumed job refills it from fresh games.

Interactive debugging on a GPU node:
```bash
salloc --qos=interactive --gpus=1 --time=01:00:00
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
