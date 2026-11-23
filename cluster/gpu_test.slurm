#!/bin/bash
# GPU test job: unit tests, GPU check, and a short GPU training run that saves a
# labelled model to models/ and chunked logs to logs/.
#
#   mkdir -p logs/slurm && sbatch cluster/gpu_test.slurm
#
# Runs on the debug QOS (30 min, 1 GPU) per the Tillicum Slurm guide.
#SBATCH --job-name=chaturaji-gputest
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME   # optional: Slurm uses your default account (remove one # to set it)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source cluster/activate.sh

echo "== node $(hostname), job ${SLURM_JOB_ID:-local}"
nvidia-smi || true

echo "== unit tests"
python -m pytest -q

echo "== GPU check"
python -m bot.gpu_check --require-gpu

echo "== short GPU training run"
python -m bot.train --tag gputest --device cuda --model mlp --selection ucb1 \
  --iterations 2 --games 4 --sims 32 --steps 50 --batch 512 --eval-games 4
