#!/bin/bash
# GPU test job: unit tests, GPU check, and a short GPU training run that saves a
# labelled model to models/ and chunked logs to logs/.
#
#   mkdir -p logs/slurm && sbatch cluster/gpu_test.slurm
#
# Fill in the account/partition lines with your Tillicum values (see cluster/README.md).
#SBATCH --job-name=chaturaji-gputest
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out

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
