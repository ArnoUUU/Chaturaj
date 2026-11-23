#!/bin/bash
# Longer GPU training run. Override settings without editing the file, e.g.
#   sbatch --time=08:00:00 --export=ALL,TAG=mlp-ucb1,SELECTION=ucb1,ITERS=100 cluster/train.slurm
# Resume a run:
#   sbatch --export=ALL,RESUME=models/<label> cluster/train.slurm
#SBATCH --job-name=chaturaji-train
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME   # optional: Slurm uses your default account (remove one # to set it)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source cluster/activate.sh

TAG=${TAG:-tillicum}
MODEL=${MODEL:-mlp}
SELECTION=${SELECTION:-puct}
ITERS=${ITERS:-50}
GAMES=${GAMES:-16}
SIMS=${SIMS:-200}
STEPS=${STEPS:-500}
BATCH=${BATCH:-1024}

python -m bot.gpu_check --require-gpu --sims 16

if [ -n "${RESUME:-}" ]; then
  python -m bot.train --device cuda --resume "$RESUME" --iterations "$ITERS" \
    --games "$GAMES" --sims "$SIMS" --steps "$STEPS" --batch "$BATCH" --selection "$SELECTION"
else
  python -m bot.train --tag "$TAG" --device cuda --model "$MODEL" --selection "$SELECTION" \
    --iterations "$ITERS" --games "$GAMES" --sims "$SIMS" --steps "$STEPS" --batch "$BATCH" \
    --eval-games 16
fi
