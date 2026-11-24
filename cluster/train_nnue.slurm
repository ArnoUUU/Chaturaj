#!/bin/bash
# Train NNUE sizes on the GPU, one after another, on the same data.
#
#   sbatch cluster/train_nnue.slurm                                   # medium, large, xl
#   sbatch --export=ALL,PRESETS="large xl",EPOCHS=6 cluster/train_nnue.slurm
#
# Data: the scraped games (scraper/data/records) and SELFPLAY (games.jsonl files). Copy
# them to Tillicum first (see cluster/README.md). The positions are built once into
# logs/nnue_cache.npz and reused by every size (and by later jobs).
#SBATCH --job-name=chaturaji-nnue
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source cluster/activate.sh

PRESETS=${PRESETS:-"medium large xl"}
EPOCHS=${EPOCHS:-4}
DROPOUT=${DROPOUT:-0.1}
BATCH=${BATCH:-1024}
LR=${LR:-5e-4}
CACHE=${CACHE:-logs/nnue_cache.npz}
SELFPLAY=${SELFPLAY:-$(ls logs/*/games.jsonl 2> /dev/null | tr '\n' ' ')}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "presets: $PRESETS | epochs $EPOCHS | dropout $DROPOUT | self-play: ${SELFPLAY:-none}"

for P in $PRESETS; do
  # shellcheck disable=SC2086
  python -m bot.train_nnue --preset "$P" --dropout "$DROPOUT" --epochs "$EPOCHS" --batch "$BATCH" \
    --lr "$LR" --device cuda --cache "$CACHE" --tag "nnue-$P" --selfplay $SELFPLAY 2>&1 | grep -v '^{'
done
