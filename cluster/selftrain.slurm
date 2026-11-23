#!/bin/bash
# Self-play training for as long as the job runs, keeping an hourly copy of the model.
#
#   sbatch cluster/selftrain.slurm                                        # new run, 24h
#   sbatch --time=48:00:00 cluster/selftrain.slurm                        # longer run
#   sbatch --export=ALL,INIT=models/<supervised-label> cluster/selftrain.slurm   # start from a model
#   sbatch --export=ALL,RESUME=models/<label> cluster/selftrain.slurm     # continue a run
#   sbatch --export=ALL,RESUME=models/<label>/snapshots/<snap>.pt cluster/selftrain.slurm
#
# Output (see cluster/README.md):
#   models/<label>/model.pt                 latest weights, updated every iteration
#   models/<label>/snapshots/snap_*.pt      weights every SNAPSHOT_MIN minutes, newest KEEP_SNAPSHOTS kept
#   logs/<label>/                           full JSON-lines log (chunked) + a sample game every 50 iterations
#   logs/slurm/<job>_<id>.out              short progress lines only
#
# The run stops starting new iterations MARGIN_MIN minutes before the job's time limit,
# so the final save and evaluation finish before Slurm kills the job.
#SBATCH --job-name=chaturaji-selftrain
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME   # optional: Slurm uses your default account (remove one # to set it)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source cluster/activate.sh

TAG=${TAG:-selftrain}
MODEL=${MODEL:-mlp}
SELECTION=${SELECTION:-puct}
GAMES=${GAMES:-16}
SIMS=${SIMS:-200}
STEPS=${STEPS:-500}
BATCH=${BATCH:-1024}
EVAL_GAMES=${EVAL_GAMES:-16}
SNAPSHOT_MIN=${SNAPSHOT_MIN:-60}
MARGIN_MIN=${MARGIN_MIN:-20}
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-6}    # older hourly snapshots are deleted (10 GB home quota)
PRINT_EVERY=${PRINT_EVERY:-10}         # one short stdout line every N iterations
SAVE_GAMES_EVERY=${SAVE_GAMES_EVERY:-50}

# Training budget = time left in this job minus a safety margin (override with HOURS=...).
if [ -z "${HOURS:-}" ]; then
  END=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'EndTime=[^ ]*' | cut -d= -f2)
  LEFT=$(( $(date -d "$END" +%s) - $(date +%s) - MARGIN_MIN * 60 ))
  if [ "$LEFT" -le 600 ]; then
    echo "Job time limit leaves only ${LEFT}s after the ${MARGIN_MIN}-minute margin; ask for more --time." >&2
    exit 1
  fi
  HOURS=$(awk -v s="$LEFT" 'BEGIN { printf "%.3f", s / 3600 }')
fi
echo "job ${SLURM_JOB_ID:-local}: training for ${HOURS}h, snapshot every ${SNAPSHOT_MIN} min"

python -m bot.gpu_check --require-gpu --sims 16 > /dev/null || { python -m bot.gpu_check --require-gpu --sims 16; exit 1; }
echo "GPU ok: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

COMMON=(--device cuda --iterations 0 --hours "$HOURS" --snapshot-every "$SNAPSHOT_MIN"
        --games "$GAMES" --sims "$SIMS" --steps "$STEPS" --batch "$BATCH"
        --selection "$SELECTION" --eval-games "$EVAL_GAMES"
        --quiet --print-every "$PRINT_EVERY" --keep-snapshots "$KEEP_SNAPSHOTS"
        --save-games-every "$SAVE_GAMES_EVERY")

if [ -n "${RESUME:-}" ]; then
  python -m bot.train --resume "$RESUME" "${COMMON[@]}"
elif [ -n "${INIT:-}" ]; then
  python -m bot.train --tag "$TAG" --init "$INIT" "${COMMON[@]}"
else
  python -m bot.train --tag "$TAG" --model "$MODEL" "${COMMON[@]}"
fi
