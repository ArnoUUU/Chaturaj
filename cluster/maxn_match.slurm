#!/bin/bash
# max^n + NNUE against MCTS + MLP: 8 single-threaded match shards in parallel on one node.
#
#   sbatch --export=ALL,NNUE=models/<nnue-label> cluster/maxn_match.slurm
#   sbatch --export=ALL,NNUE=models/<nnue-label>,MLP=models/opponent_pool/<name>.pt cluster/maxn_match.slurm
#
# Three settings, the max^n bot in one seat (every seat in turn), the MLP in the other three,
# 4-ply random openings shared between settings (same seeds):
#   vs32     max^n 250 ms per move    vs MCTS at 32 simulations (the training setting)   2 shards x 20 games
#   eq250    max^n 250 ms per move    vs MCTS at the simulations that fit in 250 ms      3 shards x 16 games
#   eq1000   max^n 1000 ms per move   vs MCTS at the simulations that fit in 1000 ms     3 shards x 16 games
# "Fit" is measured on this node at the start (python -m bot.maxn calibrate).
#
# Output: logs/maxn/<jobid>/<setting>_s<seed>.jsonl (every game, with moves) and .json (shard
# result); summary.jsonl at the end has one merged line per setting.
#SBATCH --job-name=chaturaji-maxn
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source cluster/activate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one core per shard
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1} # the compiled core too
python -c "import bot.nnue as n; print('compiled core (engine/):', 'yes' if n.CORE_AVAILABLE else 'no, NumPy fallback; run cluster/build_engine.sh')"

NNUE=${NNUE:?set NNUE=models/<nnue-label>}
MLP=${MLP:-models/opponent_pool/league_model.pt}
OUT=logs/maxn/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

MS_PER_SIM=$(python -m bot.maxn calibrate "$MLP")
SIMS250=$(awk -v m="$MS_PER_SIM" 'BEGIN { printf "%d", 250 / m }')
SIMS1000=$(awk -v m="$MS_PER_SIM" 'BEGIN { printf "%d", 1000 / m }')
echo "MCTS: ${MS_PER_SIM} ms/sim -> ${SIMS250} sims in 250 ms, ${SIMS1000} in 1000 ms"

shard() {  # name seed games time_ms sims
  python -m bot.maxn "$NNUE" --opponent "$MLP" --games "$3" --time-ms "$4" --sims "$5" \
    --seed "$2" --out "$OUT/$1_s$2.jsonl" > "$OUT/$1_s$2.json" 2> "$OUT/$1_s$2.err" &
}
shard vs32   0 20 250  32
shard vs32   1 20 250  32
shard eq250  0 16 250  "$SIMS250"
shard eq250  1 16 250  "$SIMS250"
shard eq250  2 16 250  "$SIMS250"
shard eq1000 0 16 1000 "$SIMS1000"
shard eq1000 1 16 1000 "$SIMS1000"
shard eq1000 2 16 1000 "$SIMS1000"
wait

: > "$OUT/summary.jsonl"
for s in vs32 eq250 eq1000; do
  python - "$s" "$OUT" "$MS_PER_SIM" <<'EOF' >> "$OUT/summary.jsonl"
import json, glob, sys
from bot.maxn import merge
name, out, ms = sys.argv[1], sys.argv[2], float(sys.argv[3])
shards = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/{name}_s*.json"))]
res = merge(sorted(glob.glob(f"{out}/{name}_s*.jsonl")))
res.update({"setting": name, "mcts_ms_per_sim": ms, "mcts_sims": shards[0]["sims"],
            "maxn_time_ms": shards[0]["maxn_time_ms"],
            "maxn_mean_depth": round(sum(s["depth_reached"]["mean"] for s in shards) / len(shards), 2)})
print(json.dumps(res))
EOF
done
cat "$OUT/summary.jsonl"
