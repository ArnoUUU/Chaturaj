#!/bin/bash
# Activate the chaturaji env inside a job or an interactive session:
#   source cluster/activate.sh
# where setup_env.sh put Miniforge and the env (written by setup_env.sh)
SITE_ENV="$(dirname "${BASH_SOURCE[0]}")/site.env"
# shellcheck disable=SC1090
[ -f "$SITE_ENV" ] && source "$SITE_ENV"
MINIFORGE_DIR=${MINIFORGE_DIR:-$HOME/miniforge3}
# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
if [ -n "${ENV_PREFIX:-}" ]; then conda activate "$ENV_PREFIX"; else conda activate chaturaji; fi
# match CPU threads to the Slurm allocation (self-play + data loading run on CPU)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p logs/slurm
