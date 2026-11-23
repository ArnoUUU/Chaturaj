#!/bin/bash
# One-time setup on the cluster login node: install Miniforge (if needed) and create the
# "chaturaji" conda env from cluster/environment.yml.
#
#   bash cluster/setup_env.sh
#
# Optional environment variables:
#   MINIFORGE_DIR   where Miniforge lives   (default: /gpfs/scrubbed/$USER/miniforge3 if that
#                   directory exists, else $HOME/miniforge3). The env with CUDA torch is ~6 GB,
#                   too much for a 10 GB home, so scrubbed storage is the default.
#   ENV_PREFIX      put the env at this path instead of inside Miniforge
#   CONDA_PKGS_DIRS package cache location  (default: $MINIFORGE_DIR/pkgs, emptied at the end)
set -euo pipefail
cd "$(dirname "$0")/.."

SCRUB=/gpfs/scrubbed/$USER
if [ -z "${MINIFORGE_DIR:-}" ]; then
  if [ -d "$SCRUB" ]; then MINIFORGE_DIR=$SCRUB/miniforge3; else MINIFORGE_DIR=$HOME/miniforge3; fi
fi
export PIP_NO_CACHE_DIR=1   # don't keep a copy of the ~3 GB torch wheel
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "Installing Miniforge into $MINIFORGE_DIR"
  installer="Miniforge3-$(uname)-$(uname -m).sh"
  curl -fsSL -o "/tmp/$installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
  bash "/tmp/$installer" -b -p "$MINIFORGE_DIR"
  rm -f "/tmp/$installer"
fi
# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$MINIFORGE_DIR/pkgs}

if [ -n "${ENV_PREFIX:-}" ]; then
  if [ -d "$ENV_PREFIX" ]; then conda env update -p "$ENV_PREFIX" -f cluster/environment.yml --prune
  else conda env create -p "$ENV_PREFIX" -f cluster/environment.yml; fi
  conda activate "$ENV_PREFIX"
else
  if conda env list | grep -qE '^chaturaji\s'; then conda env update -n chaturaji -f cluster/environment.yml --prune
  else conda env create -f cluster/environment.yml; fi
  conda activate chaturaji
fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda,
      "| GPU visible here:", torch.cuda.is_available(), "(login nodes usually have none)")
EOF
conda clean -afy > /dev/null   # drop downloaded package archives
echo "MINIFORGE_DIR=$MINIFORGE_DIR" > cluster/site.env
if [ -n "${ENV_PREFIX:-}" ]; then echo "ENV_PREFIX=$ENV_PREFIX" >> cluster/site.env; fi
echo "Env location saved to cluster/site.env (read by cluster/activate.sh)."
echo "Done. In jobs, use: source cluster/activate.sh"
