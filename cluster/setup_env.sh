#!/bin/bash
# One-time setup on the cluster login node: install Miniforge (if needed) and create the
# "chaturaji" conda env from cluster/environment.yml.
#
#   bash cluster/setup_env.sh
#
# Optional environment variables:
#   MINIFORGE_DIR   where Miniforge lives          (default: $HOME/miniforge3)
#   ENV_PREFIX      put the env at this path instead of inside Miniforge, e.g. a project or
#                   scratch directory with more quota than $HOME
#   CONDA_PKGS_DIRS package cache location         (default: $MINIFORGE_DIR/pkgs)
set -euo pipefail
cd "$(dirname "$0")/.."

MINIFORGE_DIR=${MINIFORGE_DIR:-$HOME/miniforge3}
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
echo "Done. In jobs, use: source cluster/activate.sh"
