#!/bin/bash
# Build the compiled core (engine/) and install it into the chaturaji env.
#   bash cluster/build_engine.sh
# Installs Rust (rustup) into /gpfs/scrubbed/$USER the first time (it's about 1.5 GB, too
# big for the 10 GB home quota next to the env), then builds with maturin.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/activate.sh

SCRUB=/gpfs/scrubbed/$USER
if [ -d "$SCRUB" ]; then BASE=$SCRUB; else BASE=$HOME; fi
export RUSTUP_HOME=${RUSTUP_HOME:-$BASE/rustup}
export CARGO_HOME=${CARGO_HOME:-$BASE/cargo}
export PATH="$CARGO_HOME/bin:$PATH"

if ! command -v cargo > /dev/null; then
  echo "installing Rust into $CARGO_HOME"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
fi
if ! command -v cc > /dev/null && command -v module > /dev/null; then
  module load gcc 2> /dev/null || true        # Rust needs a C linker
fi
python -m pip install --quiet "maturin>=1.5,<2.0"

cd engine
cargo test --no-default-features --release
python -m maturin build --release -i "$(command -v python)"
python -m pip install --no-deps --force-reinstall "$(ls -t target/wheels/*.whl | head -1)"
cd ..
python -c "import bot.nnue as n; print('chaturaji_engine installed:', n.CORE_AVAILABLE)"

# Later shells only need the Python module, not cargo. To rebuild, run this script again.
