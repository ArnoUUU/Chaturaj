# bot — self-play engine (PyTorch)

A neural network evaluates positions for a 4-player Monte Carlo Tree Search, and learns
from games it plays against itself. Details: [PLAN.md](PLAN.md); coordination between
colors: [COORDINATION.md](COORDINATION.md).

Requires `numpy` and `torch` (the CPU build works). Run commands from the repository root.
For GPU runs on the Tillicum cluster, see [../cluster/README.md](../cluster/README.md).

## Choices
- **Model** (`--model`):
  - `mlp` (default): encoding layer followed by 3 fully connected layers (`mlp.py`).
  - `resnet`: convolutional baseline (`model.py`).
- **Search selection rule** (`--selection`):
  - `puct` (default): the AlphaZero rule, which uses the network's move priors.
  - `ucb1`: the classic Upper Confidence Bound, `Q + c·sqrt(ln N / n)`. It tries every
    move once, then balances average value against how rarely a move was tried
    (`--c-ucb`, default √2).
- **Device** (`--device`): `cuda` when available, otherwise `cpu`.

## Quick test run (CPU, a few seconds)
```powershell
python -m bot.train --tag smoke-cpu --model mlp --embed 64 --widths 128 64 32 --selection ucb1 --iterations 2 --games 2 --sims 16 --steps 20 --batch 64 --max-plies 200
```

Each run gets a label (date, tag, model and size, selection rule and simulations,
iterations, device), for example
`2026-11-21_1552_smoke-cpu_mlp-e64-w128x64x32_ucb1-s16_it2_cpu`, and writes:

```
models/<label>/model.pt, meta.json   model + run details + evaluation vs random (see models/README.md)
models/index.jsonl                   one line per finished run
logs/<label>/log_0000.jsonl, ...     chunked JSON-lines log (--log-chunk lines per file)
logs/<label>/games/*.json            sample self-play games
```

Watch a self-play game, or continue a run:
```powershell
python -m chaturaji.viewer logs/<label>/games/iter0001_game0.json
```
```powershell
python -m bot.train --resume models/<label> --iterations 20
```

Check what hardware PyTorch sees, with timings (also works on CPU):
```powershell
python -m bot.gpu_check
```

## Files
| File | Contents |
|---|---|
| `features.py` | rotated board encoding (the mover always sits in Red's seat), action mapping |
| `mlp.py` | encoding layer + 3 FC evaluator, sparse features, MCTS evaluator |
| `model.py` | ResNet baseline |
| `registry.py` | model kinds, labels, save/load to `models/`, chunked logger for `logs/` |
| `mcts.py` | 4-player search with PUCT or UCB1 selection (each player maximizes its own value) |
| `selfplay.py` | plays one game and returns training samples and a game record |
| `evaluate.py` | test-run evaluation against random players (each seat in turn) |
| `train.py` | replay buffer, loss, training loop, CLI |
| `gpu_check.py` | GPU/CPU check and timings; reports go to `logs/gpu_check/` |

## Status
- The whole pipeline runs end to end on CPU and is covered by tests (`tests/test_bot.py`).
- Self-play search evaluates one position at a time, so GPUs mainly help training steps
  for now. Batched leaf evaluation is next (PLAN.md §5.6).

## Tests
```powershell
python -m pytest -q tests/test_bot.py
```
