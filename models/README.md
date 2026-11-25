# models — saved, labelled bot models

Layout:

| Path | Contents |
|---|---|
| `models/<label>/` | one folder per training run (below) |
| `models/opponent_pool/` | models kept as league opponents: `<name>.pt` + `<name>.meta.json`. A `BEST_` prefix marks the best one |
| `models/benchmark.json` | the benchmark model, as a reference such as `opponent_pool/BEST_<name>.pt` (see `bot/benchmark.py`) |
| `models/index.jsonl`, `benchmark_history.jsonl` | append-only records |
| `logs/models/<label>/` | retired model folders: weights, meta and snapshots (not used for new runs) |

Any of these can be given wherever a model is expected (`--init`, `--opponents`, `bot.evaluate`,
`bot.benchmark`): `bot.registry.find_model()` accepts a file, a folder, a label or a pool name,
with or without `BEST_`, and searches `models/`, `models/opponent_pool/` and `logs/models/`.

Every training run (`python -m bot.train`) saves its model in a folder named by its label:

```
<date>_<time>_<tag>_<kind>-<size>_<selection>-s<sims>_it<iterations>_<cpu|gpu>
e.g. 2026-11-21_1552_smoke-cpu_mlp-e64-w128x64x32_ucb1-s16_it2_cpu
```

| Part | Meaning |
|---|---|
| `smoke-cpu` | the `--tag` of the run: what kind of test it was |
| `mlp-e64-w128x64x32` | model kind and size (encoding width 64, FC widths 128/64/32) |
| `ucb1-s16` | MCTS selection rule (`puct` or `ucb1`) and simulations per move |
| `it2` | training iterations |
| `cpu` / `gpu` | where it was trained |

Each folder holds:
- `model.pt`: weights, optimizer state, model kind and config. Load it with
  `bot.registry.load_model("models/<label>")`, or continue training with
  `python -m bot.train --resume models/<label>`.
- `meta.json`: everything about the run: arguments, environment (host, GPU,
  torch/CUDA versions, git commit, Slurm job id), losses per iteration, and the
  evaluation against random players (score share, win rate, average place).

`index.jsonl` has one line per finished run, for comparing runs at a glance.

Run logs are in `logs/<label>/` (split into chunks). They aren't committed to git.

## Opponent pool
Each pool file keeps its full run label inside (`label` in the `.pt` and `.meta.json`), so the
file name can be short. The pool currently holds `BEST_taran0915_model` (the benchmark),
`ihavethesauce_model`, `eternalvoias_model` and `league_model`.

## Saved so far
| Label | What it is |
|---|---|
| `2026-11-21_1552_smoke-cpu_mlp-e64-w128x64x32_ucb1-s16_it2_cpu` | First end-to-end test run on a laptop CPU: small MLP, UCB1 search, 2 iterations × 2 self-play games. A check that the pipeline works, not a strong model (0.296 score share vs random over 4 games). |
