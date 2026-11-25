# engine_pg

Playground copy of the Chaturaj engine core (`engine/`), without the Python bindings, with
a paranoid-search option and a match runner.

## What was copied

`src/nnue.rs`, `src/rules.rs`, `src/search.rs` and `Cargo.lock` come from `engine/` as of commit
d2d12ab (selective depth and the search preferences included). `src/lib.rs` only declares the three modules (the PyO3 module is gone). `Cargo.toml`
depends on `rayon` alone.

## What changed

- `search.rs`, default behaviour unchanged:
  - `node_limit: Option<u64>` (`set_node_limit`): the search aborts once the node count passes
    the limit, like a timeout; iterative deepening keeps the last completed depth.
  - `paranoid: bool` (default false): paranoid alpha-beta. `search_paranoid` maximizes the root
    player's utility at its nodes and minimizes it at every other node, with alpha-beta cutoffs.
    Same move ordering, width selection, check extensions, and quiescence (`quiesce_paranoid`,
    same move set as `quiesce`). The discount and elimination bonus are applied per child by
    the same `shape` arithmetic (`shape_component`, with its inverse for passing windows down).
    Its own table `tt_p` stores depth, value, best move and an exact/lower/upper bound; the root
    player is part of the key.
  - `select` and `quiesce_moves` are the code that max^n and paranoid now share.
- `src/bin/arena.rs`: the match runner (below).
- This round (built with `--target-dir target/b2`, so the binary is `target/b2/release/arena`):
  - time budget: the `ms` key sets a time budget per move. Each game records the mean time per
    move for each side (`test_ms`, `base_ms` in the JSON, shown in the progress line), and the
    summary prints `time/move` and microseconds per node.
  - `pmin` (`paranoid_min_select`): where the selective width applies at the opponents' nodes,
    keep the children with the lowest root utility instead of the highest for the mover. Root
    player nodes are unchanged.
  - `psort` (`paranoid_sort`): search the kept children in the same key order as the selection
    (table move first). At depth 1 the batched leaves are sorted too, after the children without
    a leaf. Only the order changes, so the root value is the same as without `psort` at fixed
    depth (with width 0).
  - max^n ignores both options; a test checks its results are unchanged.
- `export_weights.py` (in `claude-playground/`): converts a bot/nnue.py `.npz` into `net.txt` +
  `net.bin` for the arena.

## Build and test

```
cd claude-playground/engine_pg
cargo build --release --offline
RAYON_NUM_THREADS=2 cargo test --release --offline
```

## Export weights

```
python claude-playground/export_weights.py <in.npz> <out_dir>
```

`out_dir` gets `net.txt` (H, layer count, one `n_out n_in residual` line per hidden layer, then
the output layer) and `net.bin` (little-endian float32: ft_w, ft_b, each layer's w and b, out_w,
out_b; row-major). `weights/nnue-v1/` holds the 2026-11-23 a256-h32x32 net.

## Run a match

```
RAYON_NUM_THREADS=1 ./target/release/arena --weights ../weights/nnue-v1 --games 40 --threads 4 \
    --seed 1 --out ../results/run.jsonl --test "nodes=3000,paranoid=1" --base "nodes=3000"
```

The binary also sets rayon to one thread at start-up, so each game is single-threaded. Games
run in parallel over `--threads`. Game `g` uses opening `g / 4` (4 random plies from a
seeded RNG) and puts the test player in seat `g % 4`, so each opening is played from all four
seats. One JSON line per game is appended to `--out`; the summary (mean test score with ±2 SE,
nodes and depth per side, score by test seat) is printed at the end.

## Spec keys

`--test` and `--base` take a comma list of `key=value`:

| key | default | meaning |
|---|---|---|
| depth | unset | fixed search depth (u32) |
| nodes | unset | node limit per move (u64); iterative deepening to `max_depth` |
| ms | unset | time budget per move in milliseconds (f64); iterative deepening to `max_depth` |
| max_depth | 12 | deepest iteration when `depth` is unset |
| width | 4 | children kept per node from ply `full_plies` on (0 = all) |
| full_plies | 1 | plies searched in full before selective depth |
| qdepth | 2 | quiescence depth |
| check_ext | 3 | check extensions along a line |
| discount | 0.9985 | value pull towards 0.5 per move |
| elim_bonus | 0.008 | bonus for eliminating a player |
| elim_horizon | 200 | turn at which the elimination bonus reaches 0 |
| paranoid | 0 | 1 = paranoid alpha-beta instead of max^n |
| pmin | 0 | 1 = paranoid only: at the opponents' nodes keep the `width` children with the lowest root utility |
| psort | 0 | 1 = paranoid only: search the kept children in static-evaluation order (table move first) |
| tt_size | 1000000 | transposition table entries before it is cleared |

A spec needs `depth`, `nodes` or `ms`. Note: the `depth` of a spec with `nodes` or `ms` set is
the deepest completed iteration; a limit that stops depth 1 falls back to the best child by one
batched evaluation (depth 0 in the result). With `ms` and `nodes` both set, both limits apply.
