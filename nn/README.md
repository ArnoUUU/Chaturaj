# nn/: a small neural-net engine that learns to beat greedy

Greedy (`bot/greedy.py`, preset `greedy`, depth 4) is the baseline. Every score here is the
test engine's mean **pairwise score of 3** against three greedy bots on final points (1 per
opponent finished above, 0.5 per tie), so **1.5 = even with greedy**.

## Pieces

| File | What it is |
|---|---|
| `engine/src/greedy.rs` | Rust port of `bot/greedy.py` (same eval, paranoid alpha-beta, move ordering and table). About 170x faster. Whole games at depths 3 and 4 match the Python bot move for move (`nn/check_greedy.py`). |
| `engine/src/features.rs` | 20 extra inputs: per seat, material, the order-weighted threat to its pieces before its next move, the best capture now (mover) or the biggest loss the mover can inflict, king attacked, pawns near promotion. |
| `engine/src/nnue.rs` | The network (unchanged layout plus two options): a "points head" and a wider global input read off the first layer. |
| `engine/src/search.rs` | max^n search (each player maximizes its own expected score; no paranoid or opponent model). `resign_multi` switched off for this engine, so like greedy it only claims the win with two players left. |
| `engine/src/bin/nnarena.rs` | Arena: model vs 3 greedy, openings x 4 seats, writes per-game results and training positions. |
| `nn/train.py` | Trains the network on arena positions (torch, CPU) and exports it. |
| `nn/loop.py` | The learning loop: train on all positions so far, play 160 new games, repeat. |

## Network

1600 piece-square features per seat -> **h = 32** (the small first layer), the four seats'
accumulators clipped to [0, 1] (mover first) + 36 globals -> 32 -> 32 -> 4 outputs, about 58k
parameters. The 4 outputs are each seat's expected future points F; the search's pairwise
logit is `beta * ((s_i + F_i) - (s_j + F_j))` with the current scores s. Loss: BCE on who
finished above whom (6 pairs) + 0.1 x MSE on F, where the F target is each seat's points still
to come, discounted 0.97 per ply. Every position is used from the mover's view, greedy's moves
included.

## Arena protocol

160 games per round: 40 openings (4 random plies from a seed, redrawn if a king hangs), the
model in each of the 4 seats once per opening, so colours are balanced exactly. One greedy bot
per seat (each with its own table). Model: 50 ms per move. Greedy: fixed depth 4, about 8 ms
per move in Rust (0.1 to 0.8 s in Python).

## Build and run (Linux / WSL / Colab)

```bash
cd engine && RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --bin nnarena && cd ..
python nn/check_greedy.py --games 3 --depth 4           # the port matches bot/greedy.py
python nn/loop.py --rounds 8                             # round 0 = untrained, then learn
./engine/target/release/nnarena arena --net nn/runs/r8/net.bin --games 160 --opening-seed 1000 --ms 50
```

`--test greedy` plays greedy against itself (1.50 by symmetry), `--greedy-ms 50` gives greedy
the same clock as the model (iterative deepening) instead of depth 4.

Results: see `nn/RESULTS.md`.
