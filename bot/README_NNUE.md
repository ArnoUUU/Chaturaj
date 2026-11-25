# NNUE + max^n engine: model and training

How the current engine is built and trained, end to end: what the network is, how the search
uses it, how the network is trained (first on recorded games, then on its own games), how a
new network is judged, and what has happened so far. The older MCTS + MLP engine is described in
[README.md](README.md). Run-by-run notes are in `logs/nnue_selfplay_plan.md` (local: `logs/` is not committed), and the
compiled core in [../engine/README.md](../engine/README.md).

## Contents

1. [The engine at a glance](#1-the-engine-at-a-glance)
2. [The network](#2-the-network)
3. [The search](#3-the-search)
4. [Training, stage 1: recorded games](#4-training-stage-1-recorded-games)
5. [Training, stage 2: self-play](#5-training-stage-2-self-play)
6. [How a network is judged](#6-how-a-network-is-judged)
7. [What is logged, and the standard curves](#7-what-is-logged-and-the-standard-curves)
8. [Runs so far](#8-runs-so-far)
9. [Known problems and next steps](#9-known-problems-and-next-steps)
10. [Commands](#10-commands)
11. [Files](#11-files)

---

## 1. The engine at a glance

```
board + scores ─► NNUE value network ─► 4 utilities (one per seat) ─► max^n search ─► move
                  (who finishes above whom)                          (each seat maximises
                                                                      its own utility)
```

- **The network** looks at a position and predicts, for every pair of seats, the chance that
  one finishes above the other on points. From those six numbers each seat gets a
  **utility**: its expected chess.com free-for-all score, between 0 and 1. The four
  utilities always add up to 2.
- **The search (max^n)** looks ahead a few moves. At each move, the player to move picks the
  move that is best *for that player's own utility*. This is how four-player games are
  searched: there is no single "my side vs their side" score as in chess.
- **Training** has two stages: supervised learning on recorded games (human games scraped
  from chess.com and games from the earlier MCTS + MLP engine), then self-play, where the
  engine plays itself and learns from the results.

| | Now |
|---|---|
| Network in use | **nnue-v1** (preset `small`: 444,422 weights, 1.8 MB) |
| Search | max^n, depth 5, selective below the first move, check extensions, quiescence |
| Strength | max^n + nnue-v1 beats MCTS + MLP: pairwise score 0.75 to 0.93 in every check so far |
| Self-play | has **not** yet produced a network that beats nnue-v1 (section 8) |
| Code | Python for training; the rules, the network and the search are compiled in Rust (`engine/`) |

---

## 2. The network

### 2.1 Inputs

**Pieces: 1,600 on/off features, seen from each of the four seats.**
A feature is (owner, piece type, square):

| Part | Values | Count |
|---|---|---:|
| Owner, relative to the viewing seat | me, next seat, across, previous seat, dead (an eliminated player's piece) | 5 |
| Piece type | pawn, knight, bishop (elephant), rook (boat), king | 5 |
| Square, rotated so the viewing seat sits at the bottom | a1 … h8 | 64 |
| | | **1,600** |

A position has at most 32 pieces, so at most 32 features are on per seat.

**Globals: 16 numbers**, with seats in order from the player to move (me, next, across,
previous):

| # | Feature | Scale |
|---|---|---|
| 1–4 | each seat still in the game (1/0) | |
| 5–8 | each seat's points | ÷ 30 |
| 9–12 | each seat's points minus the mean | ÷ 20 |
| 13 | how often this exact position has occurred | ÷ 3 |
| 14 | game progress (turn) | ÷ 400, capped at 1 |
| 15 | dead kings on the board | ÷ 3 |
| 16 | players still in | ÷ 4 |

### 2.2 Architecture

```
for each seat p (R, B, Y, G):  acc_p = clip( W_ft · features_p + b_ft , 0, 1 )     H numbers
                               (W_ft shared by all four seats: 1,600 × H)

x  = [ acc_mover, acc_next, acc_across, acc_previous, 16 globals ]                    4H + 16
h1 = crelu( W1 x + b1 )             crelu = clipped ReLU, min(max(x, 0), 1)
h2 = crelu( W2 h1 + b2 )            (deep presets: h_k = h_{k-1} + crelu(W_k h_{k-1} + b_k),
...                                  residual, when widths match)
out = W_out h_last + b_out          6 logits, one per pair of seats
```

- **Why "NNUE"** (efficiently updatable): the first layer is a sum of the weight rows of the
  pieces on the board. A move changes 2 to 4 features, so the search updates the four
  accumulators by adding and subtracting a few rows instead of recomputing them. Only the
  small layers after it run for every position.
- **Outputs:** 6 logits for the pairs (me, next), (me, across), (me, previous),
  (next, across), (next, previous), (across, previous). `sigmoid(logit)` = P(the first
  seat finishes above the second), a tie counting as half.
- **Utilities:** seat i's utility = the mean of its three "finishes above" probabilities
  (using 1 − P for pairs where it is the second seat). Each is in [0, 1], and the four sum
  to 2. A utility of 1 means certainly first, 0 certainly last.

### 2.3 Sizes (presets)

| Preset | Accumulator H | Hidden layers | Weights | float32 size | Status |
|---|---:|---|---:|---:|---|
| `small` | 256 | 32, 32 | 0.44 M | 1.8 MB | **in use (nnue-v1)** |
| `medium` | 512 | 7 × 512, residual | 3.5 M | 14 MB | built and tested, not trained yet |
| `large` | 1,024 | 7 × 1,024, residual | 12.2 M | 49 MB | built and tested, not trained yet |
| `xl` | 2,048 | 7 × 2,560, residual | 63.6 M | 255 MB | built and tested (up to the 300 MB limit) |

The big presets cost search speed: every evaluated position runs all 7 hidden layers. The
Rust core batches the positions one move apart (all children of a node) so the weights are
read once per batch, which is what makes them usable at all.

### 2.4 Training loss

**Labels:** after a game ends, every recorded position of it gets the game's real result,
for each pair of seats: 1 if the first finished with more points, 0.5 if equal, 0 if fewer.
**Loss:** binary cross-entropy between the 6 logits and those 6 labels, averaged ("pairwise
loss"). A network that knows nothing scores about 0.69; one that predicts the score leader
in a position is right about 68% of the time on decided pairs.

### 2.5 The same network in three places

| Implementation | Used for |
|---|---|
| PyTorch `NNUE` (`bot/nnue.py`) | training (batched, recomputes everything) |
| NumPy `NumpyNNUE` + `Accumulator` (`bot/nnue.py`) | the reference for search, incremental updates |
| Rust `chaturaji_engine.Nnue` (`engine/src/nnue.rs`) | the search used in practice |

Weights move between them as an `.npz` file (`nnue.npz`): `ft_w` (1,600 × H), `ft_b`,
`l1_w`, `l1_b`, …, `residual` (one flag per layer), `out_w` (6 × last), `out_b`. Tests
check that all three give the same outputs.

---

## 3. The search

### 3.1 max^n

Each position's value is the vector of four utilities. At a node, the player to move takes
the child with the highest utility **for that player**; that child's whole vector is passed
up. There is no alpha-beta pruning: with four independent players there is no bound to
prune on. The one cut: if a move already gives the mover utility 1 (finishes first for
certain), the other moves aren't searched.

### 3.2 What is searched

| Part | What it does | Setting now |
|---|---|---|
| **Depth** | moves searched ahead (each move = one player's turn) | 5 |
| **Selective depth** | from ply `full_plies` on, a node searches only the `width` children with the best quick evaluation for the player to move, plus the move the table remembers; every line still goes the full depth | width 4 after the first ply |
| **Check extensions** | a move that newly attacks an active king is searched one move deeper, up to `check_ext` times along a line, so chains of checks (where predator-and-prey play happens) are followed | 3 |
| **Quiescence** | at the end of a line, keep playing point-scoring captures (and, on the first step, checks) until the position is quiet, so a leaf isn't judged in the middle of an exchange | 2 steps |
| **Move order** | the table's move first, then captures by points, then checks, then the rest | |
| **Transposition table** | positions already searched (board, player to move, scores, who is out), with their depth, value and best move | up to 1,000,000 positions |
| **Fixed depth or time** | self-play uses a fixed depth so results don't depend on CPU load; matches can use a time per move (iterative deepening) | fixed depth 5 |

**Why selective depth:** a full depth-5 search took 17 to 86 s a move on this laptop
(about 30 moves per position, so each extra move multiplies the work by about 30). With
width 4 it takes about 0.3 to 0.7 s.

**What is kept between moves:** the transposition table, for the whole game (emptied only
when it gets too big). The search tree itself isn't kept. After four moves (one round), most
saved results come from a shallower search than the new one needs, so they mostly help by
putting the remembered best move first; exact results are reused only where the saved depth
is enough.

### 3.3 Preferences: shorter games and early eliminations

Two small adjustments to the value of a move as it is passed up the search (both off in
the class defaults, on in self-play from run 4):

| Preference | Rule | Setting | Effect |
|---|---|---|---|
| **Shorter games** (`discount`) | every move pulls all four values towards the average 0.5: `v = 0.5 + discount × (v − 0.5)` | 0.9985 | a player who is ahead prefers the line that ends sooner; one who is behind prefers the longer one (keeps its pieces) |
| **Early eliminations** (`elim_bonus`, `elim_horizon`) | a move that eliminates a player adds `elim_bonus × (1 − turn / elim_horizon)` (not below 0) to the mover's value | 0.008, horizon 200 turns | eliminating someone is preferred, more so early in the game; nothing after turn 200 |

**Why so small: own pieces come first.** Losing a piece must always matter more than either
preference. What the network gives a piece was measured on 121 self-play games (change in
the victim's utility when it is captured, median):

| Piece lost | Points | Victim's utility drop |
|---|---:|---:|
| pawn | 1 | about 0.010 (mean 0.016) |
| knight or king's capture worth 3 | 3 | about 0.03 (mean 0.044) |
| bishop or rook | 5 | about 0.06 |
| (eliminating a player: the eliminator's gain) | | about 0.07 |

So both preferences stay below a pawn:

- the elimination bonus is at most 0.008, less than a pawn;
- the discount changes a value by at most 0.5 × (1 − 0.9985^10) ≈ 0.007 over the longest
  line searched (5 moves + 3 check extensions + 2 quiescence steps), also less than a pawn.

They break near-ties between moves; they never make a player give up a piece. Both are
measured in each run (game length and the ply of the first elimination are logged).
The pruning cut in 3.1 is judged on the value *before* the preferences.

---

## 4. Training, stage 1: recorded games

`bot/train_nnue.py` → **nnue-v1**.

| | |
|---|---|
| Data | 2,880 scraped human games (chess.com) + 7,272 games from the MCTS + MLP league run |
| Positions | every 2nd position of each game, from each game's real final result: 722,878 for training, 39,912 for validation |
| Validation split | by game (5% of games), so no game is in both |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-5, batches of 1,024 |
| Epochs | 8, keeping the epoch with the lowest validation loss |
| Time | about 10 minutes on the laptop CPU |

| Epoch | Training loss | Validation loss | Pair accuracy |
|---:|---:|---:|---:|
| **1 (kept)** | 0.531 | **0.478** | **0.749** |
| 2 | 0.384 | 0.515 | 0.738 |
| 4 | 0.308 | 0.591 | 0.730 |
| 8 | 0.267 | 0.656 | 0.729 |

Score-lead baseline (whoever leads on points now): 0.684 pair accuracy. **Note the
overfitting after one epoch**: the training loss keeps falling while the validation loss
rises. Every position of a game shares that game's label, so the network can learn to
recognise *games* instead of judging positions. This is the main open problem (section 9).

---

## 5. Training, stage 2: self-play

`bot/selfplay_nnue.py` (the loop) and `bot/selfplay_worker.py` (games, run in worker
processes with NumPy and Rust only, so many fit in memory).

### 5.1 One iteration

```
                  ┌───────────────────────────────────────────────────────────────┐
current weights ─►│ 1. self-play: 24 games in parallel (12 workers), every seat    │
                  │    max^n with the current weights                              │
                  │ 2. label every position with its game's final result           │
                  │    every 10th game ─► held-out self-play validation set        │
                  │    the rest       ─► replay buffer (latest 400,000 positions)  │
                  │ 3. train: 40 steps of 1,024 positions                          │
                  │    = half from the replay buffer, half from the stage-1 data   │
                  │      (minus 1 game in 20, held out for validation)             │
                  │ 4. measure: both validation losses, learning rate, grad norm   │
                  │ 5. save the weights ─► next iteration's games use them         │
                  └───────────────────────────────────────────────────────────────┘
every ~18 min: a check (section 6) ─► keep, mark as new best, or go back to the best
```

- **Variety in self-play:** each game opens with 4 to 8 random moves, then a random move
  with 5% probability for the first 30 moves. Games stop at the end of the game or after 400
  turns.
- **No training until the buffer holds 20,000 positions** (about 8 iterations at depth 5),
  so the first steps aren't spent on a handful of games.
- **Why mix in the stage-1 data:** so the network can't drift away from real play while it
  learns from its own games.
- **Optimiser:** AdamW, weight decay 1e-5, gradients clipped at norm 1. The optimiser state
  is saved with the model and reloaded when a run continues another self-play run.

### 5.2 Settings

| Setting (flag) | Runs 1–2 | Run 3 | Run 4 (now) | Default |
|---|---|---|---|---|
| Starting network (`--init`) | nnue-v1 | nnue-v1 | nnue-v1 | nnue-v1 |
| Depth (`--depth`) | 2 | 5 | 5 | 2 |
| Selective width / full plies (`--width`, `--full-plies`) | off | 4 / 1 | 4 / 1 | off / 2 |
| Quiescence (`--qdepth`) | 2 | 2 | 2 | 2 |
| Check extensions (`--check-ext`) | 3 | 3 | 3 | 3 |
| Shorter-games discount (`--discount`) | off | off | 0.9985 | 0.9985 |
| Elimination bonus / horizon (`--elim-bonus`, `--elim-horizon`) | off | off | 0.008 / 200 | 0.008 / 200 |
| Random opening / exploration | 4–8 moves / 5% for 30 moves | same | same | same |
| Workers (`--workers`) | 10 | 12 | 12 | CPUs − 3, max 12 |
| Games per iteration (`--games`) | 48 | 24 | 24 | 48 |
| Steps per iteration (`--steps`), batch (`--batch`) | 80 × 1,024 | 40 × 1,024 | 40 × 1,024 | 120 × 1,024 |
| Self-play share of a batch (`1 − --base-frac`) | 50% | 50% | 50% | 50% |
| Replay buffer (`--buffer`), minimum (`--min-buffer`) | 400k / 20k | same | same | same |
| Learning rate (`--lr`) | 3e-4 | 2.5e-4 | 2.5e-4 | 3e-4 |
| Checks every (`--check-min`) | 60 min | 18 min | 18 min | 60 min |
| Check games vs best / vs MCTS (`--check-games`, `--yard-games`) | 48 / 40 | 24 / 20 | 24 / 20 | 48 / 40 |
| Validation (`--val-every`, `--val-base`, `--val-selfplay-every`) | none | 1 game in 20 (20k positions) / 1 game in 10 | same | same |
| Length (`--hours` or `--until`) | 5 h + 1 h | 8 h (stopped at 2.7 h) | 8 h | 5 h |

---

## 6. How a network is judged

**Pairwise score:** chess.com's free-for-all rule. Every game counts as three one-on-one
results for the tested seat (beat, tie or lost to each other seat on final points: 1, 0.5,
0), averaged. 0.5 = as good as the opponents. Results are given ±2 standard errors (about a
95% range).

| Check | When | Games | Decision |
|---|---|---|---|
| **vs best** | at the start of the run (vs MCTS only), every ~18 min, at the end | 24: the current weights in one seat, the best weights so far in the other three, each seat in turn, both with the same search, after the same 4 random opening moves | ≥ 0.5: **new best** (saved). Clearly worse (score + 2 SE < 0.5): **go back to the best weights**. Otherwise: keep training |
| **vs MCTS + MLP** | same times | 20: max^n with the current weights in one seat, MCTS (32 simulations) with the league's final MLP in the other three | none; a fixed yardstick, so progress can be compared across runs |

At the end of a run, if the final weights are clearly worse than the best, the best are
saved as the run's model instead. With 24 games, ±2 SE is about ±0.15, so a check only
decides large differences; the equal-time matches on the Tillicum cluster
(`cluster/maxn_match.slurm`) are the precise comparison.

---

## 7. What is logged, and the standard curves

Every run writes:

| Where | What |
|---|---|
| `logs/<label>/games.jsonl` | every game: seed, opening, final scores, how it ended, plies, every move, every capture of an active piece (ply, attacker, victim, points), every check, every elimination; check games too |
| `logs/<label>/log_*.jsonl` | every iteration and check (below) |
| `logs/<label>/curves.png` | **the standard figure**, redrawn after every check and at the end |
| `models/<label>/` | `model.pt`, `nnue.npz`, `meta.json` (settings, every iteration and check), `curves.png`, a snapshot at each new best |
| `logs/nnue_selfplay_*.log` | the console |

Logged every iteration: training loss; **validation loss on held-out stage-1 games**
(nnue-v1 trained on those games, so this shows drift away from real play rather than
generalisation); **validation loss on held-out self-play games** (never trained on: the
overfitting check); learning rate; optimiser steps; gradient norm; games, positions,
buffer size; endings; mean game length; **ply of the first elimination**; **prey share**
(share of a game's captures that hit its most-captured seat: 0.25 if spread evenly, towards
1 if one seat is hunted); **gang rate** (share of games where two players captured from the
same victim within 12 plies).

**The standard figure** (`bot/plot_run.py`, six panels against hours):

1. Loss: training, both validation losses, and the reverts to the best weights
2. Learning rate
3. Gradient norm
4. Checks: pairwise score ±2 SE vs best and vs MCTS + MLP, with the reverts marked
5. Predator and prey: prey share and gang rate
6. Game length and first elimination

Draw it for any run with `python -m bot.plot_run models/<label>` (older runs leave the
panels they didn't log empty).

---

## 8. Runs so far

| Run | Settings | Result |
|---|---|---|
| nnue-v1 (stage 1) | section 4 | val loss 0.478, pair accuracy 0.749; max^n + nnue-v1 beats MCTS + MLP 0.82 ± 0.10 |
| 1: self-play 5 h | depth 2 | every hourly check clearly worse than nnue-v1 (0.25 to 0.36), back to nnue-v1 four times; final weights tied (0.51) |
| 2: continuation to 06:00 | depth 2 | final check 0.21 ± 0.10, clearly worse, back to the start |
| 3: depth 5 | depth 5, width 4, lr 2.5e-4, checks every 18 min, validation on | first ~1.3 h only filled the buffer (no training); then 0.48 (kept training), 0.29 (clearly worse: back), 0.41; held-out self-play validation loss stayed at 0.48 to 0.54 while the training loss fell from 0.42 to 0.37. Stopped at 2.7 h to add the preferences |
| 4: depth 5 + preferences | run 3 + shorter-games discount and elimination bonus | running |

Predator-and-prey numbers have stayed flat in every run (prey share about 0.35 to 0.42,
gang rate 0.75 to 1.0): ganging up on one player is already what this search does.

---

## 9. Known problems and next steps

1. **Overfitting to games (the main problem).** In stage 1 the validation loss was best
   after one epoch; in self-play the training loss falls while the held-out self-play loss
   doesn't, and the network gets worse in play. All positions of a game share one label,
   so they are not independent examples, and each game is trained on many times. Options:
   - fewer positions per game (every 4th, or a few random ones), and fewer steps per new game;
   - more weight decay or dropout;
   - bootstrapped labels: mix the game result with the search's own value of the position
     (or the value a few moves later), which differs from position to position;
   - many more games per update (Tillicum).
2. **Noisy checks:** 24 games give ±0.15. Use more games when a check is close, or the
   cluster for exact comparisons.
3. **Checks are slow at depth 5** (10 to 12 minutes, about half the run time). Cheaper
   options: depth-5 checks less often, or the MCTS yardstick only every other check.
4. **Labels only see the final points**, not how the game was won. Shaping the labels
   (for example discounting results that come many moves later) would teach the network
   the same preferences as section 3.3, beyond the search's horizon.
5. **Larger networks** (medium to xl) are ready but not trained; they only make sense once
   problem 1 is solved and there is much more data.
6. **The user's notes on playing style** (in preparation) will add further ideas; each one
   should become a setting in the tables above and a line in section 8.

---

## 10. Commands

Run from the repository root. Build the compiled core first ([../engine/README.md](../engine/README.md)).

Stage 1, train a network on recorded games:

```powershell
python -m bot.train_nnue --tag nnue-v1 --preset small --epochs 8 --selfplay logs/2026-11-23_0144_league-5h_mlp-e512-w1024x512x256_puct-s32_timed_cpu/games.jsonl --cache logs/nnue_cache.npz
```

Stage 2, self-play (run 4's settings):

```powershell
python -u -m bot.selfplay_nnue --hours 8 --depth 5 --width 4 --full-plies 1 --qdepth 2 --check-ext 3 --discount 0.9985 --elim-bonus 0.008 --elim-horizon 200 --lr 2.5e-4 --check-min 18 --workers 12 --games 24 --steps 40 --check-games 24 --yard-games 20 --yard-workers 3 --tag nnue-selfplay-d5-prefs
```

Continue a run (keeps its optimiser state), or train until a clock time:

```powershell
python -u -m bot.selfplay_nnue --init models/<label> --until 06:00
```

Draw the standard curves for a run:

```powershell
python -m bot.plot_run models/<label>
```

Play max^n against MCTS + MLP:

```powershell
python -m bot.maxn models/<label> --opponent models/opponent_pool/league_model.pt --games 40
```

Tests (rules, network, search; Rust against Python):

```powershell
python -m pytest -q tests/test_nnue.py tests/test_rules_core.py
```

Cluster jobs (build, train sizes, matches): [../cluster/README.md](../cluster/README.md).

---

## 11. Files

| File | Contents |
|---|---|
| `bot/nnue.py` | the network: features, PyTorch model, NumPy reference, presets, weight files, loss, labels |
| `bot/train_nnue.py` | stage 1: positions from recorded games, training, validation |
| `bot/maxn.py` | the max^n search (Rust when available, Python reference), CLI for matches |
| `bot/selfplay_nnue.py` | stage 2: the self-play loop, checks, logging |
| `bot/selfplay_worker.py` | self-play games, check games, MCTS yardstick games (worker processes) |
| `bot/plot_run.py` | the standard training-curves figure |
| `bot/rules_core.py` | moves between the Python rules and the Rust core; rules check and benchmark |
| `engine/src/rules.rs`, `nnue.rs`, `search.rs`, `lib.rs` | the compiled rules, network, search, Python bindings |
| `tests/test_nnue.py`, `tests/test_rules_core.py` | tests |
| `logs/nnue_selfplay_plan.md` | run-by-run notes and decisions |
