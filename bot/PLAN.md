# Chaturaji Bot — Engine & Model Plan

An engine for 4-player Chaturaji (chess.com rules, as implemented in `chaturaji/`) that
picks moves with **Monte Carlo Tree Search (MCTS)**. Positions are evaluated by a
**deep learning model: an encoding layer that reads the game state, followed by three
fully connected layers, which output an evaluation**. The model learns first from real
chess.com games (via `scraper/`) and then from self-play.

Contents
1. Goals and constraints
2. What makes Chaturaji different (and how the design handles it)
3. Game-state encoding
4. Evaluation network (encoding layer + 3 fully connected layers)
5. MCTS design
6. Training pipeline
7. Evaluating the engine
8. Performance engineering
9. Code layout and changes to the current `bot/`
10. Milestones with acceptance criteria
11. Recommendations summary
12. Risks and open questions
13. Hyperparameter reference

---

## 1. Goals and constraints

**Goal:** finish with the most points in 4-player free-for-all Chaturaji. The aim is points,
not survival: a player whose king is captured can still win on points.

**Targets**
- Strong play at blitz speed (chess.com `1 | 5`: 1 minute plus 5 seconds per move).
  That means a few hundred to a few thousand MCTS simulations per move on a laptop CPU.
- Trainable on one machine. The current machine has a CPU-only PyTorch build. A
  consumer GPU, if added later, should speed up training without code changes.
- Every part testable in isolation: encoding, network, search, training loop.

**Fixed rules the engine relies on** (all implemented and checked against 7 real games):
8x8 board; turn order R → B → Y → G; no check restriction (kings can be captured);
pawns move 1 square and promote to a rook; capture points P1 N3 K3 B5 R5; check bonuses
+1 (double) / +5 (triple); dead pieces worth 0 except a dead king (3); the last player
standing gets +3 per dead king left; threefold repetition gives remaining players +2.

---

## 2. What makes Chaturaji different (and how the design handles it)

| Property | Consequence | Design response |
|---|---|---|
| 4 players, not 2 | No single "my eval = −their eval"; the value of a position differs per player | Evaluation outputs a **4-vector** (one value per seat). MCTS backs up vectors (max^n). |
| Points decide the winner | Material captured so far matters as much as position | Current scores go **into the encoding**. The value target is the final points outcome, not survival. |
| Kings can be captured, no check rule | Blunders are common, tactics are short and sharp | MCTS with enough simulations catches 1–3 ply tactics. Priors favour captures early in training. |
| Board layout has 90° rotational symmetry | Every seat sees the same start position from its own side | **Canonical view**: rotate so the player to move is always in Red's seat. One network plays all seats and gets 4× more training data. |
| Eliminated players' pieces stay as dead blockers | The board fills with inert pieces | Dead pieces get their own channel ("X") in the encoding. |
| Games end by last king or repetition | Repetition can be forced as a points-preserving tactic | The repetition count is an input, and repetition is a terminal state with exact scoring in search. |
| Real games are short on the clock | Search must be cheap per node | A small, fast MLP evaluator (this plan) instead of a deep ResNet. Batched inference. |

---

## 3. Game-state encoding

All encoding is from the **mover's point of view** (canonical view, as in `bot/features.py`):
- Rotate the board `k` quarter turns, where `k` = the mover's seat index (R=0, B=1, Y=2, G=3).
  The square map is `(r, c) → (7 − c, r)` applied `k` times.
- Relabel owners relative to the mover: **0 me, 1 next, 2 across, 3 previous, 4 dead (X)**.
- Vectors over seats (scores, alive flags, values) are rolled the same way:
  `relative[i] = absolute[(k + i) mod 4]`.

The encoder input has two parts.

### 3.1 Sparse piece-square features (the "board")
One binary feature per (relative owner, piece type, square):

`index = ((owner * 5) + type) * 64 + square`, giving 5 owners × 5 types × 64 squares
= **1600 features**. About 32 or fewer are active in any position.

This is the input to the **encoding layer** (§4.1). It's equivalent to the 25×8×8 planes
already produced by `features.encode()`, flattened, so both encoders can share code.

### 3.2 Dense global features (G = 16)
| # | Feature | Scaling |
|---|---|---|
| 4 | Alive flag per relative seat | 0/1 |
| 4 | Current score per relative seat | ÷ 30 |
| 4 | Score minus the mean score, per relative seat | ÷ 20 |
| 1 | Repetition count of the current board | ÷ 3 |
| 1 | Game progress (turn) | turn ÷ 400, capped at 1 |
| 1 | Number of dead kings on the board (each worth 3 to someone) | ÷ 3 |
| 1 | Number of players still alive | ÷ 4 |

**Recommendation:** start with exactly these. Possible later additions, each tested
on/off: "which of my pieces are attacked" and "which enemy kings I attack" (8 more
dense features from `is_attacked_by`). They cost move-generation time per evaluation, so
add them only if they clearly improve play.

### 3.3 What is not encoded
- **Clock times:** self-play has no clock. Supervised data from chess.com does, but
  leaving it out keeps the model about the position.
- **Move history** beyond the repetition count: the game is fully observable, so the
  current position plus scores is enough.

---

## 4. Evaluation network (encoding layer + 3 fully connected layers)

```
 sparse piece-square features (1600, ~32 active)      dense global features (16)
                │                                              │
     ┌──────────▼───────────┐                                  │
     │  ENCODING LAYER       │  EmbeddingBag(1600 → 512, sum)   │
     │  + bias, ClippedReLU  │  ("feature transformer")         │
     └──────────┬───────────┘                                  │
                └──────────────── concat (512 + 16 = 528) ◄────┘
                                   │
                  FC1  528 → 1024, LayerNorm, ReLU, Dropout 0.1
                                   │
                  FC2 1024 →  512, LayerNorm, ReLU, Dropout 0.1
                                   │
                  FC3  512 →  256, LayerNorm, ReLU
                                   │
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
   VALUE head 256→4         SCORE head 256→4          POLICY head 256→4096
   softmax: P(top score)    final score margin        move-prior logits
   per relative seat        per relative seat         (from*64+to, canonical)
          └───────────── the EVALUATION ─────┘        (used by MCTS for priors)
```

### 4.1 Encoding layer
- **What it is:** a learned embedding of each (owner, piece, square) feature, summed over
  the pieces on the board (`nn.EmbeddingBag(1600, 512, mode="sum")`). It's mathematically
  the same as a linear layer on the 1600-dim one-hot input, but it only touches the ~32
  active rows.
- **Why this encoder (recommended):**
  1. **Speed:** MCTS evaluates thousands of positions per move. A sparse sum is around
     16k additions, versus millions for convolutions.
  2. **Incremental updates (later optimization):** a move changes 2–3 features, so the
     512-dim accumulator can be updated by add/subtract instead of being recomputed. This
     is the trick NNUE chess engines use, and it fits a CPU-bound Python engine well.
  3. **It fits the requested architecture directly:** encoding layer → 3 FC layers.
- **Activation:** ClippedReLU (`clamp(x, 0, 1)`) on the encoding output keeps values
  bounded, which helps if we quantize later. Plain ReLU is fine to start.
- **Alternative encoder (for comparison):** a small convolutional stem
  (2× Conv3x3, 64 channels, then flatten to 4096 and a Linear to 512). It can spot local
  patterns (pawn chains, king safety) with less data, but it is ~10× slower per
  evaluation. Recommendation: keep the existing ResNet (`bot/model.py`) as a **baseline
  to measure against**, not as the main model.

### 4.2 The three fully connected layers
| Layer | Shape | Params | Notes |
|---|---|---|---|
| Encoder | 1600 → 512 | 819,712 | sparse, sum-pooled |
| FC1 | 528 → 1024 | 541,696 | widest layer, mixes board and global features |
| FC2 | 1024 → 512 | 524,800 | |
| FC3 | 512 → 256 | 131,328 | shared representation for all heads |
| Value head | 256 → 4 | 1,028 | |
| Score head | 256 → 4 | 1,028 | |
| Policy head | 256 → 4096 | 1,052,672 | |
| **Total** | | **≈ 3.07 M** | ≈ 2.3 M multiply-adds per evaluation |

- **Normalization:** LayerNorm after each FC layer, not BatchNorm. Batch statistics
  behave badly with MCTS's batch-size-1 inference and with self-play data whose
  distribution changes.
- **Regularization:** dropout 0.1 on FC1/FC2 during training only, plus weight decay 1e-4.
- **Residual option:** if training is unstable, add a skip connection around FC2
  (1024 → 512 → 1024 style). Start without it.
- **Init:** Kaiming-uniform for ReLU layers. Value/score heads initialized near zero, so
  early evaluations start neutral (uniform 0.25 per seat).

### 4.3 Outputs: what the "evaluation" means
- **Value head (main evaluation):** 4 logits → softmax = probability that each relative
  seat finishes with the top score. Ties share the probability. These 4 numbers are the
  evaluation MCTS backs up.
  - For display, a single number for the mover is `eval = 4 · P(me) − 1`, ranging from
    −1 (hopeless) through 0 (average, 25%) to +3 (certain win).
- **Score head (auxiliary, recommended):** predicted final `score − mean score` per seat
  (÷20). It gives a dense learning signal (points happen all game; wins happen once per
  game) and a tie-breaker for the search.
- **Policy head (recommended):** move priors for MCTS (§5.3). It shares the encoder and
  FC1–FC3, so it costs one extra 256→4096 layer.
  - **If the evaluation network must be value-only**, MCTS still works with heuristic
    priors (§5.3, option B), and the policy head can live in a separate small network.
    Recommendation: share it. It's nearly free and greatly improves search efficiency.

### 4.4 Losses
`L = CE(value, outcome_share) + λ_s · MSE(score, final_margin) + λ_p · CE(policy, target) + wd‖θ‖²`

with λ_s = 0.5 and λ_p = 1.0. The policy target is MCTS visit counts (self-play) or the
human move (supervised). Illegal moves are masked to −∞ before the softmax, so the
network never learns to put probability on them.

---

## 5. MCTS design

### 5.1 Tree structure
Each node is one position (a `ChaturajiState`) with the player to move `p`:
- `moves[]`, `priors P[a]` (from the policy head or heuristics)
- visit counts `N[a]`, value sums **`W[a][4]`** in *absolute* seat order
- the evaluation vector `V[4]` from the network (or the exact outcome if terminal)
- children are created lazily when first visited.

The tree stores states, not move lists. This is fine because `apply_move` is pure.
Memory: ~2 KB per node, so 50k nodes is about 100 MB.

### 5.2 Selection: PUCT, maximized for the player to move (max^n)
```
a* = argmax_a  Q_p(a) + c_puct · P(a) · sqrt(N_parent) / (1 + N(a))
Q_p(a) = W[a][p] / N[a]      (the mover's own entry of the mean value vector)
```
- `c_puct`: start at **1.5**, tune between 1.0 and 3.0.
- **First-play urgency (FPU):** an unvisited child's Q is the parent's value for `p`
  minus 0.1. This keeps the search from spreading over every move at low simulation
  counts.
- **Why max^n:** every player maximizes their own expected outcome. This matches
  free-for-all play on chess.com better than "paranoid" search (which assumes all three
  opponents target you).
  - Recommendation: make the opponent model a pluggable strategy and test **paranoid**
    as an alternative. Paranoid can be safer in king-hunting positions.

### 5.3 Priors (expansion)
- **Option A (recommended):** policy-head softmax over legal moves.
- **Option B (no policy head / bootstrap):** heuristic priors from the move itself:
  `prior ∝ exp( 0.6·captured_value + 1.5·captures_king + 0.4·gives_check
  + 0.3·promotes − 0.5·moves_onto_attacked_square )`. Use this during the first
  self-play iterations and as a baseline.
- **Root noise (self-play only):** `P ← 0.75·P + 0.25·Dirichlet(α)`, with α ≈ 0.3.
  There are around 20–40 legal moves, so α ≈ 10 / typical number of moves.

### 5.4 Evaluation and backup
- At a new leaf: run the network → 4-vector in relative seats → **rotate to absolute
  seats** → add it to `W` along the path (every node's `W` holds all 4 seats).
- **Terminal leaves** (last king, threefold repetition, or ply cap) use the **exact
  outcome**, computed from the final scores *including* the end-of-game points
  (`outcome_vectors` in `features.py`).
- **Points already won are not lost:** because current scores are part of the encoding,
  the network sees captures that already happened. No separate reward signal is needed
  inside the tree.

### 5.5 Transpositions, tree reuse, and caching
- **Tree reuse (implemented):** after a move, keep the chosen child's subtree as the new
  root.
- **Evaluation cache (recommended):** an LRU dictionary from `(board_key, turn % 4,
  scores, eliminated)` to the network output (~100k entries). Repetition shuffles hit
  this often.
- **Transposition table (not yet):** sharing nodes between paths complicates max^n
  backup and repetition handling. Revisit only if profiling shows many duplicate nodes.

### 5.6 Batched / parallel search
Single-position inference wastes most of the CPU/GPU time.
- **Leaf parallelism with virtual loss:** collect `B` leaves (B = 16–64) per network
  call. Each selected path temporarily gets `N += 1` and `W[p] −= 1` (virtual loss), so
  the next selection explores a different branch. Then evaluate all leaves in one batch
  and back them up.
- **Self-play parallelism:** run `n` games in worker processes. A single inference
  server process batches their leaf requests. This is the main throughput win.

### 5.7 Move selection and time management
- **Self-play:** temperature τ = 1 (sample proportional to visits) for the first 30
  plies, then τ → 0 (most visits).
- **Competitive play:** always the move with the most visits. Use a fixed simulation
  budget per move, or time-based: `min(remaining/20 + increment·0.8, hard cap)`.
- **Early stop:** stop searching when the best move can't be overtaken by the remaining
  budget.

### 5.8 Search tests (acceptance)
- Finds a free king capture that ends the game (already tested).
- Prefers capturing a rook (+5) over a pawn (+1) when both are safe.
- Avoids moving its king onto an attacked square when an equal alternative exists.
- Recognizes a draw by repetition when ahead on points (it locks in the lead and +2).

---

## 6. Training pipeline

### Phase 0: data
1. **Scraped human games** (`scraper/game_scraping.py`, `scraper/pgn4.py`). Replay
   each PGN4 through the engine, which validates every move. Each position yields
   `(encoding, human move, final outcome, final margins)`.
   - Filter: blitz (`1 | 5`) and longer, all four players rated ≥ 1800, no aborted games.
   - Target size: **2,000–10,000 games ≈ 0.3–1.5 M positions.**
2. **Self-play games** from MCTS (Phase 2).

### Phase 1: supervised bootstrap (recommended before self-play)
Self-play from random weights is very slow for a 4-player game with long, aimless random
games. Bootstrapping from human games gives a sensible starting policy and value.
- Train the evaluator on scraped positions with the policy target = the human move
  (one-hot, label smoothing 0.1), the value target = the outcome share, and the score
  target = final margins.
- 5–10 epochs, AdamW lr 1e-3 with cosine decay, batch 1024.
- **Check:** the move-prediction top-1 accuracy should beat the heuristic prior. The value
  calibration curve, P(top score) against how often that actually happened, should be
  close to the diagonal.

### Phase 2: self-play reinforcement learning (AlphaZero loop)
```
repeat:
  1. self-play  N games with current best net (MCTS, 200–800 sims/move, root noise)
  2. store      (encoding, visit distribution, mover seat) for each ply
  3. at game end, attach outcome share and score margins (rotated to each mover)
  4. train      K steps on the replay buffer (last ~500k–1M positions)
  5. evaluate   candidate vs best in an arena (§7); promote if better
```
- **Ply cap:** 400 plies. Capped games are scored by points (current behaviour).
- **Value target noise:** to reduce variance, mix the game outcome with the root MCTS
  value: `z = 0.75·outcome + 0.25·root_Q`. Start without this and add it if the value
  loss plateaus high.
- **Keep some human data:** mix 10–20% scraped positions into each batch for the first
  iterations to avoid drifting into odd self-play patterns.
- **Learning rate:** 1e-3 → 1e-4 over training. Gradient clip at 1.0.

### Data format
`.npz` shards of 50k samples: `features (uint16 indices, variable length → padded to
40)`, `globals float16[16]`, `policy (sparse: idx int16[], prob float16[])`,
`value float16[4]`, `score float16[4]`, plus `source` (human/selfplay) and `game_id`.

---

## 7. Evaluating the engine

- **Arena:** 4-seat games, with the candidate in each seat in rotation against 3
  opponents. Report the **average points share** (fair is 0.25), the **win rate** (top
  score), and the **average place** (1–4).
- **Opponents:** random mover → greedy capture bot → 1-ply points-maximizer → the
  previous best network → the ResNet baseline.
- **Promotion rule:** a candidate replaces the best if its average share is ≥ 0.27 over
  ≥ 200 games (≈ 2 standard errors above 0.25).
- **Rating:** fit a Plackett–Luce / TrueSkill model on arena results, giving a
  multiplayer Elo for tracking progress over time.
- **Against humans (later):** replay scraped positions and measure top-1 / top-3
  agreement with strong players' moves.

---

## 8. Performance engineering

Measured target: **≥ 2,000 evaluations/second** on CPU for blitz-strength search.

| Bottleneck | Plan |
|---|---|
| Python move generation (`legal_moves`) | Profile first. Then cache legal moves per node, and move to numpy bitboards or a small C/Rust extension if it's > 50% of the time. |
| State copying in `apply_move` | Copy-on-write boards (tuples or `bytes` of 64 squares) instead of lists of lists. |
| Network calls | Batch 16–64 leaves with virtual loss; `torch.inference_mode()`; `torch.set_num_threads` tuned per worker. |
| Encoder | EmbeddingBag on index lists. Later: incremental accumulator updates per move. |
| Self-play throughput | Multiprocessing workers plus one inference server process. |
| Deployment | TorchScript or ONNX export of the evaluator; optional int8 quantization (ClippedReLU makes this easier). |

---

## 9. Code layout and changes to the current `bot/`

Current state: a working end-to-end pipeline with a **ResNet** evaluator, single-threaded
MCTS and self-play.

| File | Change |
|---|---|
| `features.py` | **Add** `sparse_features(state) → (indices, globals)`. Keep `encode()` for the ResNet baseline. |
| `model.py` | **Add** `ChaturajiMLP` (encoding layer + FC1–FC3 + heads, §4). Keep `ChaturajiNet` (ResNet) as a baseline. Same `forward → (policy, value, score)` interface. |
| `mcts.py` | Add: pluggable priors (network or heuristic), FPU tweak, evaluation cache, batched leaves with virtual loss, paranoid/max^n backup option, time-based budget. |
| `selfplay.py` | Add the ply cap, temperature schedule and data-shard writer. |
| `supervised.py` | **New:** build the dataset from `scraper/data` games and run Phase 1 training. |
| `train.py` | Switch between model types (`--model mlp|resnet`), mix in human data, add an LR schedule. |
| `arena.py` | **New:** seat-rotating matches, baselines, promotion rule, rating fit. |
| `bots.py` | **New:** random, greedy-capture and 1-ply baseline players. |
| `gui/play.py` | **Done (M7):** play a timed game against the engine in the browser, with clocks, flag-fall, and the engine's live evaluation. Time management is in `bot/timing.py`; `MCTS.search(time_limit=...)` adds time-based search, early stop and tree reuse (`MCTS.advance`). |
| `tests/test_bot.py` | Add tests: MLP shapes, sparse features match the dense planes, rotation invariance, the search tests in §5.8. |

---

## 10. Milestones with acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M1 | MLP evaluator + sparse encoding | Unit tests pass. Its output matches the ResNet interface. ≥ 2k evals/s batched on CPU. |
| M2 | Baseline bots + arena | Greedy-capture bot beats random with share ≥ 0.40. The arena report runs. |
| M3 | Heuristic-prior MCTS | MCTS (400 sims, value from the untrained-but-neutral net + heuristic priors) beats greedy with share ≥ 0.30. |
| M4 | Supervised bootstrap | Top-1 human-move accuracy ≥ 35%. Value is calibrated. MCTS+network beats M3's search with share ≥ 0.30. |
| M5 | Self-play RL loop | Five consecutive promotions. Steady rating increase on the arena chart. |
| M6 | Batched / parallel search | ≥ 5× self-play throughput over M5 on the same machine. |
| M7 | Blitz-ready engine | Plays a full `1 | 5` game within time from `gui/play.py`. Beats every earlier version. |

---

## 11. Recommendations summary

1. **Use the encoding-layer + 3 FC MLP as the main evaluator**, with the sparse
   piece-square embedding as the encoding layer. It's fast enough for real search on a
   CPU, and the canonical rotation removes the need for the network to learn the four
   seat orientations separately.
2. **Give the shared trunk three heads:** value (4-way top-score probability), score
   margin (auxiliary) and policy (MCTS priors). If a value-only network is required, use
   the heuristic priors from §5.3.
3. **Use max^n PUCT MCTS with vector backups**, exact terminal scoring, FPU and tree
   reuse. Add batching with virtual loss before scaling up.
4. **Bootstrap from scraped human games before self-play.** It's the largest
   time-saver available.
5. **Keep the ResNet as a measured baseline.** Switch only if it wins in the arena at
   equal time per move, not equal simulations.
6. **Measure everything with the seat-rotating arena** (points share), not loss curves
   alone.

---

## 12. Risks and open questions

- **Repetition +2 rule** is based on one real game. If it's wrong, the engine may value
  forced repetitions incorrectly. Confirm it with more scraped repetition games.
- **Check-bonus semantics** (when exactly double/triple checks score) matched every
  scraped game so far, but edge cases with discovered checks remain unconfirmed.
- **MLP vs. spatial patterns:** a fully connected network may need more data to learn
  local tactics than a CNN. Mitigations: more data, the search itself, and the optional
  attack features in §3.2.
- **Value target choice:** top-score share is noisy with 4 players. The score head
  reduces the variance, and rank-based targets are an alternative to try.
- **Opponent modelling:** human free-for-all play includes "kingmaking" and ganging up on
  the leader, which max^n doesn't model. Paranoid search or a learned opponent model
  could help later.
- **Compute:** CPU-only training limits self-play volume, so the supervised bootstrap
  and batching are what make the plan feasible.

---

## 13. Hyperparameter reference

| Area | Parameter | Start value | Range to try |
|---|---|---|---|
| Encoder | embedding width | 512 | 256–1024 |
| FC layers | widths | 1024 / 512 / 256 | 512/256/128 … 2048/1024/512 |
| FC layers | dropout | 0.1 | 0–0.2 |
| Loss | λ_score, λ_policy | 0.5, 1.0 | 0.25–1.0 |
| Optimizer | AdamW lr, weight decay | 1e-3, 1e-4 | 3e-4–2e-3 |
| Training | batch size | 1024 | 256–4096 |
| MCTS | c_puct | 1.5 | 1.0–3.0 |
| MCTS | simulations (self-play) | 200 | 100–800 |
| MCTS | Dirichlet α, ε | 0.3, 0.25 | 0.15–0.5, 0.15–0.3 |
| MCTS | FPU reduction | 0.1 | 0–0.3 |
| MCTS | leaf batch size | 32 | 8–128 |
| Self-play | temperature plies | 30 | 15–60 |
| Self-play | ply cap | 400 | 300–600 |
| Replay | buffer size | 500k | 200k–2M |
| Arena | promotion share / games | 0.27 / 200 | 0.26–0.30 / 100–400 |
