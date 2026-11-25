# claude-playground: search experiments

Experiments on the bot's search, kept apart from the rest of the repo. Nothing outside
`claude-playground/` is changed by this work. Everything runs locally on the laptop (no
cluster, no paid compute), and nothing here plays on chess.com.

How the work is split: Opus picks the ideas, designs the experiments and reviews the code;
Haiku subagents write the code. Every change and every measured result is recorded below.

Status: two rounds, 9 experiments, 432 games, done on 2026-11-24. **Result: paranoid
alpha-beta beats the current max^n search, at equal nodes (0.68), equal time (0.73) and a
doubled budget (0.76), where 0.5 is even.** Width 4 and quiescence 2 are confirmed. See
section 4.

## 1. Setup

### The arena

`engine_pg/` is a copy of the repo's Rust core (`engine/src/nnue.rs`, `rules.rs`, `search.rs`
as of commit d2d12ab, which includes selective depth and the search preferences) without the
Python bindings, and `engine_pg/src/bin/arena.rs` is a match runner. Details, build and test
commands, and every option are in `engine_pg/README.md`.

A match is one **test** player against three copies of a **base** player, all on the same
network (nnue-v1, the 2026-11-23 a256-h32x32 net, exported with `export_weights.py` to
`weights/nnue-v1/`).

- 48 games per experiment: 12 openings (4 random plies from a fixed seed, seed 7 in every
  experiment so all experiments use the same openings), each played with the test player in
  all four seats.
- Score: the test player's **pairwise** result against each of the three others on final
  points (win 1, tie 0.5, loss 0), averaged. 0.5 = even with the base player. ±2 SE is over
  the 48 games.
- "1st place" = the share of games in which the test player beat all three. 0.25 = even.

### Why a node budget instead of time

The laptop is shared with an 8-hour self-play training run (12 workers). Times per move
change with that load, so the main experiments give each side the same **number of nodes**
per move (10,000), with iterative deepening keeping the last completed depth. A node is one
network evaluation or one interior node; both searches count them the same way. Round 2
checks that the round-1 winner also wins at equal **time**.

Every player uses the self-play settings unless the experiment changes one: width 4 after
the first ply, quiescence 2, check extensions 3, discount 0.9985, elimination bonus 0.008 to
turn 200.

Results are in `results/<experiment>.jsonl` (one line per game) and `.log` (progress); the
table is made by `python claude-playground/summarize.py`.

## 2. Round 1

### Ideas

1. **Paranoid alpha-beta.** max^n (the current search) backs up all four players' values
   and cannot prune, apart from the immediate-win cut. So at a fixed budget it searches
   shallow: about 3 plies at 10,000 nodes. Paranoid search assumes the three opponents play
   together against the searching player. That makes the game two-sided (my value against
   theirs), so alpha-beta pruning works and the same budget reaches deeper. The risk is
   that it is pessimistic and plays too defensively. In Chaturaji the assumption is partly
   realistic: in self-play, 75 to 100% of games include a gang-up (three players capturing
   from one).
2. **Width 3 and 6** (against the current 4). Width 4 was picked to make depth 5 affordable
   in self-play. At a fixed budget, width trades breadth against depth.
3. **Quiescence depth 4** (against 2). Longer capture sequences settle more tactics, at a
   node cost.

### Code (Haiku)

- `engine_pg/src/search.rs`:
  - a node limit (`node_limit`) that stops the search like a timeout;
  - `paranoid`: paranoid alpha-beta (`search_paranoid`, `quiesce_paranoid`) with its own
    transposition table storing bounds. The preferences (discount, elimination bonus) are
    applied with the same arithmetic as in max^n, and the alpha-beta window is passed down
    through its inverse.
  - max^n and paranoid share the move selection (`select`) and the quiescence move list.
- `engine_pg/src/bin/arena.rs`: the match runner.
- `export_weights.py`: `.npz` to the arena's `net.txt` + `net.bin`.
- Tests: paranoid returns a legal move and restores the position; its root value is a
  utility; the node limit is respected (for both searches).

Opus review of the paranoid code:
- The preference shaping is monotone, so passing the window through its inverse is exact.
- Fail-soft bounds are stored with the right flags.
- Move ordering and the width cut are shared with max^n.
- The width cut at opponent nodes keeps the opponent's own best moves rather than the
  moves worst for the searching player. That is a choice, tested in round 2 (`pmin`).

### Results

Base player in every row: max^n, 10,000 nodes a move, self-play settings. Run 2026-11-24,
15:04 to 16:12, 4 game threads next to the self-play run.

| Experiment | Test player | Score ±2 SE | 1st place | Depth test / base | Turns |
|---|---|---:|---:|---:|---:|
| e1 | paranoid alpha-beta | **0.684 ± 0.095** | 0.38 | 4.23 / 2.92 | 165 |
| e2 | width 6 | 0.372 ± 0.106 | 0.12 | 2.09 / 2.85 | 151 |
| e3 | width 3 | 0.503 ± 0.107 | 0.21 | 3.09 / 2.87 | 156 |
| e4 | quiescence depth 4 | 0.413 ± 0.104 | 0.12 | 2.32 / 2.89 | 163 |

Depth is the mean completed iteration per move.

- **Paranoid alpha-beta is clearly better** (the ±2 SE range is well above 0.5). With the
  same nodes it searches 1.3 plies deeper (4.2 against 2.9). It beats each max^n opponent
  in 68% of pairings, and it wins outright in 38% of games, against 25% for an even player.
  It scored above 0.5 from every seat (0.76, 0.64, 0.76, 0.57).
- **Width 6 is clearly worse**: the extra breadth costs almost a ply (2.1 against 2.9).
- **Width 3 is even** with width 4: a little deeper (3.1), but it misses moves. Width 4 stays.
- **Quiescence 4 is probably worse** (0.41, not significant): it costs half a ply.
- The pattern: at this budget, **depth is worth more than breadth or longer capture
  sequences**. The one change that gains is the one that buys depth without cutting moves.

Caveat: paranoid gets its depth from pruning, but a node of it may cost more time than a
node of max^n (alpha-beta bookkeeping, the second table). Round 2 tests it at equal time.

## 3. Round 2

### Ideas

Round 1 makes paranoid search the candidate. Round 2 first checks that the result is real,
then tries two cheap refinements and one simplification.

1. **Equal time** (f1). Nodes are a fair budget only if a paranoid node and a max^n node take
   the same time. A 4-game smoke test (under load) gave 84 µs a node for paranoid against
   67 µs for max^n, so this needs checking: both sides get 500 ms a move (about 10,000 nodes
   for max^n).
2. **Sorted children** (f2, `psort`). Alpha-beta cuts more when the best move is searched
   first. The width cut already evaluates every child, so ordering the kept children by that
   evaluation (best for the side to move first, the table's move still first) costs nothing.
   It should give more depth for the same nodes. It never changes the value of a fixed-depth
   search, only how fast it is found (tested).
3. **Min-select at opponent nodes** (f3, `pmin`). The width cut keeps each opponent's own 4
   best moves. A fully paranoid search would keep the 4 moves worst for the searching player
   instead. That is more consistent with the paranoid assumption but may make the player
   more fearful.
4. **No width cut** (f4, width 0). Alpha-beta prunes on its own, so the width cut may cost
   more than it saves.
5. **Twice the budget** (f5, 20,000 nodes each). Does the gain hold as both sides search
   deeper?

### Code (Haiku)

- `arena.rs`:
  - `ms=` time budget per move;
  - time per move for each side in the JSON and the summary, with µs per node.
- `search.rs`:
  - `paranoid_min_select` (`pmin`) and `paranoid_sort` (`psort`), paranoid only and off by
    default;
  - `select` takes the sort key (which player's utility, which direction) and whether to
    reorder the kept children.
  - max^n calls it exactly as before.
- New tests:
  - each option returns a legal move and restores the position;
  - `psort` leaves the fixed-depth paranoid value unchanged (within 1e-12);
  - max^n's move, value and node count don't change with the options set.
  - 16 tests pass with no warnings.
- Built into `target/b2/` so the running round-1 binary was not overwritten.

Opus review:
- With the options off, the kept set and the order are the same as before.
- `psort` puts the table's move first, then sorts by the same key as the selection.
- In the depth-1 branch, extended (check) children have no static value, so they are
  searched first.

### Results

Base player: max^n with the same budget as the test player. Run 2026-11-24, 16:12 to 17:59,
same openings as round 1. e1 from round 1 is repeated for comparison. Time is the mean per
move; µs per node is under load from the self-play run, so it is a rough guide.

| Experiment | Test player (paranoid +) | Budget | Score ±2 SE | 1st place | Depth test / base | Nodes test / base | ms a move test / base |
|---|---|---|---:|---:|---:|---:|---:|
| e1 | – | 10,000 nodes | 0.684 ± 0.095 | 0.38 | 4.23 / 2.92 | 9,595 / 9,812 | – |
| f1 | – | **500 ms** | **0.729 ± 0.093** | 0.46 | 3.68 / 2.76 | 5,862 / 7,407 | 508 / 505 |
| f2 | sorted children (`psort`) | 10,000 nodes | 0.677 ± 0.085 | 0.29 | 4.37 / 2.88 | 9,644 / 9,781 | 879 / 700 |
| f3 | min-select (`pmin`) | 10,000 nodes | 0.733 ± 0.090 | 0.44 | 4.36 / 2.90 | 9,562 / 9,901 | 955 / 776 |
| f4 | no width cut (width 0) | 10,000 nodes | 0.674 ± 0.094 | 0.35 | 2.75 / 2.98 | 9,700 / 9,823 | 691 / 653 |
| f5 | – | **20,000 nodes** | **0.760 ± 0.090** | 0.52 | 4.98 / 3.45 | 18,733 / 19,618 | 1,682 / 1,401 |

- **The paranoid result is real.**
  - **Equal time (f1):** it scored 0.73. A paranoid node costs about 25% more time
    (87 against 68 µs), so in 500 ms it searches fewer nodes (5,900 against 7,400) but
    still goes 0.9 plies deeper.
  - **Twice the budget (f5):** the gain grows, to 0.76 and first place in half the games.
    Both sides go deeper, and paranoid stays 1.5 plies ahead.
- **Depth is not the whole story (f4).** With no width cut, paranoid searches a little
  *less* deep than max^n (2.75 against 2.98) and still scores 0.67. At about the same depth,
  it searches every move where max^n keeps 4. So the gain there comes from the extra breadth
  (which max^n can't afford: width 6 cost it a ply), from the paranoid model itself, or
  both. The model is plausible here: assuming the other three play against you fits a game
  where 75 to 100% of self-play games include a gang-up. Telling the two apart would need
  paranoid against max^n at the same fixed depth.
- **Neither refinement helps.**
  - `psort` gains a little depth (4.37 against 4.23) but nothing in score. Sorting costs
    time, and the order from `ordered` (table move, captures, checks) is already good.
  - `pmin` scores 0.73 against 0.68, but the difference is well inside the noise (±0.13 for
    a difference of two experiments). Not worth a change on this evidence.
- The width cut neither helps nor hurts paranoid at 10,000 nodes (f4 against e1). Width 4
  is kept, for the same settings as self-play.

## 4. Conclusions

1. **Paranoid alpha-beta beats the current max^n search** at equal nodes (0.68), equal time
   (0.73) and a doubled budget (0.76). It is the one change from these experiments worth
   taking into the engine.
2. **Keep width 4 and quiescence 2.** Width 6 is clearly worse (0.37), width 3 is even
   (0.50), and quiescence 4 is probably worse (0.41).
3. `psort` and `pmin` are not worth the change.

Limits of these tests:
- Every match is one paranoid player against three max^n players, all on the same network
  (nnue-v1). How paranoid players do against each other, against MCTS, or against people
  is not tested.
- A paranoid player may be too cautious when the others are not ganging up.
- 48 games give ±0.09 to ±0.11 (2 SE). That is enough for the large effects here, not for
  small ones.

## 5. Possible next steps

These go beyond the playground, so they are for the user to decide.

- **Bring paranoid search into the engine** (`engine/src/search.rs`, the Python bindings,
  `bot/maxn.py` as reference): an option on `MaxN`, off by default, with the parity tests.
- **Use it in self-play** (depth 5 with paranoid is cheaper than max^n, or deeper in the same
  time) and as the playing engine. Self-play with all four players paranoid is a different
  game from these 1-against-3 tests, so check the result against MCTS + MLP.
- **Other searches from the multi-player literature:** best-reply search (only the
  strongest opponent reply at each level) is reported to beat both paranoid and max^n in
  some games. It needs a "pass" in the rules for the opponents who don't move.

## Files

| Path | What it is |
|---|---|
| `engine_pg/` | the Rust copy of the engine with node and time limits, paranoid search, the arena (`engine_pg/README.md`) |
| `export_weights.py` | `.npz` network to the arena's format |
| `run_experiments.sh`, `run_batch2.sh` | the two rounds, exactly as run |
| `summarize.py` | the results table from `results/*.jsonl` |
| `results/` | one `.jsonl` (per game) and `.log` (progress) per experiment; `batch1.txt`, `batch2.txt` the summaries |
| `weights/` | the exported network (not committed: `python claude-playground/export_weights.py models/2026-11-23_1607_nnue-v1_nnue-a256-h32x32_ep8_cpu/nnue.npz claude-playground/weights/nnue-v1`) |
