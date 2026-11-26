# Results: small NNUE + max^n vs greedy (2026-11-26)

Score = mean pairwise score of 3 against 3 greedy bots on final points (1.5 = even), ± 2 SE.
Model: 50 ms per move (4 threads, one game per thread). Greedy: preset `greedy`, depth 4
(~7 ms per move in Rust) unless noted.

## Controls (160 games)
| Test player | Score |
|---|---|
| greedy (itself) | 1.50 (by symmetry) |
| naive (depth-2 greedy) | 0.29 |
| untrained net (all leaves tie) | 0.50 |

## Learning rounds (160 new games each, fresh openings each round)
Each round trains a fresh net on every position so far, then plays.

| Round | Games trained on | Score |
|---|---|---|
| 0 (untrained) | 0 | 0.50 |
| 1 | 160 | 1.14 |
| 2 | 320 | 1.52 |
| 3 | 480 | 1.24 |
| 4 | 640 | 1.43 |
| 5 | 800 | 1.46 |
| 6 | 960 | 1.76 |
| 7 | 1,120 | 1.84 |
| 8 | 1,280 | 1.54 |

Per-round scores swing by about ±0.25 because each round's net is a single training run.

## Selection (320 games, openings seed 1000)
| Net | Trained on | Score |
|---|---|---|
| r7 | 1,120 games | 1.78 ± 0.11 |
| r8 | 1,280 games | 1.51 ± 0.11 |
| r9 | 1,440 games | **1.86 ± 0.11** |

## Final (r9, 320 games, openings seed 2000, not used for selection)
| Opponents | Score | Finished strictly first |
|---|---|---|
| greedy, depth 4 | **1.91 ± 0.12** | 35% (25% = even) |
| greedy, 50 ms per move (same clock, iterative deepening) | **1.68 ± 0.11** | 21% |

By seat vs depth 4 (R B Y G): 2.02, 1.94, 1.89, 1.77. The model reaches depth 2.2 on average
in 50 ms (max^n, full width, quiescence 2).

## What mattered
- The network inputs. With piece-square features and the 16 plain globals only, nets trained
  on these games scored 0.2 to 0.4. Adding 20 threat/material inputs per position
  (`engine/src/features.rs`) let the net learn what greedy's hand eval encodes.
  A hand-set version of the same net scored 1.95 (80 games), which is what showed the search
  was not the bottleneck.
- Training: clipped-ReLU units all died in the first epoch until the hidden biases started at
  0.5, the output bias at the target mean, with one epoch of warm-up and batch 128.
- The points head: logit(i above j) = beta * ((s_i + F_i) - (s_j + F_j)) with F learned as
  discounted future points (0.97 per ply). Pure pairwise logits from 160 games overfit at once.
- No resigning with 3+ players: a score-aware eval otherwise resigns while ahead.
- Greedy port note: `claude-playground/greedy/match.py` and the GUI share one `Bot` (one
  transposition table) across all greedy seats, and the table key does not include the
  searching seat, so values from one seat's view leak into another's search. Here every seat
  has its own bot.
