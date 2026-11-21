# chaturaji — rules engine

The board state and rules for Chaturaji as played on chess.com. It is pure Python with no
dependencies. The full rules are in [../PLAN.md](../PLAN.md).

## Rules in brief
- 8x8 board, standard notation (`a1` bottom-left). Red starts on the bottom row.
- Turn order: Red → Blue → Yellow → Green.
- Pieces are 2-character codes: owner `R B Y G` (or `X` for dead) + type `K B N R P`,
  e.g. `"RK"`, `"XN"`. Empty squares are `None`.
- There is no check restriction: kings can walk into attack and be captured.
- Pawns move one square forward in their own direction and promote to a rook on the far edge.
- Capture points: P 1, N 3, K 3, B 5, R 5. Capturing a dead piece scores 0.
- Check bonuses: +1 for checking 2 kings at once, +5 for 3.
- When a king is captured (or its player resigns or times out), that player is out and
  their pieces become dead `X` pieces.
- A resigned or timed-out player's dead king is still worth 3 when captured. The last
  player standing gets +3 for each dead king left on the board.
- The game ends on last king standing or threefold repetition (board only). Repetition
  gives each remaining player +2. The highest score wins.
- These scoring rules reproduce the final scores of 7 real chess.com games exactly
  (`tests/test_pgn4.py::test_real_chesscom_game_scores_match` replays one of them).

## Try it
From the repository root:

```powershell
python -m chaturaji.simulate --games 3 --seed 0 --out games
```

```powershell
python -m chaturaji.viewer games/game_0.json
```

The viewer prints the 4-column move table first, then steps through the game:
Enter/`n` next, `p` previous, a number jumps to that ply, `s`/`e` start/end, `t` table,
`q` quit.

## Use it from Python
```python
from chaturaji import ChaturajiState, Game, legal_moves, apply_move, game_over_reason

s = ChaturajiState.initial()
print(s)                       # board, turn, scores
moves = legal_moves(s)         # list[Move]
s2, info = apply_move(s, moves[0])   # pure: returns a new state + MoveInfo(points, checks, ...)

g = Game()                     # keeps history for saving/viewing
g.play_uci("d2d3")
g.play_uci("a7c8")
print(g.move_table())
g.save("games/mine.json")      # opens in the viewer
```

## Files
| File | Contents |
|---|---|
| `constants.py` | turn order, points, check bonus, pawn directions, `STARTING_LAYOUT` |
| `state.py` | `ChaturajiState` (8x8 board, integer `turn`, scores, eliminated, repetition counts) |
| `moves.py` | `Move`, `legal_moves`, attack detection |
| `game.py` | `apply_move`, `eliminate` (resign/timeout), `game_over_reason`, `winners` |
| `notation.py` | chess-style notation and the 4-column move table |
| `record.py` | `Game`: history, save/load JSON, replay every position |
| `viewer.py` | terminal viewer for saved games |
| `simulate.py` | random games (sample data for the viewer) |
| `encoding.py` | simple numpy encoding (the bot uses `bot/features.py` instead) |

Blue, Yellow and Green's starting corners are assumed. To change them, edit
`STARTING_LAYOUT` in `constants.py`.

## Tests
```powershell
python -m pytest -q tests/test_chaturaji.py
```
