# Chaturaji Board State — Plan & Implementation Notes

A Python game state for simulating Chaturaji (chess.com rules) on an 8x8 board, built so a
model can train on it. The plan below is implemented in `chaturaji/` and tested in `tests/`.

---

## 1. Rules

Source: https://www.chess.com/terms/chaturaji, plus rules confirmed by the user.

- **Board:** 8x8, standard chess notation (`a1` bottom-left, `h8` top-right).
- **Players:** 4. Turn order **Red → Blue → Yellow → Green**. Red goes first and sits on the
  bottom row.
- **Pieces per player:** King `K`, Bishop `B`, Knight `N`, Boat/Rook `R`, 4 Pawns `P`.
- **Movement:** K, B, N, R move exactly like in chess. Pawns move 1 square forward in their
  own direction and capture 1 square diagonally forward. No double step, no en passant, no
  castling.
- **Promotion:** a pawn reaching the 8th rank relative to its own side (the far edge in its
  direction) always becomes a Rook/Boat.
- **No check restriction:** kings may walk into attack, and are captured like any piece.
  There is always at least one legal move.
- **Capture points (active pieces only):** P +1, N +3, K +3, B +5, R +5.
- **Check bonuses:** a move that gives check and leaves the mover attacking 2 active enemy
  kings at once scores +1. Three kings at once scores +5. Holding an existing double check
  while making a quiet move elsewhere does not score again.
- **Elimination:** when a king is captured, that player can no longer move and their turns
  are skipped. Their remaining pieces become **colorless dead pieces** (`X` + type): they
  stay on the board, block, can be captured by anyone, and are worth **0** points.
- **Game end:**
  - **Last king standing:** only one king remains.
  - **Threefold repetition:** the same board (piece placement only) occurs 3 times.
- **Winner:** the highest score, even if that player was eliminated. Ties are possible.

### Starting position
Red's placement is confirmed. The other corners are assumed and live in one table
(`STARTING_LAYOUT` in `constants.py`):

```
      a    b    c    d    e    f    g    h
8    BR   BP   .    .    YK   YB   YN   YR
7    BN   BP   .    .    YP   YP   YP   YP
6    BB   BP   .    .    .    .    .    .
5    BK   BP   .    .    .    .    .    .
4    .    .    .    .    .    .    GP   GK
3    .    .    .    .    .    .    GP   GB
2    RP   RP   RP   RP   .    .    GP   GN
1    RR   RN   RB   RK   .    .    GP   GR
```
Pawn directions: Red up, Blue right, Yellow down, Green left.

---

## 2. Representation

- **Piece:** a 2-char string, owner + type. Owners are `R B Y G`, plus `X` for dead pieces.
  Example: `"RK"`, `"BB"`, `"XN"`. An empty square is `None`.
- **`ChaturajiState`** (`state.py`):
  - `board`: 8x8 `list[list[str | None]]`. `board[0]` is rank 8 and `board[7]` is rank 1.
  - `turn`: an integer that goes up by 1 per move and also counts skipped turns.
    `current_player() = TURN_ORDER[turn % 4]`.
  - `scores`, `eliminated`.
  - `position_counts`: board-only key → number of times seen, for threefold repetition.
- **`Move(start, end, promotion)`**, with `.uci` (`"d2d3"`) and `Move.from_uci`.

---

## 3. Modules

```
chaturaji/
  constants.py  rules data: turn order, points, check bonus, pawn directions, start layout
  state.py      ChaturajiState, square <-> index helpers, board_key, to_dict/from_dict
  moves.py      Move, legal_moves, is_attacked_by, kings_attacked_by
  game.py       apply_move -> (new_state, MoveInfo), game_over_reason, winners
  notation.py   chess-style SAN per move + 4-column move table
  record.py     Game: state + move history, save/load JSON, replay all states
  viewer.py     minimal terminal viewer for saved games
  encoding.py   numpy tensors for training: state planes, action index, legal mask
  simulate.py   random self-play, writes game records
tests/test_chaturaji.py
```

### Training use
- Use `ChaturajiState` + `apply_move` directly. It's pure (returns a new state) and keeps no
  history overhead. `MoveInfo.points` can be used as a per-move reward.
- `encoding.encode_state(state)` returns a `(30, 8, 8)` float32 array: 25 one-hot piece
  planes (5 owners × 5 types), 4 planes for the player to move, and 1 repetition-count plane.
- Actions: `start * 64 + end` (4096 total). Promotion is forced, so no extra bits are needed.
  `legal_action_mask(state)`, `move_to_action`, `action_to_move`.

### Notation (`notation.py`)
- Like chess SAN: `Nc3`, `Bxe5`, pawn moves by destination (`d3`), and promotion `=R`.
- Pawn captures prefix the origin file for Red/Yellow (`cxd3`) and the origin rank for
  Blue/Green (`6xc7`), since those pawns move along ranks.
- Disambiguation: file, then rank, then full square.
- `+` means the mover now attacks an active king. `#` means the move captured a king.
- The move table has 4 columns (Red, Blue, Yellow, Green), with `--` for skipped turns:

```
     Red       Blue      Yellow    Green
  1. c3        Nc8       e6        Kg5
  ...
  7. axb3#     --        g6+       Kxe6
```

### Viewing past games
```
python -m chaturaji.simulate --games 5 --seed 0 --out games
python -m chaturaji.viewer games/game_0.json
```
Viewer commands: Enter/`n` next, `p` previous, a number jumps to that ply, `s`/`e` start/end,
`t` move table, `q` quit. `Game.load(path).states()` gives every position, so a graphical
UI can be built on top of it later.

---

## 4. Open questions
1. Exact corners and piece order for Blue, Yellow and Green (one table to change).
2. Whether chess.com awards check bonuses exactly as implemented: on a checking move, by the
   number of kings attacked after it.
