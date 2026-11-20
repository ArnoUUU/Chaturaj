"""Tensor encodings for model training (requires numpy).

State planes, shape (PLANES, 8, 8), float32:
  0..24  one-hot pieces: owner (R, B, Y, G, X=dead) x type (K, B, N, R, P),
         plane index = owner_index * 5 + type_index
  25..28 one-hot player to move (constant planes)
  29     repetition count of the current board / 3 (constant plane)

Actions: start_square * 64 + end_square (4096 total), squares indexed row * 8 + col.
Promotion is forced, so it needs no extra action bits.
"""

from __future__ import annotations

import numpy as np

from .constants import BOARD_SIZE, OWNERS, PIECE_TYPES, TURN_ORDER
from .moves import Move, is_promotion_square, legal_moves
from .state import ChaturajiState, board_key

N = BOARD_SIZE
NUM_SQUARES = N * N
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES
PIECE_PLANES = len(OWNERS) * len(PIECE_TYPES)
PLANES = PIECE_PLANES + len(TURN_ORDER) + 1

_OWNER_IDX = {o: i for i, o in enumerate(OWNERS)}
_TYPE_IDX = {t: i for i, t in enumerate(PIECE_TYPES)}


def encode_state(state: ChaturajiState) -> np.ndarray:
    x = np.zeros((PLANES, N, N), dtype=np.float32)
    for r, row in enumerate(state.board):
        for c, p in enumerate(row):
            if p is not None:
                x[_OWNER_IDX[p[0]] * 5 + _TYPE_IDX[p[1]], r, c] = 1.0
    x[PIECE_PLANES + TURN_ORDER.index(state.current_player())] = 1.0
    x[PLANES - 1] = state.position_counts.get(board_key(state.board), 0) / 3.0
    return x


def move_to_action(move: Move) -> int:
    (sr, sc), (er, ec) = move.start, move.end
    return (sr * N + sc) * NUM_SQUARES + er * N + ec


def action_to_move(action: int, state: ChaturajiState) -> Move:
    start, end = divmod(action, NUM_SQUARES)
    s, e = divmod(start, N), divmod(end, N)
    piece = state.board[s[0]][s[1]]
    promo = "R" if piece and piece[1] == "P" and is_promotion_square(piece[0], e) else None
    return Move(s, e, promo)


def legal_action_mask(state: ChaturajiState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in legal_moves(state):
        mask[move_to_action(m)] = True
    return mask
