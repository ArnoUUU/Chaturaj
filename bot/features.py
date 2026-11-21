"""Canonical (mover-relative, rotated) encoding of Chaturaji states for the network.

The board is rotated so the player to move always sits in Red's seat (bottom-left,
pawns moving up). Rotating k quarter turns maps square (r, c) -> (7 - c, r) k times,
which is exactly np.rot90(plane, k). Owners are relabelled relative to the mover:
0 = me, 1 = next, 2 = across, 3 = previous, 4 = dead (X).
"""

from __future__ import annotations

import numpy as np

from chaturaji.constants import BOARD_SIZE, DEAD, PIECE_TYPES, TURN_ORDER
from chaturaji.moves import Move, is_promotion_square, legal_moves
from chaturaji.state import ChaturajiState, board_key

N = BOARD_SIZE
NUM_SQUARES = N * N
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES

PIECE_PLANES = 5 * len(PIECE_TYPES)      # 25
ALIVE_PLANES = 4
SCORE_PLANES = 4
NUM_PLANES = PIECE_PLANES + ALIVE_PLANES + SCORE_PLANES + 2  # + repetition + progress

SCORE_SCALE = 30.0
TURN_SCALE = 400.0

_TYPE_IDX = {t: i for i, t in enumerate(PIECE_TYPES)}
_COLOR_IDX = {c: i for i, c in enumerate(TURN_ORDER)}


def rotate_square(r: int, c: int, k: int) -> tuple[int, int]:
    """Apply (r, c) -> (7 - c, r) k times (real -> canonical for mover index k)."""
    for _ in range(k % 4):
        r, c = N - 1 - c, r
    return r, c


def _build_action_maps() -> tuple[np.ndarray, np.ndarray]:
    sq_map = np.zeros((4, NUM_SQUARES), dtype=np.int64)
    for k in range(4):
        for r in range(N):
            for c in range(N):
                cr, cc = rotate_square(r, c, k)
                sq_map[k, r * N + c] = cr * N + cc
    real_to_canon = np.zeros((4, NUM_ACTIONS), dtype=np.int64)
    for k in range(4):
        s = sq_map[k]
        real_to_canon[k] = (s[:, None] * NUM_SQUARES + s[None, :]).reshape(-1)
    canon_to_real = np.zeros_like(real_to_canon)
    for k in range(4):
        canon_to_real[k, real_to_canon[k]] = np.arange(NUM_ACTIONS)
    return real_to_canon, canon_to_real


REAL_TO_CANON, CANON_TO_REAL = _build_action_maps()


def mover_index(state: ChaturajiState) -> int:
    return _COLOR_IDX[state.current_player()]


def relative_owner(owner: str, k: int) -> int:
    if owner == DEAD:
        return 4
    return (_COLOR_IDX[owner] - k) % 4


def encode(state: ChaturajiState) -> np.ndarray:
    """(NUM_PLANES, 8, 8) float32 planes from the mover's point of view."""
    k = mover_index(state)
    x = np.zeros((NUM_PLANES, N, N), dtype=np.float32)
    for r, row in enumerate(state.board):
        for c, p in enumerate(row):
            if p is not None:
                cr, cc = rotate_square(r, c, k)
                x[relative_owner(p[0], k) * 5 + _TYPE_IDX[p[1]], cr, cc] = 1.0
    for i in range(4):
        color = TURN_ORDER[(k + i) % 4]
        if color not in state.eliminated:
            x[PIECE_PLANES + i] = 1.0
        x[PIECE_PLANES + ALIVE_PLANES + i] = state.scores[color] / SCORE_SCALE
    x[NUM_PLANES - 2] = state.position_counts.get(board_key(state.board), 0) / 3.0
    x[NUM_PLANES - 1] = min(state.turn / TURN_SCALE, 1.0)
    return x


def real_action(move: Move) -> int:
    (sr, sc), (er, ec) = move.start, move.end
    return (sr * N + sc) * NUM_SQUARES + er * N + ec


def canonical_action(move: Move, k: int) -> int:
    return int(REAL_TO_CANON[k, real_action(move)])


def action_to_move(canon_action: int, k: int, state: ChaturajiState) -> Move:
    a = int(CANON_TO_REAL[k, canon_action])
    s, e = divmod(a, NUM_SQUARES)
    start, end = divmod(s, N), divmod(e, N)
    piece = state.board[start[0]][start[1]]
    promo = "R" if piece and piece[1] == "P" and is_promotion_square(piece[0], end) else None
    return Move(start, end, promo)


def legal_canonical(state: ChaturajiState) -> tuple[list[Move], np.ndarray]:
    """Legal moves and their canonical action indices (same order)."""
    k = mover_index(state)
    moves = legal_moves(state)
    idx = np.fromiter((canonical_action(m, k) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idx


def to_relative(values_abs: np.ndarray, k: int) -> np.ndarray:
    """Absolute seat vector (R, B, Y, G) -> relative (me, next, across, prev)."""
    return np.roll(values_abs, -k)


def to_absolute(values_rel: np.ndarray, k: int) -> np.ndarray:
    return np.roll(values_rel, k)


def outcome_vectors(state: ChaturajiState) -> tuple[np.ndarray, np.ndarray]:
    """Absolute (winner_share, score_margin) targets from a finished state."""
    scores = np.array([state.scores[c] for c in TURN_ORDER], dtype=np.float32)
    best = scores.max()
    winners = (scores == best).astype(np.float32)
    share = winners / winners.sum()
    margin = (scores - scores.mean()) / 20.0
    return share, margin
