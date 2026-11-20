"""Applying moves, scoring, elimination, turn order and game end."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import BOARD_SIZE, CHECK_BONUS, DEAD, PIECE_POINTS, TURN_ORDER
from .moves import Move, kings_attacked_by, legal_moves
from .state import ChaturajiState, board_key


@dataclass(frozen=True)
class MoveInfo:
    """What a move did; used for notation, logging and reward shaping."""
    color: str
    piece: str                 # moving piece code before the move, e.g. "RP"
    captured: str | None       # captured piece code (may be dead, "X..")
    points: int                # total points gained by the mover this move
    check_bonus: int
    checked: frozenset[str]    # active enemy kings the mover attacks after the move
    eliminated: str | None     # color eliminated by this move (king captured)


def apply_move(state: ChaturajiState, move: Move) -> tuple[ChaturajiState, MoveInfo]:
    """Return the new state (input is not modified) and info about the move."""
    color = state.current_player()
    new = state.copy()
    board = new.board
    (sr, sc), (er, ec) = move.start, move.end
    piece = board[sr][sc]
    if piece is None or piece[0] != color:
        raise ValueError(f"{move} does not move a {color} piece")

    checked_before = kings_attacked_by(state.board, color, state.active_players())

    captured = board[er][ec]
    board[er][ec] = color + move.promotion if move.promotion else piece
    board[sr][sc] = None

    points = 0
    eliminated = None
    if captured is not None and captured[0] != DEAD:
        points += PIECE_POINTS[captured[1]]
        if captured[1] == "K":
            eliminated = captured[0]
            new.eliminated.add(eliminated)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    p = board[r][c]
                    if p is not None and p[0] == eliminated:
                        board[r][c] = DEAD + p[1]

    checked_after = kings_attacked_by(board, color, new.active_players())
    bonus = 0
    if checked_after - checked_before:  # the move gave (at least one new) check
        bonus = CHECK_BONUS.get(len(checked_after), 0)
    points += bonus
    new.scores[color] += points

    _advance_turn(new)
    key = board_key(board)
    new.position_counts[key] = new.position_counts.get(key, 0) + 1

    info = MoveInfo(color, piece, captured, points, bonus, frozenset(checked_after), eliminated)
    return new, info


def _advance_turn(state: ChaturajiState) -> None:
    state.turn += 1
    if len(state.active_players()) <= 1:
        return
    while state.current_player() in state.eliminated:
        state.turn += 1


def game_over_reason(state: ChaturajiState) -> str | None:
    """'last_king', 'repetition', or None if the game continues."""
    if len(state.active_players()) <= 1:
        return "last_king"
    if any(n >= 3 for n in state.position_counts.values()):
        return "repetition"
    return None


def is_game_over(state: ChaturajiState) -> bool:
    return game_over_reason(state) is not None


def winners(state: ChaturajiState) -> list[str]:
    """Highest scorer(s). Points decide the winner, not survival."""
    best = max(state.scores.values())
    return [c for c in TURN_ORDER if state.scores[c] == best]


def require_moves(state: ChaturajiState) -> list[Move]:
    moves = legal_moves(state)
    if not moves:
        raise RuntimeError(f"No legal moves for {state.current_player()} at turn {state.turn}")
    return moves
