"""Applying moves, scoring, elimination, turn order and game end."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    BOARD_SIZE,
    CHECK_BONUS,
    DEAD,
    DEAD_KING_POINTS,
    LAST_STANDING_POINTS_PER_DEAD_KING,
    PIECE_POINTS,
    REPETITION_POINTS,
    TURN_ORDER,
)
from .moves import RESIGN, Move, Resign, kings_attacked_by, legal_moves
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
    end_bonus: tuple[tuple[str, int], ...] = ()  # (color, points) awarded because the game ended


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
    if captured == DEAD + "K":  # king of a player who resigned / timed out
        points += DEAD_KING_POINTS
    elif captured is not None and captured[0] != DEAD:
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

    end = _award_end_bonus(new)
    points += dict(end).get(color, 0)
    info = MoveInfo(color, piece, captured, points, bonus, frozenset(checked_after), eliminated, end)
    return new, info


def _award_end_bonus(state: ChaturajiState) -> tuple[tuple[str, int], ...]:
    """Add end-of-game points if `state` just ended the game. Call once per transition
    (after a move or an elimination on a state where the game was still running)."""
    reason = game_over_reason(state)
    awards: dict[str, int] = {}
    if reason == "last_king":
        dead_kings = sum(row.count(DEAD + "K") for row in state.board)
        for color in state.active_players():
            awards[color] = LAST_STANDING_POINTS_PER_DEAD_KING * dead_kings
    elif reason == "repetition":
        for color in state.active_players():
            awards[color] = REPETITION_POINTS
    out = tuple((c, p) for c, p in awards.items() if p)
    for c, p in out:
        state.scores[c] += p
    return out


def eliminate(state: ChaturajiState, color: str) -> ChaturajiState:
    """Remove `color` without a king capture (resignation / timeout in real games).

    All of that color's pieces, king included, become dead (X). The dead king is still
    worth 3 to whoever captures it, and to the last player standing if it is never
    captured. If it was that color's turn, play passes to the next active player.
    Returns a new state.
    """
    if color in state.eliminated or is_game_over(state):
        return state.copy()
    new = state.copy()
    new.eliminated.add(color)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = new.board[r][c]
            if p is not None and p[0] == color:
                new.board[r][c] = DEAD + p[1]
    if new.current_player() == color:
        _advance_turn(new)
    _award_end_bonus(new)
    return new


def apply_action(state: ChaturajiState, action: Move | Resign) -> tuple[ChaturajiState, MoveInfo | None]:
    """Play a move, or resign for the player to move (`RESIGN`). Returns the new state and
    the move's info (None for a resignation)."""
    if action is RESIGN:
        return eliminate(state, state.current_player()), None
    return apply_move(state, action)


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
