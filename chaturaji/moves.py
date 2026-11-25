"""Move generation. There is no check restriction: every pseudo-legal move is legal."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    BISHOP_DIRS,
    BOARD_SIZE,
    DEAD,
    KING_OFFSETS,
    KNIGHT_OFFSETS,
    PAWN_DIRECTION,
    ROOK_DIRS,
)
from .state import Board, ChaturajiState, index_to_square, square_to_index

N = BOARD_SIZE


@dataclass(frozen=True)
class Move:
    start: tuple[int, int]
    end: tuple[int, int]
    promotion: str | None = None  # always "R" when present (forced)

    @property
    def uci(self) -> str:
        return index_to_square(*self.start) + index_to_square(*self.end)

    @classmethod
    def from_uci(cls, text: str, board: Board) -> "Move | Resign":
        if text == Resign.uci:
            return RESIGN       # "resign" in game records
        start = square_to_index(text[:2])
        end = square_to_index(text[2:4])
        piece = board[start[0]][start[1]]
        promo = "R" if piece and piece[1] == "P" and is_promotion_square(piece[0], end) else None
        return cls(start, end, promo)

    def __str__(self) -> str:
        return self.uci


class Resign:
    """The other action a player has besides a move: resigning (chess.com's "R").

    The player is out, their pieces (king included) become dead, and their score is frozen.
    With two players left it ends the game ("claimed the win" on chess.com): the other
    player is the last king standing and gets the dead-king bonus, then the higher score
    wins. Use the single instance `RESIGN`.
    """

    __slots__ = ()
    uci = "resign"

    def __repr__(self) -> str:
        return "RESIGN"

    __str__ = __repr__

    def __reduce__(self):
        return "RESIGN"         # pickles (worker processes) back to the same instance


RESIGN = Resign()


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < N and 0 <= c < N


def is_promotion_square(color: str, square: tuple[int, int]) -> bool:
    """True if `square` is the far edge (8th rank relative to `color`)."""
    dr, dc = PAWN_DIRECTION[color]
    r, c = square
    if dr:
        return r == (0 if dr < 0 else N - 1)
    return c == (0 if dc < 0 else N - 1)


def _pawn_capture_offsets(color: str) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = PAWN_DIRECTION[color]
    if dr:
        return (dr, -1), (dr, 1)
    return (-1, dc), (1, dc)


def piece_moves(board: Board, r: int, c: int) -> list[Move]:
    piece = board[r][c]
    if piece is None or piece[0] == DEAD:
        return []
    color, ptype = piece[0], piece[1]
    out: list[Move] = []

    def target_ok(tr: int, tc: int) -> bool:
        t = board[tr][tc]
        return t is None or t[0] != color

    if ptype in ("R", "B"):
        dirs = ROOK_DIRS if ptype == "R" else BISHOP_DIRS
        for dr, dc in dirs:
            tr, tc = r + dr, c + dc
            while _on_board(tr, tc):
                t = board[tr][tc]
                if t is None:
                    out.append(Move((r, c), (tr, tc)))
                else:
                    if t[0] != color:
                        out.append(Move((r, c), (tr, tc)))
                    break
                tr += dr
                tc += dc
    elif ptype in ("N", "K"):
        for dr, dc in (KNIGHT_OFFSETS if ptype == "N" else KING_OFFSETS):
            tr, tc = r + dr, c + dc
            if _on_board(tr, tc) and target_ok(tr, tc):
                out.append(Move((r, c), (tr, tc)))
    else:  # pawn
        dr, dc = PAWN_DIRECTION[color]
        tr, tc = r + dr, c + dc
        if _on_board(tr, tc) and board[tr][tc] is None:
            promo = "R" if is_promotion_square(color, (tr, tc)) else None
            out.append(Move((r, c), (tr, tc), promo))
        for odr, odc in _pawn_capture_offsets(color):
            tr, tc = r + odr, c + odc
            if _on_board(tr, tc):
                t = board[tr][tc]
                if t is not None and t[0] != color:
                    promo = "R" if is_promotion_square(color, (tr, tc)) else None
                    out.append(Move((r, c), (tr, tc), promo))
    return out


def legal_moves(state: ChaturajiState, color: str | None = None) -> list[Move]:
    color = color or state.current_player()
    board = state.board
    moves: list[Move] = []
    for r in range(N):
        row = board[r]
        for c in range(N):
            p = row[c]
            if p is not None and p[0] == color:
                moves.extend(piece_moves(board, r, c))
    return moves


def is_attacked_by(board: Board, square: tuple[int, int], color: str) -> bool:
    """True if any piece of `color` attacks `square`."""
    r, c = square
    for dr, dc in KNIGHT_OFFSETS:
        tr, tc = r + dr, c + dc
        if _on_board(tr, tc) and board[tr][tc] == color + "N":
            return True
    for dr, dc in KING_OFFSETS:
        tr, tc = r + dr, c + dc
        if _on_board(tr, tc) and board[tr][tc] == color + "K":
            return True
    for dirs, ptype in ((ROOK_DIRS, "R"), (BISHOP_DIRS, "B")):
        for dr, dc in dirs:
            tr, tc = r + dr, c + dc
            while _on_board(tr, tc):
                t = board[tr][tc]
                if t is not None:
                    if t == color + ptype:
                        return True
                    break
                tr += dr
                tc += dc
    # A pawn of `color` at p attacks p + offset, so look back from the square.
    for odr, odc in _pawn_capture_offsets(color):
        tr, tc = r - odr, c - odc
        if _on_board(tr, tc) and board[tr][tc] == color + "P":
            return True
    return False


def find_king(board: Board, color: str) -> tuple[int, int] | None:
    target = color + "K"
    for r in range(N):
        for c in range(N):
            if board[r][c] == target:
                return r, c
    return None


def kings_attacked_by(board: Board, color: str, active: list[str]) -> set[str]:
    """Active enemy colors whose king `color` currently attacks."""
    out = set()
    for enemy in active:
        if enemy == color:
            continue
        k = find_king(board, enemy)
        if k is not None and is_attacked_by(board, k, color):
            out.add(enemy)
    return out
