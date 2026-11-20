"""Chess-style algebraic notation for Chaturaji moves.

Rules, mirroring regular SAN:
- Pieces: K, B, N, R (boat) + destination, e.g. "Nc3". Pawns: destination only, e.g. "d3".
- Captures use "x": "Bxe5". Pawn captures prefix the pawn's origin file for Red/Yellow
  (vertical movers) or origin rank for Blue/Green (horizontal movers): "cxd3", "6xc7".
- Disambiguation like chess: file, then rank, then full square when needed.
- Promotion: "=R".
- Suffix "#" when the move captures an active king (eliminates a player), otherwise "+"
  when the mover attacks at least one active enemy king after the move.
- A skipped turn (eliminated player) is shown as "--" in move tables.
"""

from __future__ import annotations

from .constants import PAWN_DIRECTION, TURN_ORDER
from .game import MoveInfo
from .moves import Move, legal_moves
from .state import ChaturajiState, index_to_square

COLOR_NAMES = {"R": "Red", "B": "Blue", "Y": "Yellow", "G": "Green"}


def to_san(state: ChaturajiState, move: Move, info: MoveInfo) -> str:
    """SAN for `move`, given the state *before* the move and the move's info."""
    board = state.board
    (sr, sc), (er, ec) = move.start, move.end
    piece = board[sr][sc]
    color, ptype = piece[0], piece[1]
    dest = index_to_square(er, ec)
    capture = board[er][ec] is not None
    origin = index_to_square(sr, sc)

    if ptype == "P":
        if capture:
            vertical = PAWN_DIRECTION[color][0] != 0
            san = (origin[0] if vertical else origin[1:]) + "x" + dest
        else:
            san = dest
    else:
        rivals = [m for m in legal_moves(state, color)
                  if m.end == move.end and m.start != move.start
                  and board[m.start[0]][m.start[1]] == piece]
        dis = ""
        if rivals:
            if all(m.start[1] != sc for m in rivals):
                dis = origin[0]
            elif all(m.start[0] != sr for m in rivals):
                dis = origin[1:]
            else:
                dis = origin
        san = ptype + dis + ("x" if capture else "") + dest

    if move.promotion:
        san += "=" + move.promotion
    if info.eliminated:
        san += "#"
    elif info.checked:
        san += "+"
    return san


def move_table(history: list[dict], col_width: int = 10) -> str:
    """Render a move list with one column per color, like a 4-column chess scoresheet.

    `history` entries need "turn", "color" and "san" keys. Turns that were skipped
    (eliminated players) show "--".
    """
    by_turn = {h["turn"]: h["san"] for h in history}
    if not by_turn:
        return ""
    last_round = max(by_turn) // 4
    header = " " * 5 + "".join(COLOR_NAMES[c].ljust(col_width) for c in TURN_ORDER)
    lines = [header.rstrip()]
    for rnd in range(last_round + 1):
        cells = []
        for i in range(4):
            t = rnd * 4 + i
            if t in by_turn:
                cells.append(by_turn[t])
            elif t < max(by_turn):
                cells.append("--")
            else:
                cells.append("")
        lines.append(f"{rnd + 1:>3}. " + "".join(c.ljust(col_width) for c in cells).rstrip())
    return "\n".join(lines)
