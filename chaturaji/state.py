"""Board state for Chaturaji."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import BOARD_SIZE, COLORS, STARTING_LAYOUT, TURN_ORDER

Board = list[list[str | None]]


def square_to_index(square: str) -> tuple[int, int]:
    """'a1' -> (7, 0), 'h8' -> (0, 7)."""
    file, rank = square[0], int(square[1:])
    return BOARD_SIZE - rank, ord(file) - ord("a")


def index_to_square(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def board_key(board: Board) -> str:
    """Board-only key used for threefold repetition."""
    return "/".join("".join(p if p else ".." for p in row) for row in board)


@dataclass
class ChaturajiState:
    board: Board
    turn: int = 0
    scores: dict[str, int] = field(default_factory=lambda: {c: 0 for c in COLORS})
    eliminated: set[str] = field(default_factory=set)
    position_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def initial(cls) -> "ChaturajiState":
        board: Board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for color, pieces in STARTING_LAYOUT.items():
            for square, ptype in pieces.items():
                r, c = square_to_index(square)
                board[r][c] = color + ptype
        state = cls(board=board)
        state.position_counts[board_key(board)] = 1
        return state

    def current_player(self) -> str:
        return TURN_ORDER[self.turn % len(TURN_ORDER)]

    def active_players(self) -> list[str]:
        return [c for c in COLORS if c not in self.eliminated]

    def piece_at(self, row: int, col: int) -> str | None:
        return self.board[row][col]

    def copy(self) -> "ChaturajiState":
        return ChaturajiState(
            board=[row[:] for row in self.board],
            turn=self.turn,
            scores=dict(self.scores),
            eliminated=set(self.eliminated),
            position_counts=dict(self.position_counts),
        )

    def to_dict(self) -> dict:
        return {
            "board": [row[:] for row in self.board],
            "turn": self.turn,
            "scores": dict(self.scores),
            "eliminated": sorted(self.eliminated),
            "position_counts": dict(self.position_counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChaturajiState":
        return cls(
            board=[list(row) for row in data["board"]],
            turn=data["turn"],
            scores=dict(data["scores"]),
            eliminated=set(data["eliminated"]),
            position_counts=dict(data["position_counts"]),
        )

    def __str__(self) -> str:
        lines = []
        for r, row in enumerate(self.board):
            cells = " ".join(p if p else ". " for p in row)
            lines.append(f"{BOARD_SIZE - r} {cells}")
        lines.append("  " + " ".join(f"{chr(ord('a') + c)} " for c in range(BOARD_SIZE)))
        lines.append(f"turn={self.turn} to_move={self.current_player()} "
                     f"scores={self.scores} eliminated={sorted(self.eliminated)}")
        return "\n".join(lines)
