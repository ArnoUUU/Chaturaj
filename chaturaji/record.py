"""Game wrapper that keeps a move history, for saving and viewing past games.

Training code can use `ChaturajiState` + `apply_move` directly (no history overhead);
use `Game` when you want a game saved for later viewing.
"""

from __future__ import annotations

import json
from pathlib import Path

from .game import MoveInfo, apply_move, eliminate, game_over_reason, winners
from .moves import Move
from .notation import move_table, to_san
from .state import ChaturajiState

RECORD_VERSION = 1


class Game:
    def __init__(self, state: ChaturajiState | None = None, metadata: dict | None = None):
        self.initial = (state or ChaturajiState.initial()).copy()
        self.state = self.initial.copy()
        self.history: list[dict] = []
        self.metadata: dict = dict(metadata or {})

    def play(self, move: Move) -> MoveInfo:
        before = self.state
        after, info = apply_move(before, move)
        self.history.append({
            "turn": before.turn,
            "color": info.color,
            "uci": move.uci,
            "san": to_san(before, move, info),
            "points": info.points,
        })
        self.state = after
        return info

    def play_uci(self, text: str) -> MoveInfo:
        return self.play(Move.from_uci(text, self.state.board))

    def eliminate(self, color: str, reason: str = "resign") -> None:
        """Record a resignation/timeout ("R"/"T" in the move table, like chess.com)."""
        self.history.append({
            "turn": self.state.turn,
            "color": color,
            "uci": None,
            "event": reason,
            "san": "T" if reason == "timeout" else "R",
            "points": 0,
        })
        self.state = eliminate(self.state, color)

    def _replay(self, entry: dict, state: ChaturajiState) -> ChaturajiState:
        if entry.get("uci") is None:
            return eliminate(state, entry["color"])
        return apply_move(state, Move.from_uci(entry["uci"], state.board))[0]

    def result(self) -> dict:
        return {
            "reason": game_over_reason(self.state),
            "scores": dict(self.state.scores),
            "winners": winners(self.state),
        }

    def move_table(self) -> str:
        return move_table(self.history)

    def states(self) -> list[ChaturajiState]:
        """Every state from the start to the current position (len = moves + 1)."""
        out = [self.initial.copy()]
        s = self.initial
        for h in self.history:
            s = self._replay(h, s)
            out.append(s)
        return out

    # --- persistence -----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": RECORD_VERSION,
            "metadata": self.metadata,
            "initial": self.initial.to_dict(),
            "moves": self.history,
            "result": self.result(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Game":
        game = cls(ChaturajiState.from_dict(data["initial"]), data.get("metadata"))
        for h in data["moves"]:
            if h.get("uci") is None:
                game.eliminate(h["color"], h.get("event", "resign"))
            else:
                game.play_uci(h["uci"])
        return game

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Game":
        return cls.from_dict(json.loads(Path(path).read_text()))
