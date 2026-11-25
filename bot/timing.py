"""Clock and per-move time budget for timed games (chess.com style "minutes | increment").

    clock = Clock.parse("1|5")        # 1 minute each, +5 s after every move
    clock.start("R")
    ...                               # Red thinks
    clock.press("R")                  # charges the elapsed time, adds the increment

Times are in seconds, measured with time.monotonic(). `Clock` holds no lock: callers that
share it between threads (gui/play.py) guard it themselves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from chaturaji.constants import TURN_ORDER


def parse_time_control(text: str) -> tuple[float, float]:
    """'1|5', '1+5' or '1 | 5' -> (60.0, 5.0). Minutes may be fractional ('0.5|2')."""
    for sep in ("|", "+"):
        if sep in text:
            base, inc = text.split(sep, 1)
            return float(base) * 60.0, float(inc)
    return float(text) * 60.0, 0.0


def move_budget(remaining: float, increment: float, moves_to_go: float = 20.0,
                increment_share: float = 0.8, reserve: float = 0.5,
                max_fraction: float = 0.4, minimum: float = 0.05) -> float:
    """Seconds to spend on this move (bot/PLAN.md section 5.7).

    remaining/moves_to_go + 0.8*increment, never more than `max_fraction` of what is
    left, and always keeping `reserve` seconds for move overhead. The increment is only
    added after the move, so it is never spent in advance beyond what is on the clock.
    """
    t = remaining / moves_to_go + increment * increment_share
    t = min(t, remaining * max_fraction, remaining - reserve)
    return max(t, minimum) if remaining > minimum * 2 else minimum / 2


@dataclass
class Clock:
    base: float
    increment: float
    remaining: dict[str, float] = field(default_factory=dict)
    running: str | None = None
    started_at: float = 0.0
    untimed: set[str] = field(default_factory=set)   # seats with no clock (never flag, never add increment)
    now = staticmethod(time.monotonic)

    def __post_init__(self):
        if not self.remaining:
            self.remaining = {c: self.base for c in TURN_ORDER}

    @classmethod
    def parse(cls, text: str) -> "Clock":
        return cls(*parse_time_control(text))

    @property
    def label(self) -> str:
        return f"{self.base / 60:g} | {self.increment:g}"

    def start(self, color: str | None) -> None:
        self.running, self.started_at = color, self.now()

    def left(self, color: str) -> float:
        """Time left for `color`, counting the move in progress (infinite for untimed seats)."""
        if color in self.untimed:
            return math.inf
        t = self.remaining[color]
        if color == self.running:
            t -= self.now() - self.started_at
        return t

    def flagged(self) -> str | None:
        """The running color if its time is up."""
        if self.running is not None and self.left(self.running) <= 0:
            return self.running
        return None

    def press(self, color: str) -> float:
        """`color` finished a move: charge the time used and add the increment.
        Returns the seconds used. Stops the clock (call start() for the next player)."""
        used = self.now() - self.started_at if color == self.running else 0.0
        if color not in self.untimed:
            self.remaining[color] -= used
            if self.remaining[color] > 0:
                self.remaining[color] += self.increment
        self.running = None
        return used

    def stop(self) -> None:
        if self.running is not None and self.running not in self.untimed:
            self.remaining[self.running] = self.left(self.running)
        self.running = None

    def snapshot(self) -> dict[str, float | None]:
        """Seconds left per seat (None for an untimed seat)."""
        return {c: (None if c in self.untimed else round(max(self.left(c), 0.0), 2)) for c in TURN_ORDER}
