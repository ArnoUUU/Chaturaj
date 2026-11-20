"""Minimal terminal viewer for saved games.

    python -m chaturaji.viewer path/to/game.json

Commands: Enter or n = next move, p = previous, <number> = jump to that ply,
e = end, s = start, t = full move table, q = quit.
"""

from __future__ import annotations

import sys

from .notation import COLOR_NAMES
from .record import Game


def render(game: Game, states, ply: int) -> str:
    s = states[ply]
    lines = [str(s)]
    if ply > 0:
        h = game.history[ply - 1]
        lines.append(f"Last: {COLOR_NAMES[h['color']]} {h['san']} (+{h['points']})")
    lines.append(f"Ply {ply}/{len(game.history)}")
    if ply == len(game.history):
        lines.append(f"Result: {game.result()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return
    game = Game.load(argv[0])
    states = game.states()
    ply = 0
    print(game.move_table(), "\n")
    while True:
        print(render(game, states, ply))
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            return
        if cmd in ("", "n"):
            ply = min(ply + 1, len(game.history))
        elif cmd == "p":
            ply = max(ply - 1, 0)
        elif cmd == "s":
            ply = 0
        elif cmd == "e":
            ply = len(game.history)
        elif cmd == "t":
            print(game.move_table())
        elif cmd == "q":
            return
        elif cmd.isdigit():
            ply = max(0, min(int(cmd), len(game.history)))
        print()


if __name__ == "__main__":
    main()
