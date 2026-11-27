"""Reference games from bot/greedy.py, for `arena greedycheck` to replay in C++.

Four separate greedy bots (one per seat, each with its own transposition table, as in
bot.greedy.match) play at a fixed depth after 4 random opening plies. One line per ply:
`<move or resign> <value> <nodes> <depth>` (value/nodes/depth are `-` for random plies).

    python -I tools/greedy_ref.py <out.txt> [games] [depth] [seed] [preset] [max_plies]
"""
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyrules  # noqa: E402

pyrules.load()
from chaturaji.game import apply_action, is_game_over, require_moves  # noqa: E402
from chaturaji.moves import RESIGN  # noqa: E402
from chaturaji.state import ChaturajiState  # noqa: E402


def load_greedy():
    pkg = types.ModuleType("bot")
    pkg.__path__ = [str(pyrules.ROOT / "bot")]
    sys.modules["bot"] = pkg
    mod = types.ModuleType("bot.greedy")
    mod.__file__ = str(pyrules.ROOT / "bot" / "greedy.py")
    mod.__package__ = "bot"
    sys.modules["bot.greedy"] = mod
    exec(compile(Path(mod.__file__).read_text(encoding="utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def main():
    out = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    preset = sys.argv[5] if len(sys.argv) > 5 else "greedy"
    max_plies = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    g = load_greedy()
    rng = random.Random(seed)
    lines = []
    for n in range(games):
        lines.append(f"game {n}")
        bots = [g.Bot(preset, depth=depth) for _ in range(4)]
        st = ChaturajiState.initial()
        ply = 0
        while not is_game_over(st) and ply < max_plies:
            if ply < 4:
                act = rng.choice(require_moves(st))
                lines.append(f"{act.uci} - - -")
            else:
                bot = bots[st.turn % 4]
                act = bot(st)
                s = bot.stats
                lines.append(f"{'resign' if act is RESIGN else act.uci} {s['value']} {s['nodes']} {s['depth']}")
            st, _ = apply_action(st, act)
            ply += 1
        print(f"game {n}: {ply} plies, scores {st.scores}", flush=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
