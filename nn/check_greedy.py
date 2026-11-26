"""Check the Rust greedy port (engine/src/greedy.rs) against bot/greedy.py: four greedy bots
(each its own Bot) play a whole game from the same random opening in both, and every move
and root value is compared.

    python nn/check_greedy.py --games 3 --depth 4
"""
import argparse, random, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BIN = ROOT / "engine/target/release/nnarena"


def code(m):
    from chaturaji.moves import RESIGN
    if m is RESIGN:
        return 65535
    (sr, sc), (er, ec) = m.start, m.end
    return (sr * 8 + sc) * 64 + er * 8 + ec


def one(args):
    seed, depth, plies, preset = args
    from bot.greedy import Bot
    from chaturaji.game import apply_action, is_game_over, require_moves
    from chaturaji.state import ChaturajiState
    rng = random.Random(seed)
    st = ChaturajiState.initial()
    opening = []
    for _ in range(4):
        m = rng.choice(require_moves(st))
        opening.append(code(m))
        st, _ = apply_action(st, m)
    rust = subprocess.run([str(BIN), "greedy-game", "--depth", str(depth), "--plies", str(plies), "--preset", preset,
                           "--moves", *map(str, opening)], capture_output=True, text=True, check=True).stdout.split()
    bots = [Bot(preset, depth=depth) for _ in range(4)]
    t = time.time()
    i = 0
    while not is_game_over(st) and st.turn < plies:
        k = st.turn % 4
        a = bots[k].best_move(st)
        rm, rv = rust[i].split(":")
        if int(rm) != code(a) or abs(float(rv) - bots[k].stats["value"]) > 0.011:
            return f"seed {seed}: MISMATCH at ply {st.turn} (move {i}): python {code(a)} {bots[k].stats['value']} rust {rm} {rv}"
        st, _ = apply_action(st, a)
        i += 1
    extra = "" if i == len(rust) else f" (rust played {len(rust)} moves)"
    return f"seed {seed}: {i} moves identical, scores {st.scores}, {time.time() - t:.0f}s{extra}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--plies", type=int, default=400)
    ap.add_argument("--preset", default="greedy")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, [(a.seed0 + s, a.depth, a.plies, a.preset) for s in range(a.games)]):
            print(r, flush=True)
