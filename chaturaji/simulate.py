"""Play random games, e.g. to produce sample records for the viewer.

    python -m chaturaji.simulate --games 1 --seed 0 --out games/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .game import is_game_over, require_moves
from .record import Game


def random_game(seed: int | None = None, max_plies: int = 2000) -> Game:
    rng = random.Random(seed)
    game = Game(metadata={"players": "random", "seed": seed})
    while not is_game_over(game.state) and len(game.history) < max_plies:
        game.play(rng.choice(require_moves(game.state)))
    return game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.games):
        g = random_game(args.seed + i)
        path = out / f"game_{args.seed + i}.json"
        g.save(path)
        print(f"{path}: {len(g.history)} plies, {g.result()}")


if __name__ == "__main__":
    main()
