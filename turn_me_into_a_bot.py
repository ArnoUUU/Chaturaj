"""Turn a chess.com player into a bot.

Scrapes the player's past Chaturaji games, trains a bot on them (2-4 epochs, fewer for
bigger samples) and adds the result to the opponent pool (models/opponent_pool/).

    python turn_me_into_a_bot.py --username arnav2007
    python turn_me_into_a_bot.py                      # asks for the username

Needs a chess.com login for the scraper (it opens Firefox once, if no login is saved).
Only blitz games are used by default. Optional: --tc-type rapid (repeatable; blitz, rapid,
bullet, ...), --max-games N (default 300 per time control), --device cpu|cuda.

The new bot appears as models/opponent_pool/<username>_model.pt, so it can be used wherever
a model is expected, e.g. `python -m gui.play --engine <username>_model`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL = ROOT / "models" / "opponent_pool"
COOKIES = ROOT / "scraper" / ".firefox-profile" / "chesscom-cookies.json"
MIN_GAMES = 5          # fewer than this and there is nothing to learn from
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,40}$")


def epochs_for(games: int) -> int:
    """More passes over a small sample, fewer over a big one: 2 to 4."""
    return 4 if games < 100 else 3 if games < 300 else 2


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username", help="your chess.com username (asked for if left out)")
    ap.add_argument("--max-games", type=int, default=300, help="most games to scrape per time control")
    ap.add_argument("--tc-type", action="append", dest="tc_types",
                    help="chess.com time-control type to use (repeatable; default: blitz)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    args = ap.parse_args(argv)

    name = (args.username or input("Your chess.com username: ")).strip()
    if not NAME_RE.match(name):
        raise SystemExit(f"{name!r} does not look like a chess.com username")
    name = name.lower()
    tc_types = args.tc_types or ["blitz"]

    # 1. scrape (incremental: games already saved are skipped)
    if not COOKIES.exists():
        print("No saved chess.com login: opening Firefox so you can log in once.")
        run(["-m", "scraper.game_scraping", "login"])
    scrape = ["-m", "scraper.game_scraping", "player", name, "--max-per-tc", str(args.max_games)]
    for t in tc_types:
        scrape += ["--tc-type", t]
    run(scrape)

    # 2. pick the number of epochs from how many games came back
    from bot import registry, supervised
    from chaturaji.record import Game
    paths = supervised.record_paths(supervised.RECORDS, name, None, tc_types)
    games = 0
    for p in paths:
        try:
            games += supervised.plays_in(Game.load(p), name)
        except Exception:
            pass   # a record that no longer replays; the trainer skips it too
    if games < MIN_GAMES:
        raise SystemExit(f"only {games} usable games found for {name}; need at least {MIN_GAMES}.")
    epochs = epochs_for(games)
    print(f"\n{games} games -> training for {epochs} epochs")

    # 3. train (supervised imitation of the recorded games; the benchmark is left alone)
    train = ["--player", name, "--epochs", str(epochs), "--tag", f"me-as-bot-{name}",
             "--eval-games", "0", "--no-promote"]
    for t in tc_types:
        train += ["--tc-type", t]
    if args.device:
        train += ["--device", args.device]
    label = supervised.run(supervised.parse_args(train))

    # 4. add it to the opponent pool, the same files the other pool models have
    src = registry.model_dir(label)
    POOL.mkdir(parents=True, exist_ok=True)
    dst = POOL / f"{name}_model"
    if dst.with_suffix(".pt").exists():
        print(f"replacing the earlier {dst.with_suffix('.pt').name}")
    shutil.copy2(src / "model.pt", dst.with_suffix(".pt"))
    shutil.copy2(src / "meta.json", dst.with_suffix(".meta.json"))
    print(f"\nDone: {name} is now a bot -> {dst.with_suffix('.pt').relative_to(ROOT)}")
    print(f"Play against it:  python -m gui.play --engine {name}_model")


if __name__ == "__main__":
    main()
