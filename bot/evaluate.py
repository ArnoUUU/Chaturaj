"""Evaluation: the bot (MCTS + model) in one seat against three opponents.

Opponents are random movers, or another model (e.g. the supervised warm start or an older
snapshot) searching with the same MCTS settings. The opponent model fills all three other
seats; each seat searches for itself, so the three don't cooperate.

The bot takes each seat in turn (game g -> seat g % 4) so no seat advantage leaks in.
Reported numbers:
  score_share : bot's points / all points (fair = 0.25)
  win_rate    : fraction of games where the bot had the top score (ties count)
  avg_place   : 1 (best) to 4

Compare past models from the command line:
    python -m bot.evaluate models/<new> --opponent models/<supervised> --games 40
    python -m bot.evaluate models/<new> --opponent models/<new>/snapshots/snap_003_*.pt random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chaturaji.constants import TURN_ORDER
from chaturaji.game import apply_move, is_game_over, require_moves
from chaturaji.state import ChaturajiState

from .mcts import MCTS, Node


def evaluate(evaluator, games: int = 4, sims: int = 16, selection: str = "puct",
             opponent=None, opponent_name: str = "random", max_plies: int = 300,
             seed: int = 0) -> dict:
    """`opponent` is an MCTS evaluator for the three other seats, or None for random moves."""
    rng = np.random.default_rng(seed)
    mcts = MCTS(evaluator, num_simulations=sims, selection=selection, max_turn=None, rng=rng)
    opp = (MCTS(opponent, num_simulations=sims, selection=selection, max_turn=None, rng=rng)
           if opponent is not None else None)
    shares, wins, places = [], [], []
    for g in range(games):
        bot = TURN_ORDER[g % 4]
        state = ChaturajiState.initial()
        plies = 0
        while not is_game_over(state) and plies < max_plies:
            if state.current_player() == bot:
                root = mcts.search(Node(state))
                move = root.moves[mcts.choose(root)]
            elif opp is not None:
                root = opp.search(Node(state))
                move = root.moves[opp.choose(root)]
            else:
                moves = require_moves(state)
                move = moves[rng.integers(len(moves))]
            state, _ = apply_move(state, move)
            plies += 1
        scores = state.scores
        total = sum(scores.values())
        shares.append(scores[bot] / total if total else 0.25)
        wins.append(scores[bot] == max(scores.values()))
        places.append(1 + sum(1 for c in TURN_ORDER if scores[c] > scores[bot]))
    return {"opponent": opponent_name, "games": games, "sims": sims, "selection": selection,
            "score_share": round(float(np.mean(shares)), 3),
            "win_rate": round(float(np.mean(wins)), 3),
            "avg_place": round(float(np.mean(places)), 2)}


def evaluate_vs_random(evaluator, games: int = 4, sims: int = 16, selection: str = "puct",
                       max_plies: int = 300, seed: int = 0) -> dict:
    return evaluate(evaluator, games, sims, selection, None, "random", max_plies, seed)


def model_name(path: str | Path) -> str:
    """Short name for a model path: the run label, plus the snapshot name if it is one."""
    p = Path(path)
    if p.name == "model.pt":
        p = p.parent
    if p.parent.name == "snapshots":
        return f"{p.parent.parent.name}/{p.stem}"
    return p.name


def load_evaluator(path: str | Path, device: str = "cpu"):
    from . import registry
    model, kind, _, _ = registry.load_model(path, device)
    model.eval()
    return registry.evaluator(kind, model, device)


def evaluate_vs(evaluator, opponent: str | Path, device: str = "cpu", **kw) -> dict:
    """Evaluate against "random" or a saved model (models/<label>, model.pt or a snapshot)."""
    if str(opponent) == "random":
        return evaluate(evaluator, opponent=None, opponent_name="random", **kw)
    return evaluate(evaluator, opponent=load_evaluator(opponent, device),
                    opponent_name=model_name(opponent), **kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="models/<label>, a model.pt, or a snapshot .pt")
    ap.add_argument("--opponent", nargs="+", default=["random"],
                    help='one or more opponents: "random" or model paths (each is evaluated separately)')
    ap.add_argument("--games", type=int, default=40, help="games per opponent (multiples of 4 balance seats)")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--selection", default="puct")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    ev = load_evaluator(args.model, args.device)
    for opp in args.opponent:
        result = evaluate_vs(ev, opp, args.device, games=args.games, sims=args.sims,
                             selection=args.selection, max_plies=args.max_plies, seed=args.seed)
        print(json.dumps({"model": model_name(args.model), **result}), flush=True)


if __name__ == "__main__":
    main()
