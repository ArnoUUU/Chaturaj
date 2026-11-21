"""Quick test-run evaluation: the bot (MCTS + model) against random players.

The bot takes each seat in turn (game g -> seat g % 4) so no seat advantage leaks in.
Reported numbers:
  score_share : bot's points / all points (fair = 0.25)
  win_rate    : fraction of games where the bot had the top score (ties count)
  avg_place   : 1 (best) to 4
"""

from __future__ import annotations

import numpy as np

from chaturaji.constants import TURN_ORDER
from chaturaji.game import apply_move, is_game_over, require_moves
from chaturaji.state import ChaturajiState

from .mcts import MCTS, Node


def evaluate_vs_random(evaluator, games: int = 4, sims: int = 16, selection: str = "puct",
                       max_plies: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mcts = MCTS(evaluator, num_simulations=sims, selection=selection, max_turn=None, rng=rng)
    shares, wins, places = [], [], []
    for g in range(games):
        bot = TURN_ORDER[g % 4]
        state = ChaturajiState.initial()
        plies = 0
        while not is_game_over(state) and plies < max_plies:
            if state.current_player() == bot:
                root = mcts.search(Node(state))
                move = root.moves[mcts.choose(root)]
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
    return {"opponent": "random", "games": games, "sims": sims, "selection": selection,
            "score_share": round(float(np.mean(shares)), 3),
            "win_rate": round(float(np.mean(wins)), 3),
            "avg_place": round(float(np.mean(places)), 2)}
