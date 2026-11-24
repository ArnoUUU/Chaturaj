"""Self-play: MCTS games that produce training samples and viewer-compatible records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chaturaji.game import is_game_over
from chaturaji.record import Game

from .features import encode, mover_index, outcome_vectors, to_relative
from .mcts import MCTS, Node


@dataclass
class Sample:
    x: object                 # encoded position (dense planes for resnet, (indices, globals) for mlp)
    actions: np.ndarray       # canonical action indices of legal moves (int64)
    policy: np.ndarray        # MCTS visit distribution over `actions` (float32)
    value: np.ndarray         # (4,) top-score share, relative seats
    score: np.ndarray         # (4,) final score margin, relative seats


def play_game(mcts: MCTS, temp_plies: int = 30, max_plies: int = 400,
              metadata: dict | None = None, encoder=encode) -> tuple[list[Sample], Game]:
    game = Game(metadata=metadata)
    pending = []  # (planes, actions, policy, mover_k)
    root = Node(game.state)
    while (not is_game_over(game.state) and len(game.history) < max_plies
           and not mcts.is_terminal(game.state)):
        mcts.search(root, add_noise=True)
        policy = MCTS.visit_policy(root).astype(np.float32)
        pending.append((encoder(root.state), root.actions.copy(), policy, mover_index(root.state)))
        temp = 1.0 if len(game.history) < temp_plies else 0.0
        i = mcts.choose(root, temp)
        game.play(root.moves[i])
        # reuse the chosen subtree (it was built from the same state)
        child = root.children[i]
        root = child if child is not None else Node(game.state)
        root.state = game.state

    share, margin = outcome_vectors(game.state)
    samples = [Sample(p, a, pol, to_relative(share, k), to_relative(margin, k))
               for p, a, pol, k in pending]
    game.metadata["result"] = game.result()
    return samples, game


def play_league_game(learner: MCTS, seat_players: dict[str, MCTS], learner_seat: str,
                     temp_plies: int = 30, max_plies: int = 400, metadata: dict | None = None,
                     encoder=encode) -> tuple[list[Sample], Game]:
    """One game with the learner in `learner_seat` and other models in the other seats.

    Every seat runs its own MCTS with its own network (a fresh tree per move, since the
    seats don't share search statistics). Only the learner's moves become training
    samples: its visit distribution is the policy target and the final result the value
    target. The learner adds root noise; everyone samples moves with temperature 1 for
    the first `temp_plies` plies, so games differ even between deterministic opponents.
    """
    game = Game(metadata=metadata)
    pending = []
    while not is_game_over(game.state) and len(game.history) < max_plies \
            and not learner.is_terminal(game.state):
        mover = game.state.current_player()
        mine = mover == learner_seat
        mcts = learner if mine else seat_players[mover]
        root = mcts.search(Node(game.state), add_noise=mine)
        if mine:
            policy = MCTS.visit_policy(root).astype(np.float32)
            pending.append((encoder(root.state), root.actions.copy(), policy, mover_index(root.state)))
        temp = 1.0 if len(game.history) < temp_plies else 0.0
        game.play(root.moves[mcts.choose(root, temp)])
    share, margin = outcome_vectors(game.state)
    samples = [Sample(p, a, pol, to_relative(share, k), to_relative(margin, k))
               for p, a, pol, k in pending]
    game.metadata["result"] = game.result()
    return samples, game
