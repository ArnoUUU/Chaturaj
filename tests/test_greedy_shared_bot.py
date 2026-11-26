"""One greedy Bot playing every seat must play like one Bot per seat (the GUI and the match
scripts share a Bot per preset; its transposition table is keyed by the searching seat)."""
import random

from bot.greedy import Bot
from chaturaji.game import apply_action, is_game_over, require_moves
from chaturaji.state import ChaturajiState


def _game(bots, plies=80, seed=3):
    rng = random.Random(seed)
    st = ChaturajiState.initial()
    for _ in range(4):
        st, _ = apply_action(st, rng.choice(require_moves(st)))
    moves = []
    while not is_game_over(st) and st.turn < plies:
        a = bots[st.turn % 4].best_move(st)
        moves.append(str(a))
        st, _ = apply_action(st, a)
    return moves


def test_shared_bot_matches_separate_bots():
    shared = Bot("greedy", depth=3)
    assert _game([shared] * 4) == _game([Bot("greedy", depth=3) for _ in range(4)])
