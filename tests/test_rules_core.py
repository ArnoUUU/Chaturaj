"""The compiled rules (engine/src/rules.rs) against the Python rules engine."""
import numpy as np
import pytest

from bot.nnue import CORE_AVAILABLE
from chaturaji import ChaturajiState, legal_moves
from chaturaji.game import apply_move, is_game_over

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="chaturaji_engine not built (see engine/README.md)")

if CORE_AVAILABLE:
    from bot.rules_core import (
        RECORDS, check, check_game, differences, move_code, move_from_code, position_from_state, random_games,
    )


def test_random_games_with_resignations_match():
    for name, initial, entries in random_games(40, seed=3):
        assert check_game(initial, entries, name) == []


def test_scraped_games_match():
    if not RECORDS.is_dir():
        pytest.skip("no scraped records")
    res = check(RECORDS, limit=25, n_random=0)
    assert res["games"] == 25 and res["problems"] == 0, res["first_problems"]


def test_checker_notices_a_difference():
    s = ChaturajiState.initial()
    pos = position_from_state(s)
    s2, _ = apply_move(s, legal_moves(s)[0])
    assert "board" in differences(s2, pos)
    t = s.copy()
    t.scores["Y"] = 4
    assert any(d.startswith("scores") for d in differences(t, pos))


def test_mid_game_state_round_trip_and_unmake():
    rng = np.random.default_rng(9)
    s = ChaturajiState.initial()
    for _ in range(90):
        if is_game_over(s):
            break
        ms = legal_moves(s)
        s, _ = apply_move(s, ms[rng.integers(len(ms))])
    pos = position_from_state(s)
    assert differences(s, pos) == []
    for code in pos.legal_moves()[:10]:
        m = move_from_code(code, s.board)
        assert move_code(m) == code
        after, info = apply_move(s, m)
        points, *_ = pos.push(code)
        assert points == info.points and differences(after, pos) == []
        pos.pop()
        assert differences(s, pos) == []
    with pytest.raises(IndexError):
        pos.pop()


# --- the compiled max^n search ------------------------------------------------------------

import torch

from bot.maxn import MaxN
from bot.nnue import NNUE, NumpyNNUE


def _positions(seed, n=80, every=8):
    rng = np.random.default_rng(seed)
    s, out = ChaturajiState.initial(), []
    for ply in range(n):
        if is_game_over(s):
            break
        if ply % every == every - 2:
            out.append(s)
        ms = legal_moves(s)
        s, _ = apply_move(s, ms[rng.integers(len(ms))])
    return out


@pytest.mark.parametrize("depth,qdepth", [(1, 2), (2, 1), (3, 1)])
def test_compiled_search_matches_python(depth, qdepth):
    torch.manual_seed(5)
    net = NumpyNNUE.from_model(NNUE().eval())
    py = MaxN(net, time_ms=None, depth=depth, qdepth=qdepth, backend="python")
    rs = MaxN(net, time_ms=None, depth=depth, qdepth=qdepth, backend="rust")
    for s in _positions(17) + _positions(23):
        assert py.best_move(s) == rs.best_move(s)
        assert np.allclose(py.stats["value"], rs.stats["value"], atol=2e-3)


def test_compiled_search_time_budget_and_deeper():
    torch.manual_seed(1)
    net = NumpyNNUE.from_model(NNUE().eval())
    rs = MaxN(net, time_ms=200, backend="rust")
    py = MaxN(net, time_ms=200, backend="python")
    s = _positions(5)[3]
    m = rs.best_move(s)
    assert m in legal_moves(s) and rs.stats["seconds"] < 0.4
    py.best_move(s)
    assert rs.stats["depth"] >= py.stats["depth"] and rs.stats["nodes"] > 5 * py.stats["nodes"]
