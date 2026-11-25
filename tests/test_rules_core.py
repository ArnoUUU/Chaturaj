"""The compiled rules (engine/src/rules.rs) against the Python rules engine."""
import numpy as np
import pytest

from bot.nnue import CORE_AVAILABLE
from chaturaji import ChaturajiState, legal_moves
from chaturaji.game import apply_move, is_game_over
from chaturaji.moves import RESIGN

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="chaturaji_engine not built (see engine/README.md)")

if CORE_AVAILABLE:
    from bot.rules_core import (
        RECORDS, check, check_game, differences, move_code, move_from_code, position_from_state, random_games,
    )


def test_move_from_code_maps_the_resign_code():
    assert move_from_code(65535, ChaturajiState.initial().board) is RESIGN


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


@pytest.mark.parametrize("depth,qdepth,width,full,prefs", [(1, 2, 0, 2, False), (2, 1, 0, 2, False),
                                                           (2, 2, 0, 2, False), (3, 1, 3, 1, False),
                                                           (4, 1, 2, 1, False), (2, 2, 0, 2, True),
                                                           (3, 1, 3, 1, True)])
def test_compiled_search_matches_python(depth, qdepth, width, full, prefs):
    """Same move and root value; selective depth (width) and the game-length / elimination
    preferences (exaggerated here so they matter) in the later cases."""
    torch.manual_seed(5)
    net = NumpyNNUE.from_model(NNUE().eval())
    kw = dict(time_ms=None, depth=depth, qdepth=qdepth, width=width, full_plies=full)
    if prefs:
        kw.update(discount=0.97, elim_bonus=0.2, elim_horizon=150)
    py = MaxN(net, backend="python", **kw)
    rs = MaxN(net, backend="rust", **kw)
    exact = n = 0
    for s in _positions(17)[:6] + _positions(23)[:3]:
        a, b = py.best_move(s), rs.best_move(s)
        assert a == b or np.allclose(py.stats["value_full"], rs.stats["value_full"], atol=1e-6)
        assert np.allclose(py.stats["value"], rs.stats["value"], atol=2e-3)
        exact += a == b
        n += 1
    assert exact >= n - 1


def test_compiled_search_time_budget_and_deeper():
    torch.manual_seed(1)
    net = NumpyNNUE.from_model(NNUE().eval())
    rs = MaxN(net, time_ms=200, backend="rust")
    py = MaxN(net, time_ms=200, backend="python")
    s = _positions(5)[3]
    m = rs.best_move(s)
    assert (m is RESIGN or m in legal_moves(s)) and rs.stats["seconds"] < 0.4
    py.best_move(s)
    assert rs.stats["depth"] >= py.stats["depth"] and rs.stats["nodes"] > 5 * py.stats["nodes"]


def test_features_from_codes_match_python_features():
    from bot.nnue import features_from_codes, nnue_features
    from bot.rules_core import position_from_state
    for s in _positions(31, 120, 7):
        pos = position_from_state(s)
        want, g = nnue_features(s)
        got = features_from_codes(np.asarray(pos.board), pos.current_player)
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
        assert np.array_equal(np.asarray(pos.globals(pos.current_player)), g)


def test_gives_check_matches_python():
    from chaturaji.moves import kings_attacked_by
    from bot.rules_core import move_code, position_from_state
    n = 0
    for s in _positions(41, 160, 3):
        pos = position_from_state(s)
        before = kings_attacked_by(s.board, s.current_player(), s.active_players())
        for m in legal_moves(s):
            _, info = apply_move(s, m)
            want = bool(info.checked - before)
            assert pos.gives_check(move_code(m)) == want
            n += want
    assert n > 0


def test_preferences_shape_values():
    """discount pulls every value towards 0.5; the elimination bonus goes to the mover only,
    shrinking with the turn and gone after the horizon."""
    net = NumpyNNUE.from_model(NNUE().eval())
    p = MaxN(net, time_ms=None, depth=1, discount=0.9, elim_bonus=0.01, elim_horizon=200, backend="python")
    v = np.array([1.0, 0.0, 0.5, 0.5])
    assert np.allclose(p._shape(v, 0, 0, False), [0.95, 0.05, 0.5, 0.5])
    assert np.allclose(p._shape(v, 1, 50, True), [0.95, 0.05 + 0.0075, 0.5, 0.5])
    assert np.allclose(p._shape(v, 1, 300, True), [0.95, 0.05, 0.5, 0.5])
    with pytest.raises(ValueError):
        MaxN(net, time_ms=None, depth=1, discount=0.0)


def test_elimination_bonus_finds_a_king_capture():
    """With a large bonus the search takes an available king capture (an elimination)."""
    torch.manual_seed(2)
    net = NumpyNNUE.from_model(NNUE().eval())
    found = 0
    for seed in range(40):
        for s in _positions(seed):
            kids = MaxN._expand(s, legal_moves(s))
            if any(e[3] for e in kids) and not all(e[3] for e in kids):
                for backend in ("python", "rust"):
                    p = MaxN(net, time_ms=None, depth=1, qdepth=0, elim_bonus=5.0, elim_horizon=10_000,
                             backend=backend, resign=False)     # a resignation isn't a capture
                    m = p.best_move(s)
                    assert dict((e[0], e[3]) for e in kids)[m], backend
                found += 1
                break
        if found >= 3:
            break
    assert found >= 1
