"""NNUE architecture: features, incremental accumulators, PyTorch/NumPy agreement."""
import numpy as np
import pytest
import torch

from bot import registry
from bot.features import mover_index, to_relative
from bot.nnue import (
    NNUE, PAIRS, Accumulator, NumpyNNUE, board_diff, collate_nnue, nnue_features,
    pairwise_loss, pairwise_targets, perspective_indices, utilities,
)
from bot.mlp import sparse_features
from chaturaji import ChaturajiState, legal_moves
from chaturaji.game import apply_move, eliminate, is_game_over


def random_game(seed, plies=120):
    rng = np.random.default_rng(seed)
    s = ChaturajiState.initial()
    states = [s]
    for _ in range(plies):
        if is_game_over(s):
            break
        ms = legal_moves(s)
        s, _ = apply_move(s, ms[rng.integers(len(ms))])
        states.append(s)
    return states


def test_mover_perspective_matches_mlp_features():
    for s in random_game(1, 40):
        idx, g = nnue_features(s)
        mlp_idx, mlp_g = sparse_features(s)
        assert sorted(idx[0].tolist()) == sorted(mlp_idx.tolist())
        assert np.array_equal(g, mlp_g)


def test_start_position_is_symmetric_across_perspectives():
    per = perspective_indices(ChaturajiState.initial().board)
    ref = sorted(per[0].tolist())
    assert all(sorted(p.tolist()) == ref for p in per[1:])


def test_parameter_count():
    m = NNUE()
    n = registry.param_count(m)
    assert 1600 * 256 < n < 460_000


def test_incremental_accumulator_matches_refresh_and_torch():
    torch.manual_seed(0)
    model = NNUE().eval()
    net = NumpyNNUE.from_model(model)
    for seed in range(3):
        states = random_game(seed, 200)
        acc = Accumulator(net, states[0].board)
        for s in states[1:]:
            acc.push(s.board)
            assert np.allclose(acc.acc, net.refresh(s.board), atol=1e-4)
        # unwinding the stack returns to earlier positions exactly
        for back in range(1, 6):
            acc.pop()
            assert np.allclose(acc.acc, net.refresh(states[-1 - back].board), atol=1e-4)
        s = states[len(states) - 6]
        with torch.no_grad():
            t = model(*collate_nnue([nnue_features(s)]))[0].numpy()
        assert np.allclose(net.logits(acc.acc, s), t, atol=1e-4)


def test_elimination_is_an_incremental_update_too():
    torch.manual_seed(1)
    net = NumpyNNUE.from_model(NNUE())
    s = random_game(4, 30)[-1]
    acc = Accumulator(net, s.board)
    dead = eliminate(s, "B")
    removed, added = board_diff(s.board, dead.board)
    assert len(removed) >= 5 and len(removed) == len(added)
    acc.push(dead.board)
    assert np.allclose(acc.acc, net.refresh(dead.board), atol=1e-4)


def test_utilities_and_targets():
    t = pairwise_targets(np.array([10, 3, 10, 0]))
    assert t.tolist() == [1.0, 0.5, 1.0, 0.0, 1.0, 1.0]
    u = utilities(t)
    assert np.isclose(u.sum(), 2.0)              # 6 pairs, each splits 1 point between 2 seats
    assert np.isclose(u[0], 2.5 / 3) and np.isclose(u[3], 0.0)


def test_evaluation_is_seat_consistent():
    """Utilities come out in absolute seat order and agree with the mover's own view."""
    torch.manual_seed(2)
    model = NNUE().eval()
    net = NumpyNNUE.from_model(model)
    s = random_game(5, 23)[-1]
    acc = Accumulator(net, s.board)
    u_abs = acc.evaluate(s)
    p = 1 / (1 + np.exp(-net.logits(acc.acc, s)))
    assert np.allclose(to_relative(u_abs, mover_index(s)), utilities(p))
    assert np.isclose(u_abs.sum(), 2.0)


def test_nnue_learns_pairwise_outcomes():
    """A tiny overfit: the architecture can fit pairwise targets from game results."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for seed in range(6):
        states = random_game(seed, 80)
        final = states[-1]
        for s in states[::8]:
            k = mover_index(s)
            sc = to_relative(np.array([final.scores[c] for c in "RBYG"], dtype=np.float32), k)
            xs.append(nnue_features(s))
            ys.append(pairwise_targets(sc))
    y = torch.from_numpy(np.stack(ys))
    model = NNUE()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    inputs = collate_nnue(xs)
    first = None
    for _ in range(300):
        loss = pairwise_loss(model(*inputs), y)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first


def test_registry_round_trip(tmp_path):
    cfg = registry.default_config("nnue")
    m = registry.build_model("nnue", cfg)
    assert registry.size_tag("nnue", cfg) == "a256-h32x32"
    ev = registry.evaluator("nnue", m.eval(), "cpu")
    moves, actions, priors, value = ev(ChaturajiState.initial())
    assert len(moves) == len(actions) == len(priors) and np.isclose(value.sum(), 1.0)
    xs = [registry.encoder("nnue")(s) for s in random_game(0, 5)]
    out = m(*registry.collate("nnue", xs, "cpu"))
    assert out.shape == (len(xs), len(PAIRS))


# --- max^n ------------------------------------------------------------------------------

from bot.maxn import MaxN, exact_utilities
from bot.evaluate import evaluate


def test_exact_utilities():
    s = ChaturajiState.initial()
    s.scores = {"R": 10, "B": 3, "Y": 10, "G": 0}
    u = exact_utilities(s)
    assert np.allclose(u, [2.5 / 3, 1 / 3, 2.5 / 3, 0.0]) and np.isclose(u.sum(), 2.0)


def test_maxn_takes_an_outright_win():
    """If some move ends the game with the mover above all three, max^n plays such a move."""
    torch.manual_seed(0)
    player = MaxN(NumpyNNUE.from_model(NNUE()), time_ms=None, depth=1, qdepth=0)
    found = 0
    for seed in range(40):
        for s in random_game(seed, 300):
            if is_game_over(s):
                continue
            me = "RBYG".index(s.current_player())
            wins = []
            for m in legal_moves(s):
                after, _ = apply_move(s, m)
                if is_game_over(after) and exact_utilities(after)[me] == 1.0:
                    wins.append(m)
            if wins:
                after, _ = apply_move(s, player.best_move(s))
                assert is_game_over(after) and exact_utilities(after)[me] == 1.0
                found += 1
    if not found:
        pytest.skip("no outright-win positions in the sample games")


def test_maxn_plays_legal_games_through_evaluate():
    torch.manual_seed(0)
    player = MaxN(NumpyNNUE.from_model(NNUE()), time_ms=None, depth=1, qdepth=1)
    res = evaluate(None, games=4, opponent=None, max_plies=60, seed=0, bot_player=player)
    assert res["games"] == 4 and 0.0 <= res["pairwise"] <= 1.0
    assert player.stats["depth"] in (0, 1)


def test_maxn_time_budget_and_depth():
    torch.manual_seed(0)
    player = MaxN(NumpyNNUE.from_model(NNUE()), time_ms=150)
    s = random_game(2, 20)[-1]
    import time as _t
    t = _t.perf_counter()
    m = player.best_move(s)
    assert m in legal_moves(s)
    assert _t.perf_counter() - t < 0.6
    assert player.stats["depth"] >= 1


# --- deep presets and the compiled core ---------------------------------------------------

from bot.nnue import CORE_AVAILABLE, PRESETS, board_codes

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="chaturaji_engine not built (see engine/README.md)")


def deep_net(width=48, layers=7):
    torch.manual_seed(3)
    m = NNUE(32, tuple([width] * layers), residual=True).eval()
    with torch.no_grad():                       # random weights large enough to exercise clipping
        for layer in m.layers:
            layer.bias.normal_(0.0, 0.3)
    return m


def test_deep_residual_numpy_matches_torch():
    m = deep_net()
    assert m.skip == [False] + [True] * 6       # the first layer changes width
    net = NumpyNNUE.from_model(m, use_core=False)
    for s in random_game(7, 60)[::10]:
        acc = Accumulator(net, s.board, use_core=False)
        with torch.no_grad():
            t = m(*collate_nnue([nnue_features(s)]))[0].numpy()
        assert np.allclose(acc.logits(s), t, atol=1e-4)


@needs_core
@pytest.mark.parametrize("deep", [False, True])
def test_core_matches_numpy_through_games(deep):
    m = deep_net() if deep else NNUE().eval()
    ref = NumpyNNUE.from_model(m, use_core=False)
    fast = NumpyNNUE.from_model(m)
    assert fast.core is not None and fast.core.num_params == ref.num_params
    states = random_game(11, 200)
    a_ref, a_fast = Accumulator(ref, states[0].board, use_core=False), Accumulator(fast, states[0].board)
    for s in states[1:]:
        a_ref.push(s.board)
        a_fast.push(s.board)
        assert np.allclose(a_fast.acc, a_ref.acc, atol=1e-4)
        assert np.allclose(a_fast.logits(s), a_ref.logits(s), atol=1e-4)
    for _ in range(5):
        a_ref.pop()
        a_fast.pop()
    assert np.allclose(a_fast.acc, a_ref.acc, atol=1e-4)
    assert np.allclose(fast.core.refresh(board_codes(states[3].board)), ref.refresh(states[3].board), atol=1e-4)


@needs_core
def test_core_rejects_bad_input():
    fast = NumpyNNUE.from_model(NNUE())
    acc = fast.core.accumulator(board_codes(ChaturajiState.initial().board))
    with pytest.raises(IndexError):
        acc.pop()
    with pytest.raises(ValueError):
        acc.push(np.zeros(10, dtype=np.uint8))
    with pytest.raises(ValueError):
        acc.logits(0, np.zeros(3, dtype=np.float32))


def test_presets_build_and_registry_tags():
    for name, cfg in PRESETS.items():
        if name == "xl":
            continue                            # 255 MB: built in test_xl_size only by count
        assert registry.size_tag("nnue", cfg).startswith(f"a{cfg['acc']}-h")
    assert registry.size_tag("nnue", PRESETS["large"]) == "a1024-h1024x7-res"
    m = registry.build_model("nnue", PRESETS["medium"])
    assert len(m.layers) == 7 and 3_000_000 < registry.param_count(m) < 4_000_000


@needs_core
def test_batched_children_match_numpy():
    m = deep_net()
    ref, fast = NumpyNNUE.from_model(m, use_core=False), NumpyNNUE.from_model(m)
    s = random_game(13, 30)[-1]
    kids = [apply_move(s, mv)[0] for mv in legal_moves(s)]
    a_ref, a_fast = Accumulator(ref, s.board, use_core=False), Accumulator(fast, s.board)
    for x, y in zip(a_fast.evaluate_children(kids), a_ref.evaluate_children(kids)):
        assert np.allclose(x, y, atol=1e-5)


@needs_core
def test_maxn_same_moves_with_and_without_core():
    torch.manual_seed(5)
    m = NNUE().eval()
    slow = MaxN(NumpyNNUE.from_model(m, use_core=False), time_ms=None, depth=2, qdepth=1)
    fast = MaxN(NumpyNNUE.from_model(m), time_ms=None, depth=2, qdepth=1)
    for s in random_game(17, 60)[8::12]:
        assert slow.best_move(s) == fast.best_move(s)
