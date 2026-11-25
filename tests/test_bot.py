import json
import numpy as np
import pytest
import torch

from bot.features import (
    NUM_ACTIONS, NUM_PLANES, PIECE_PLANES, action_to_move, canonical_action, encode,
    legal_canonical, mover_index, to_absolute, to_relative,
)
from bot.evaluate import evaluate
from bot.mcts import MCTS, NetEvaluator, Node, UniformEvaluator
from bot.model import ChaturajiNet, NetConfig, load_checkpoint, save_checkpoint
from bot.selfplay import play_game
from bot.train import ReplayBuffer, compute_loss
from chaturaji import RESIGN, ChaturajiState, legal_moves
from chaturaji.game import eliminate
from chaturaji.moves import Resign


def seat_view(k: int) -> ChaturajiState:
    s = ChaturajiState.initial()
    s.turn = k
    return s


def test_canonical_view_is_seat_invariant():
    """The start layout is rotationally symmetric, so every seat sees the same board."""
    ref = encode(seat_view(0))[:PIECE_PLANES]
    for k in range(1, 4):
        assert np.array_equal(encode(seat_view(k))[:PIECE_PLANES], ref)
    # and the mover's king is on d1 (row 7, col 3) in canonical coords
    assert ref[0 * 5 + 0, 7, 3] == 1


def test_canonical_actions_match_across_seats():
    sets = []
    for k in range(4):
        s = seat_view(k)
        moves, idx = legal_canonical(s)
        sets.append(set(idx.tolist()))
        for m, a in zip(moves, idx):
            assert action_to_move(int(a), k, s) == m
    assert sets[0] == sets[1] == sets[2] == sets[3]


def test_relative_absolute_roundtrip():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    for k in range(4):
        assert np.allclose(to_absolute(to_relative(v, k), k), v)
    assert to_relative(v, 1)[0] == 0.2  # Blue to move -> "me" is Blue


def test_model_shapes_and_checkpoint(tmp_path):
    model = ChaturajiNet(NetConfig(16, 1, 32))
    x = torch.from_numpy(np.stack([encode(seat_view(0)), encode(seat_view(1))]))
    p, v, sc = model(x)
    assert p.shape == (2, NUM_ACTIONS) and v.shape == (2, 4) and sc.shape == (2, 4)
    save_checkpoint(tmp_path / "m.pt", model)
    m2, _ = load_checkpoint(tmp_path / "m.pt")
    model.eval(); m2.eval()
    assert torch.allclose(model(x)[0], m2(x)[0])


@pytest.mark.parametrize("evaluator", ["uniform", "net"])
def test_mcts_returns_legal_move(evaluator):
    ev = UniformEvaluator() if evaluator == "uniform" else NetEvaluator(ChaturajiNet(NetConfig(16, 1, 32)).eval())
    mcts = MCTS(ev, num_simulations=24, rng=np.random.default_rng(0))
    s = ChaturajiState.initial()
    root = mcts.search(Node(s), add_noise=True)
    assert MCTS.real_visits(root) == 24
    assert root.moves[mcts.choose(root)] in legal_moves(s)


def test_mcts_finds_free_king_capture():
    """A capture that ends the game as a win should be found even with a neutral evaluator."""
    from tests.test_chaturaji import make_state
    s = make_state({"a1": "RR", "a8": "BK", "e1": "RK", "h6": "XP"}, eliminated={"Y", "G"})
    s.scores = {"R": 0, "B": 2, "Y": 0, "G": 0}  # Red must capture the king (+3) to win
    mcts = MCTS(UniformEvaluator(), num_simulations=400, rng=np.random.default_rng(0))
    root = mcts.search(Node(s))
    best = root.moves[mcts.choose(root)]
    assert best.uci == "a1a8"


def test_selfplay_and_train_step():
    model = ChaturajiNet(NetConfig(16, 1, 32))
    model.eval()
    mcts = MCTS(NetEvaluator(model), num_simulations=4, max_turn=24, rng=np.random.default_rng(0))
    samples, game = play_game(mcts, temp_plies=4, max_plies=24)
    assert len(samples) == len(game.history) > 0
    assert samples[0].x.shape == (NUM_PLANES, 8, 8)
    assert np.isclose(samples[0].policy.sum(), 1.0) and np.isclose(samples[0].value.sum(), 1.0)
    buf = ReplayBuffer(1000)
    buf.extend(samples)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    loss, parts = compute_loss(model, buf.batch(8, "cpu"))
    opt.zero_grad(); loss.backward(); opt.step()
    assert np.isfinite(parts["total"])


def test_mlp_evaluator_skeleton():
    from bot.mlp import ChaturajiMLP, MLPEvaluator, batch_features, sparse_features
    s = ChaturajiState.initial()
    idx, g = sparse_features(s)
    assert len(idx) == 32 and g.shape == (16,)
    # rotation: every seat sees the same start position
    for k in range(1, 4):
        t = ChaturajiState.initial(); t.turn = k
        assert sorted(sparse_features(t)[0]) == sorted(idx)
    model = ChaturajiMLP(embed=32, widths=(64, 32, 16)).eval()
    p, v, sc = model(*batch_features([s, s]))
    assert p.shape == (2, 4096) and v.shape == (2, 4) and sc.shape == (2, 4)
    mcts = MCTS(MLPEvaluator(model), num_simulations=16, rng=np.random.default_rng(0))
    root = mcts.search(Node(s))
    assert root.moves[mcts.choose(root)] in legal_moves(s)


# --- UCB1, registry, chunked logs, training run ---------------------------------------

def test_ucb1_tries_every_move_then_finds_win():
    from tests.test_chaturaji import make_state
    s = ChaturajiState.initial()
    mcts = MCTS(UniformEvaluator(), num_simulations=40, selection="ucb1", rng=np.random.default_rng(0))
    root = mcts.search(Node(s))
    assert len(root.moves) == 9 and MCTS.real_visits(root) == 40  # 8 moves + resign
    assert (root.child_n[:-1] >= 1).all()  # UCB1 tries every move before revisiting any
    s = make_state({"a1": "RR", "a8": "BK", "e1": "RK", "h6": "XP"}, eliminated={"Y", "G"})
    s.scores = {"R": 0, "B": 2, "Y": 0, "G": 0}
    mcts = MCTS(UniformEvaluator(), num_simulations=300, selection="ucb1", rng=np.random.default_rng(0))
    root = mcts.search(Node(s))
    assert root.moves[mcts.choose(root)].uci == "a1a8"


def test_bad_selection_rejected():
    with pytest.raises(ValueError):
        MCTS(UniformEvaluator(), selection="nope")


def test_registry_label_save_load(tmp_path):
    from bot import registry
    cfg = {"embed": 16, "widths": [32, 16, 8], "dropout": 0.0}
    label = registry.make_label("mlp", cfg, "ucb1", 16, 2, "cpu", "unit")
    assert label.endswith("_unit_mlp-e16-w32x16x8_ucb1-s16_it2_cpu")
    model = registry.build_model("mlp", cfg)
    registry.save_model(model, "mlp", cfg, label, {"status": "complete", "kind": "mlp"}, root=tmp_path)
    registry.register(label, {"kind": "mlp", "status": "complete"}, root=tmp_path)
    m2, kind, cfg2, ckpt = registry.load_model(tmp_path / label)
    assert kind == "mlp" and cfg2 == cfg and ckpt["label"] == label
    assert "unit" in (tmp_path / "index.jsonl").read_text()


def test_chunked_logger(tmp_path):
    from bot.registry import ChunkedLogger
    log = ChunkedLogger("run", chunk_lines=3, root=tmp_path, echo=False)
    for i in range(7):
        log.log("tick", i=i)
    files = sorted((tmp_path / "run").glob("log_*.jsonl"))
    assert [f.name for f in files] == ["log_0000.jsonl", "log_0001.jsonl", "log_0002.jsonl"]
    assert len(files[-1].read_text().splitlines()) == 1


@pytest.mark.parametrize("kind,selection", [("mlp", "ucb1"), ("resnet", "puct")])
def test_training_run_saves_labelled_model(tmp_path, monkeypatch, kind, selection):
    from bot import registry, train as train_mod
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    args = train_mod.parse_args([
        "--tag", "unit", "--device", "cpu", "--model", kind, "--selection", selection,
        "--embed", "16", "--widths", "32", "16", "8", "--channels", "8", "--blocks", "1",
        "--iterations", "1", "--games", "1", "--sims", "4", "--max-plies", "24",
        "--steps", "2", "--batch", "8", "--eval-games", "1"])
    label = train_mod.train(args)
    meta = json.loads((tmp_path / "models" / label / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["evaluation"]["games"] == 1
    assert (tmp_path / "logs" / label / "log_0000.jsonl").exists()
    assert list((tmp_path / "logs" / label / "games").glob("*.json"))


def test_timed_run_stops_on_budget_and_keeps_snapshots(tmp_path, monkeypatch):
    import types
    from bot import registry, train as train_mod
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    clock = {"t": 1_000_000.0}

    def fake_time():  # every clock read moves time on by one minute
        clock["t"] += 60
        return clock["t"]
    monkeypatch.setattr(train_mod, "time", types.SimpleNamespace(time=fake_time))
    label = train_mod.train(train_mod.parse_args([
        "--tag", "unit", "--device", "cpu", "--embed", "16", "--widths", "32", "16", "8",
        "--iterations", "0", "--hours", "1", "--snapshot-every", "10",
        "--games", "1", "--sims", "4", "--max-plies", "16", "--steps", "3", "--batch", "8",
        "--eval-games", "0"]))
    assert "_timed_" in label
    meta = json.loads((tmp_path / "models" / label / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["history"]
    snaps = sorted((tmp_path / "models" / label / "snapshots").glob("snap_*.pt"))
    assert len(snaps) >= 2 and len(snaps) == len(meta["snapshots"])
    assert snaps[0].name.startswith("snap_001_")
    _, kind, _, ckpt = registry.load_model(snaps[-1])  # snapshots load / resume like model.pt
    assert kind == "mlp" and ckpt["label"] == label
    with pytest.raises(SystemExit):  # unlimited iterations need a time budget
        train_mod.train(train_mod.parse_args(["--device", "cpu", "--iterations", "0"]))


def test_supervised_on_records(tmp_path, monkeypatch):
    from bot import registry, supervised
    from chaturaji.simulate import random_game
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    rec = tmp_path / "records"
    rec.mkdir()
    for seed in range(3):
        g = random_game(seed, max_plies=40)
        g.metadata["seats"] = {"Red": {"name": "me"}, "Blue": {"name": "b"},
                               "Yellow": {"name": "y"}, "Green": {"name": "g"}}
        g.save(rec / f"{seed}.json")
    game = supervised.Game.load(rec / "0.json")
    samples = supervised.samples_from_game(game, registry.encoder("mlp"))
    assert len(samples) == sum(1 for h in game.history if h["uci"])
    assert all(np.isclose(s.policy.sum(), 1.0) and s.policy.max() > 0.8 for s in samples)
    mine = supervised.samples_from_game(game, registry.encoder("mlp"), only_player="me")
    assert 0 < len(mine) < len(samples)
    label = supervised.run(supervised.parse_args([
        "--records", str(rec), "--tag", "unit", "--device", "cpu", "--embed", "16",
        "--widths", "32", "16", "8", "--epochs", "2", "--batch", "16", "--eval-games", "1",
        "--eval-sims", "4"]))
    meta = json.loads((tmp_path / "models" / label / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["data"]["games"] == 3
    assert "top1" in meta["history"][-1]["val"]
    # self-play can start from it under a new label
    from bot import train as train_mod
    new = train_mod.train(train_mod.parse_args([
        "--init", str(tmp_path / "models" / label), "--tag", "rl", "--device", "cpu",
        "--iterations", "1", "--games", "1", "--sims", "4", "--max-plies", "20",
        "--steps", "1", "--batch", "8", "--eval-games", "0"]))
    assert new != label and (tmp_path / "models" / label / "model.pt").exists()


def test_evaluate_against_saved_model_and_resolve_opponent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bot import evaluate, registry, train
    from bot.mcts import UniformEvaluator

    res = evaluate.evaluate(UniformEvaluator(), games=4, sims=2, opponent=UniformEvaluator(),
                            opponent_name="old", max_plies=12)
    assert res["opponent"] == "old" and res["games"] == 4 and 0 <= res["score_share"] <= 1
    assert evaluate.model_name("models/run_a/snapshots/snap_001.pt") == "run_a/snap_001"
    assert evaluate.model_name("models/run_a/model.pt") == "run_a"

    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)

    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path)
    args = lambda **kw: SimpleNamespace(**{"eval_opponent": None, "init": None, "resume": None, **kw})
    assert train.resolve_eval_opponent(args(init="models/sup"), "x") == "models/sup"
    assert train.resolve_eval_opponent(args(init="models/sup", eval_opponent="random"), "x") == "random"
    assert train.resolve_eval_opponent(args(), "x") == "benchmark"
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "meta.json").write_text('{"eval_opponent": "models/sup"}')
    assert train.resolve_eval_opponent(args(resume="models/run"), "run") == "models/sup"


def test_supervised_plays_in_checks_seats():
    from bot.supervised import plays_in
    from chaturaji.record import Game
    g = Game(metadata={"seats": {"Red": {"name": "Ihavethesauce"}, "Blue": {"name": "taran0915"}}})
    assert plays_in(g, "ihavethesauce") and plays_in(g, "TARAN0915")
    assert not plays_in(g, "arnav2007") and not plays_in(Game(metadata={}), "x")


def test_evaluate_paired_openings_and_per_seat():
    from bot import evaluate
    from bot.mcts import UniformEvaluator
    res = evaluate.evaluate(UniformEvaluator(), games=8, sims=2, opponent=UniformEvaluator(),
                            max_plies=16, opening_plies=4)
    assert set(res["per_seat_share"]) == {"R", "B", "Y", "G"} and len(res["game_shares"]) == 8
    assert res["score_share_se"] is not None
    assert sum(res["wins_draws_losses"]) == 8 * 3 and 0 <= res["pairwise"] <= 1
    w, d, l = res["wins_draws_losses"]
    assert abs(res["pairwise"] - (w + 0.5 * d) / 24) < 1e-3


def test_benchmark_verdict_and_ladder(tmp_path, monkeypatch):
    from bot import benchmark, evaluate, registry
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path)
    good = {"games": 40, "pairwise": 0.62, "pairwise_se": 0.04,
            "per_seat_pairwise": {"R": 0.6, "B": 0.7, "Y": 0.66, "G": 0.45}}
    assert benchmark.verdict(good)[0]
    assert not benchmark.verdict({**good, "pairwise_se": 0.07})[0]              # within noise
    assert not benchmark.verdict({**good, "per_seat_pairwise": {"R": 0.9, "B": 0.9, "Y": 0.4, "G": 0.4}})[0]
    assert not benchmark.verdict({**good, "games": 20})[0]

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.pt").write_bytes(b"x")
        (tmp_path / name / "meta.json").write_text("{}")
    assert benchmark.challenge("models/a")["promoted"]          # first model becomes benchmark
    assert benchmark.current()["label"] == "a"
    monkeypatch.setattr(benchmark, "load_evaluator", lambda path, device="cpu": None)
    monkeypatch.setattr(benchmark, "evaluate", lambda *a, **k: {**good, "games": 40})
    out = benchmark.challenge("models/b/model.pt")
    assert out["promoted"] and benchmark.current() == {**benchmark.current(), "label": "b", "previous": "a"}
    assert json.loads((tmp_path / "b" / "meta.json").read_text()) == {}          # model files untouched
    assert benchmark.challenge("models/a", promote=False)["promoted"] is False
    assert len((tmp_path / "benchmark_history.jsonl").read_text().splitlines()) == 4
    assert benchmark.current()["label"] == "b"                              # not promoted


def test_league_game_trains_only_on_learner_moves():
    import numpy as np
    from bot.mcts import MCTS, UniformEvaluator
    from bot.selfplay import play_league_game
    rng = np.random.default_rng(0)
    mk = lambda: MCTS(UniformEvaluator(), num_simulations=2, max_turn=None, rng=rng)
    samples, game = play_league_game(mk(), {c: mk() for c in "RYG"}, "B", temp_plies=4,
                                     max_plies=20, metadata={"learner_seat": "B"})
    blue_moves = sum(1 for h in game.history if h["color"] == "B" and h.get("uci"))
    assert len(samples) == blue_moves > 0 and game.metadata["learner_seat"] == "B"


def test_league_training_with_hourly_checks(tmp_path, monkeypatch):
    import types
    from bot import registry, train as train_mod
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    tiny = ["--device", "cpu", "--embed", "16", "--widths", "32", "16", "8", "--sims", "2",
            "--max-plies", "16", "--steps", "2", "--batch", "8", "--eval-games", "0"]
    opp = train_mod.train(train_mod.parse_args(["--tag", "opp", "--iterations", "1", "--games", "1"] + tiny))
    clock = {"t": 1_000_000.0}

    def fake_time():
        clock["t"] += 60
        return clock["t"]
    monkeypatch.setattr(train_mod, "time", types.SimpleNamespace(time=fake_time))
    label = train_mod.train(train_mod.parse_args([
        "--tag", "league", "--init", str(tmp_path / "models" / opp), "--iterations", "0",
        "--hours", "1", "--snapshot-every", "20", "--games", "2",
        "--opponents", str(tmp_path / "models" / opp), str(tmp_path / "models" / opp),
        "--check-games", "4", "--check-sims", "2", "--check-opponent", str(tmp_path / "models" / opp)]
        + tiny))
    meta = json.loads((tmp_path / "models" / label / "meta.json").read_text())
    assert meta["opponents"] and meta["history"][0]["league_games"] == 2
    checks = meta["checks"]
    assert checks[0]["snapshot"] == "start" and "vs_yardstick" in checks[0]
    assert len(checks) == len(meta["snapshots"]) + 1 >= 2
    assert all("decision" in c and "vs_best" in c for c in checks[1:])
    assert meta["best_snapshot"] in ["start"] + [s["name"] for s in meta["snapshots"]]
    assert checks[-1]["snapshot"].startswith("final") and "vs_pool" in checks[0]
    games = [json.loads(l) for l in (tmp_path / "logs" / label / "games.jsonl").read_text().splitlines()]
    train_games = [g for g in games if g["type"] == "train"]
    assert train_games and all(g["moves"] and g["opponents"] and len(g["results"]) == 3 for g in train_games)
    seats = [g["learner_seat"] for g in train_games]
    for i in range(0, len(seats) - len(seats) % 4, 4):   # every block of 4 uses each seat once
        assert sorted(seats[i:i + 4]) == sorted("RBYG")
    assert {g["check"] for g in games if g["type"] == "check"} >= {"vs_pool", "vs_yardstick"}


def test_league_pool_grows_once_learner_beats_it(tmp_path, monkeypatch):
    import types
    from bot import benchmark, registry, train as train_mod
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    tiny = ["--device", "cpu", "--embed", "16", "--widths", "32", "16", "8", "--sims", "2",
            "--max-plies", "16", "--steps", "2", "--batch", "8", "--eval-games", "0"]
    opp = train_mod.train(train_mod.parse_args(["--tag", "opp", "--iterations", "1", "--games", "1"] + tiny))
    clock = {"t": 1_000_000.0}

    def fake_time():
        clock["t"] += 60
        return clock["t"]
    monkeypatch.setattr(train_mod, "time", types.SimpleNamespace(time=fake_time))
    monkeypatch.setattr(benchmark, "verdict", lambda r, min_games=40: (True, "forced"))
    label = train_mod.train(train_mod.parse_args([
        "--tag", "grow", "--init", str(tmp_path / "models" / opp), "--iterations", "0",
        "--hours", "1", "--snapshot-every", "20", "--games", "2",
        "--opponents", str(tmp_path / "models" / opp), "--check-games", "4", "--check-sims", "2",
        "--check-opponent", "none", "--no-revert"] + tiny))
    meta = json.loads((tmp_path / "models" / label / "meta.json").read_text())
    assert meta["pool_open"] and meta["pool_added"]
    later = [c for c in meta["checks"] if c.get("added_to_pool")]
    assert later and len(later[-1]["vs_pool"]["pool"]) >= 2


def test_model_references_resolve_pool_and_archive(tmp_path, monkeypatch):
    from bot import benchmark, registry
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "RUNS_DIR", tmp_path / "logs" / "models")
    monkeypatch.setattr(registry, "LOGS_DIR", tmp_path / "logs")
    pool = tmp_path / "models" / "opponent_pool"
    pool.mkdir(parents=True)
    (pool / "BEST_x_model.pt").write_bytes(b"x")
    (pool / "BEST_x_model.meta.json").write_text(json.dumps({"label": "2026_x_run"}))
    (tmp_path / "logs" / "models" / "old_run").mkdir(parents=True)
    (tmp_path / "logs" / "models" / "old_run" / "model.pt").write_bytes(b"x")

    assert registry.find_model("opponent_pool/BEST_x_model.pt") == pool / "BEST_x_model.pt"
    assert registry.find_model("opponent_pool/x_model.pt") == pool / "BEST_x_model.pt"  # BEST_ optional
    assert registry.find_model("models/old_run") == tmp_path / "logs" / "models" / "old_run" / "model.pt"
    assert registry.label_for(pool / "BEST_x_model.pt") == "2026_x_run"
    with pytest.raises(FileNotFoundError):
        registry.find_model("nope")

    benchmark.set_benchmark(pool / "BEST_x_model.pt", "test")
    bench = benchmark.current()
    assert bench["model"] == "opponent_pool/BEST_x_model.pt" and bench["label"] == "2026_x_run"
    assert benchmark.model_file() == pool / "BEST_x_model.pt"


def _two_left(**scores):
    """Red and Blue left to play; Yellow and Green resigned (their dead kings stay)."""
    s = eliminate(eliminate(ChaturajiState.initial(), "Y"), "G")
    for color, pts in scores.items():
        s.scores[color] = pts
    return s


def test_mcts_claims_the_win_when_ahead():
    s = _two_left(R=50, B=0)  # resigning gives Blue 3 per dead king: 3 kings -> 9 < 50
    mcts = MCTS(UniformEvaluator(), num_simulations=32, max_turn=None)
    root = mcts.search(Node(s))
    assert isinstance(root.moves[-1], Resign)
    assert root.moves[mcts.choose(root)] is RESIGN
    assert root.children[-1].terminal and root.children[-1].value[0] == 1.0
    assert root.priors.sum() == pytest.approx(1.0)


def test_mcts_does_not_claim_the_win_when_it_would_lose():
    s = _two_left(R=50, B=0, Y=100)  # a frozen score of 100 wins if Red resigns
    mcts = MCTS(UniformEvaluator(), num_simulations=32, max_turn=None)
    root = mcts.search(Node(s))
    assert root.moves[mcts.choose(root)] is not RESIGN
    assert root.children[-1].value[0] == 0.0


def test_three_players_resign_only_at_the_root():
    mcts = MCTS(UniformEvaluator(), num_simulations=16, max_turn=None)
    s = ChaturajiState.initial()
    root = mcts.search(Node(s))
    i = mcts.choose(root)
    assert root.moves[i] is RESIGN or root.moves[i] in legal_moves(s)
    assert isinstance(root.moves[-1], Resign)
    for child in root.children:
        if child is not None and child.expanded:
            if len(child.state.active_players()) >= 3:
                assert not MCTS._has_resign(child)


def test_reused_subtree_gets_its_resign_child():
    mcts = MCTS(UniformEvaluator(), num_simulations=16, max_turn=None)
    root = mcts.search(Node(ChaturajiState.initial()))
    j = next(k for k, c in enumerate(root.children) if c is not None and not isinstance(root.moves[k], Resign))
    child = MCTS.advance(root, root.moves[j])
    assert child is root.children[j] and not MCTS._has_resign(child)
    mcts.search(child)
    assert isinstance(child.moves[-1], Resign)
    mcts.search(child)  # never added twice
    assert sum(isinstance(m, Resign) for m in child.moves) == 1


def test_advance_follows_resign():
    mcts = MCTS(UniformEvaluator(), num_simulations=0, max_turn=None)
    root = mcts.search(Node(_two_left(R=50, B=0)))
    assert MCTS.advance(root, RESIGN) is root.children[-1]


def test_resign_can_be_turned_off():
    mcts = MCTS(UniformEvaluator(), num_simulations=8, max_turn=None, resign=False)
    root = mcts.search(Node(_two_left(R=50, B=0)))
    assert not any(isinstance(m, Resign) for m in root.moves)


def test_temperature_samples_resign_only_when_it_is_the_argmax():
    mcts = MCTS(UniformEvaluator(), num_simulations=0, max_turn=None, rng=np.random.default_rng(0))
    root = mcts.search(Node(ChaturajiState.initial()))
    r = len(root.moves) - 1
    assert isinstance(root.moves[r], Resign)
    root.child_n[:] = 0
    root.child_n[-1] = 1           # the pre-visit only: resign has 0 real visits
    root.child_n[0] = 3
    assert {mcts.choose(root, 1.0) for _ in range(200)} == {0}
    root.child_n[-1] = 11          # 10 real visits: resign is the argmax
    assert {mcts.choose(root, 1.0) for _ in range(20)} == {r}
    root.child_n[-1] = 4           # tie at 3 real visits: a move wins
    assert mcts.choose(root) == 0


def test_policy_targets_ignore_the_resign_entry():
    mcts = MCTS(UniformEvaluator(), num_simulations=16, max_turn=None)
    root = mcts.search(Node(ChaturajiState.initial()))
    n = len(root.moves) - 1
    assert len(MCTS.visit_policy(root)) == n and MCTS.visit_policy(root).sum() == pytest.approx(1)
    assert len(MCTS.move_actions(root)) == n and (MCTS.move_actions(root) >= 0).all()
    samples, _ = play_game(MCTS(UniformEvaluator(), num_simulations=4, max_turn=None), max_plies=3)
    assert samples
    for smp in samples:
        assert (smp.actions >= 0).all() and len(smp.actions) == len(smp.policy)
        assert smp.policy.sum() == pytest.approx(1, abs=1e-5)


def test_evaluate_records_a_bot_resignation():
    res = evaluate(UniformEvaluator(), games=4, sims=2, bot_player=lambda state: RESIGN, record=True)
    assert len(res["records"]) == 4
    for rec in res["records"]:
        assert rec["moves"].count("resign") == 1
        assert rec["reason"] != "ply_cap"


def test_policy_falls_back_to_priors_when_only_resign_was_visited():
    mcts = MCTS(UniformEvaluator(), num_simulations=0, max_turn=None)
    root = mcts.search(Node(_two_left(R=50, B=0)))
    root.child_n[:] = 0
    root.child_n[-1] = 5           # every simulation went to the resign child
    pol = MCTS.visit_policy(root)
    assert np.isfinite(pol).all() and pol.sum() == pytest.approx(1)
    assert np.allclose(pol, root.priors[:-1] / root.priors[:-1].sum())
