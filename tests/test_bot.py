import json
import numpy as np
import pytest
import torch

from bot.features import (
    NUM_ACTIONS, NUM_PLANES, PIECE_PLANES, action_to_move, canonical_action, encode,
    legal_canonical, mover_index, to_absolute, to_relative,
)
from bot.mcts import MCTS, NetEvaluator, Node, UniformEvaluator
from bot.model import ChaturajiNet, NetConfig, load_checkpoint, save_checkpoint
from bot.selfplay import play_game
from bot.train import ReplayBuffer, compute_loss
from chaturaji import ChaturajiState, legal_moves


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
    assert root.child_n.sum() == 24
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
    assert len(root.moves) == 8 and root.child_n.sum() == 40
    assert (root.child_n >= 1).all()  # UCB1 tries every move before revisiting any
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


def test_supervised_on_records(tmp_path, monkeypatch):
    from bot import registry, supervised
    from chaturaji.simulate import random_game
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
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
