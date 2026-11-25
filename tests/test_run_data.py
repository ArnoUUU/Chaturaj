"""Verbose run records and their loader (bot/run_data.py), plus the per-move records."""
import json

import numpy as np
import pandas as pd
import pytest

from bot import run_data as RD
from bot import selfplay_worker as W
from bot.evaluate import evaluate
from bot.maxn import MaxN
from bot.nnue import NNUE, NumpyNNUE, save_weights
from chaturaji import ChaturajiState

CORE = W.E is not None and hasattr(W.E, "RESIGN")
needs_core = pytest.mark.skipif(not CORE, reason="chaturaji_engine with RESIGN not built (engine/README.md)")
LABEL = "2026-01-01_0000_test"


def _write_run(root):
    models, logs = root / "models" / LABEL, root / "logs" / LABEL
    models.mkdir(parents=True)
    logs.mkdir(parents=True)
    meta = {"label": LABEL, "history": [
        {"iteration": 0, "hours": 0.0, "steps": 0, "lr": 3e-4, "val_base_loss": 0.69},
        {"iteration": 1, "hours": 0.2, "steps": 120, "loss": 0.6, "plies_min": 9,
         "endings": {"last_king": 3, "ply_cap": 1}, "resigns": {"claims": 1, "early": 0},
         "search": {"mean_depth": 2.0, "mean_nodes": 1234.5, "mean_sec": 0.01}}],
        "checks": [{"check": "start", "iteration": 0, "hours": 0.0,
                    "vs_mcts": {"pairwise": 0.5, "pairwise_se": 0.1, "maxn_seconds_per_move": 0.02}}]}
    (models / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    new = {"type": "selfplay", "iteration": 1, "seed": [0, 1, 2], "opening": 2, "scores": [10, 4, 2, 1],
           "reason": "resign", "plies": 4, "moves": [3 * 64 + 12, 1 * 64 + 9, 2 * 64 + 18, 65535],
           "captures": [], "checks": [], "eliminations": [[3, 0, 3]],
           "movers": [0, 1, 2, 3], "kinds": ["opening", "explore", "search", "resign"],
           "values": [None, None, [0.5, 0.2, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
           "depths": [None, None, 2, 2], "nodes": [None, None, 100, 200], "secs": [None, None, 0.01, 0.02],
           "resigns": [{"ply": 3, "seat": 3, "active": 4, "scores": [10, 4, 2, 1]}]}
    old = {"type": "selfplay", "iteration": 1, "seed": [0, 1, 1], "opening": 4, "scores": [3, 3, 0, 0],
           "reason": "ply_cap", "plies": 4, "moves": [3 * 64 + 12, 65535, 0, 0],
           "captures": [], "checks": [], "eliminations": []}
    check = {"type": "check", "check": "start", "vs": "mcts", "seat": "R",
             "opponents": {"B": "mlp", "Y": "mlp", "G": "mlp"}, "scores": {"R": 3, "B": 1, "Y": 0, "G": 0},
             "pairwise": 0.667, "reason": "resign", "plies": 3, "opening_plies": 1,
             "moves": ["d2d4", "b8c6", "resign"],
             "move_info": [{"seat": "R", "by": "opening"},
                           {"seat": "B", "by": "bot", "depth": 2, "nodes": 50, "seconds": 0.01,
                            "value": [0.4, 0.3, 0.2, 0.1]},
                           {"seat": "Y", "by": "opponent", "sims": 8, "value": [0.25] * 4,
                            "q": [0.3, 0.2, 0.25, 0.25]}]}
    with (logs / "games.jsonl").open("w", encoding="utf-8") as f:
        for r in (new, old, check):
            f.write(json.dumps(r) + "\n")
    (logs / "steps.csv").write_text("iteration,step,global_step,loss,grad_norm,lr\n"
                                    "1,1,1,0.7,0.5,0.0003\n1,2,2,0.6,0.4,0.0003\n", encoding="utf-8")
    return root


def test_uci_conversion_matches_the_rules_module():
    assert RD.uci(3 * 64 + 12) == "d8e7"
    assert RD.uci(65535) == "resign"
    from bot.rules_core import move_code, move_from_code
    board = ChaturajiState.initial().board
    for code in (3 * 64 + 12, 2 * 64 + 18):
        assert RD.uci(code) == move_from_code(code, board).uci
    assert move_code(move_from_code(3 * 64 + 12, board)) == 3 * 64 + 12


def test_loader_tables_from_a_synthetic_run(tmp_path):
    root = _write_run(tmp_path)
    t = RD.load(LABEL, root=str(root))
    assert set(t) == set(RD.TABLES)

    it = t["iterations"]
    assert len(it) == 2
    assert it["endings.last_king"].tolist()[1] == 3
    assert it["search.mean_depth"].iloc[1] == 2.0
    assert "endings.last_king" in it.columns and pd.isna(it["loss"].iloc[0])

    ck = t["checks"]
    assert len(ck) == 1 and ck["vs_mcts.pairwise"].iloc[0] == 0.5

    g = t["games"]
    assert g["game_id"].tolist() == [1, 2, 3]
    assert g["winners"].tolist()[:2] == ["R", "R+B"]
    assert g["winners"].iloc[2] == "R"
    assert g["n_resigns"].tolist() == [1, 1, 1]
    assert g["first_elim_ply"].iloc[0] == 3 and pd.isna(g["first_elim_ply"].iloc[1])
    assert g["seed"].iloc[0] == "[0, 1, 2]"
    assert g["score_B"].iloc[2] == 1

    mv = t["moves"]
    assert len(mv) == 4 + 3                       # the old-format game has no per-move records
    assert set(mv["game_id"]) == {1, 3}
    first = mv[mv["game_id"] == 1].reset_index(drop=True)
    assert first["move"].tolist() == ["d8e7", "b8b7", "c8c6", "resign"]
    assert first["seat"].tolist() == ["R", "B", "Y", "G"]
    assert first["kind"].tolist() == ["opening", "explore", "search", "resign"]
    assert first["value_R"].iloc[2] == 0.5 and pd.isna(first["value_R"].iloc[0])
    assert first["value_mover"].iloc[3] == 0.4
    assert first["depth"].iloc[2] == 2 and pd.isna(first["depth"].iloc[0])
    assert first["nodes"].iloc[2] == 100
    chk = mv[mv["game_id"] == 3].reset_index(drop=True)
    assert chk["move"].tolist() == ["d2d4", "b8c6", "resign"]
    assert chk["by"].tolist() == ["opening", "bot", "opponent"]
    assert chk["depth"].iloc[1] == 2 and chk["secs"].iloc[1] == 0.01
    assert chk["sims"].iloc[2] == 8 and chk["q_Y"].iloc[2] == 0.25
    assert chk["value_B"].iloc[1] == 0.3

    st = t["steps"]
    assert st.columns.tolist() == RD.STEP_COLS and len(st) == 2 and st["global_step"].tolist() == [1, 2]

    rs = t["resigns"]
    assert len(rs) == 1
    assert rs["seat"].iloc[0] == "G" and rs["active"].iloc[0] == 4 and rs["score_R"].iloc[0] == 10


def test_loader_handles_an_empty_run(tmp_path):
    (tmp_path / "models" / "empty").mkdir(parents=True)
    (tmp_path / "logs" / "empty").mkdir(parents=True)
    t = RD.load("empty", root=str(tmp_path))
    assert set(t) == set(RD.TABLES)
    for name, df in t.items():
        assert len(df) == 0, name
    assert "value_R" in t["moves"].columns and "resigns" not in t["games"].columns


def test_export_writes_one_csv_per_table(tmp_path):
    root = _write_run(tmp_path / "repo")
    out = tmp_path / "out"
    paths = RD.export(LABEL, out_dir=out, root=str(root))
    assert set(paths) == set(RD.TABLES)
    for name, p in paths.items():
        assert p.is_file() and p.parent == out and p.name == f"{name}.csv"
    assert len(pd.read_csv(paths["moves"])) == 7
    # the default folder is logs/<label>/tables
    default = RD.export(LABEL, root=str(root))
    assert default["games"] == root / "logs" / LABEL / "tables" / "games.csv"
    assert default["games"].is_file()


@needs_core
def test_selfplay_game_records_every_ply(tmp_path):
    net = NNUE(8, (4,)).eval()
    path = tmp_path / "nnue.npz"
    save_weights(net, path)
    task = {"seed": [3, 1, 0], "weights": str(path), "depth": 1, "qdepth": 0, "check_ext": 0,
            "max_turn": 40, "stride": 1, "open_lo": 4, "open_hi": 4, "explore": 0.0, "explore_plies": 0}
    g = W.selfplay_game(task)
    n = g["plies"]
    assert n >= 4
    for k in W.MOVE_FIELDS:
        assert len(g[k]) == n, k
    assert set(g["kinds"]) <= {"opening", "explore", "search", "forced", "resign"}
    assert g["kinds"][:4] == ["opening"] * 4
    assert set(g["movers"]) <= {0, 1, 2, 3}
    for kind, v, d, nd, sec in zip(g["kinds"], g["values"], g["depths"], g["nodes"], g["secs"]):
        if kind in ("opening", "explore", "forced"):
            assert v is None and d is None and nd is None and sec is None
        else:
            assert len(v) == 4 and d == 1 and nd is not None and sec is not None
    assert len(g["moves"]) == n and len(g["scores"]) == 4
    assert len(g["resigns"]) == g["kinds"].count("resign")
    for rs in g["resigns"]:
        assert set(rs) == {"ply", "seat", "active", "scores"} and len(rs["scores"]) == 4
        assert 2 <= rs["active"] <= 4

    m = W.match_game({"cand": str(path), "opp": str(path), "seat": 0, "opening_seed": [1, 2],
                      "opening_plies": 4, "depth": 1, "qdepth": 0, "check_ext": 0, "max_turn": 40})
    assert m["plies"] >= 4 and len(m["moves"]) == m["plies"]
    for k in W.MOVE_FIELDS:
        assert len(m[k]) == m["plies"], k
    assert m["kinds"][:4] == ["opening"] * 4 and m["opening_seed"] == [1, 2]


def test_evaluate_records_move_info_for_a_bot_player():
    player = MaxN(NumpyNNUE.from_model(NNUE()), time_ms=None, depth=1, qdepth=0)
    res = evaluate(None, games=4, sims=2, opponent=None, max_plies=40, seed=0, opening_plies=2,
                   record=True, bot_player=player, bot_stats=lambda: player.stats)
    assert len(res["records"]) == 4
    for rec in res["records"]:
        info = rec["move_info"]
        assert len(info) == len(rec["moves"]) and len(info) > 0
        assert [i["by"] for i in info[:2]] == ["opening", "opening"]
        assert set(i["by"] for i in info) <= {"opening", "bot", "random"}
        bot = [i for i in info if i["by"] == "bot"]
        assert bot and all("depth" in i for i in bot)
        assert all(i["seat"] == rec["seat"] for i in bot)
