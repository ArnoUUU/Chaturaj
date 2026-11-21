from scraper.game_scraping import add_move_timing, align_chat, game_summary, parse_headers

PGN = '''[GameNr "113031547"]
[TimeControl "1+5"]
[Variant "FFA"]
[RuleVariants "CaptureTheKing EnPassant PointsForMate=3 PromoteTo=R SemiAnonymous"]
[Red "Pamplemouse17"]
[RedElo "2388"]
[Blue "Sarwar03"]
[BlueElo "2225"]
[Yellow "arnav2007"]
[YellowElo "2436"]
[Green "Ytyus_B"]
[GreenElo "2321"]
[Result "Pamplemouse17: 3 - Sarwar03: 27 - Ytyus_B: 25 - arnav2007: 21"]
[Termination "King captured"]

1. e5-e6 .. e9-f9 .. i10-i9 .. j6-i6
'''


def test_game_summary_from_headers():
    s = game_summary(parse_headers(PGN))
    assert s["game_id"] == "113031547"
    assert s["time_control"] == "1+5"
    assert s["anonymity"] == "SemiAnonymous"
    assert s["seats"]["Yellow"] == {"name": "arnav2007", "rating": 2436}
    assert s["final_points"]["Sarwar03"] == 27
    assert s["termination"] == "King captured"


def test_move_timing_gaps_and_midnight_wrap():
    moves = add_move_timing([{"move": "a", "time": "23:59:58"}, {"move": "b", "time": "00:00:03"},
                             {"move": "c", "time": None}, {"move": "d", "time": "00:00:10"}])
    assert [m["seconds_since_prev"] for m in moves] == [None, 5, None, 7]
    assert [m["ply"] for m in moves] == [0, 1, 2, 3]


def test_align_chat_uses_timed_events():
    # shape of game 113028841: a checkmate mid-game, then a resignation and a claimed win
    moves = add_move_timing([
        {"move": "g5-g6", "time": "13:30:03"},
        {"move": "h10xKi9#", "time": "13:31:22"},
        {"move": "f9-f10", "time": "13:43:41"},
        {"move": "R", "time": "13:43:48"},
        {"move": "R", "time": "13:43:51"},
    ])
    chat = [
        {"system": True, "user": None, "text": "Game #113028841 started."},
        {"system": False, "user": "arnav2007", "text": "hi guys"},
        {"system": True, "user": None, "text": "Logik-47 checkmated!"},
        {"system": False, "user": "Logik-47", "text": "Why did I not play f4"},
        {"system": True, "user": None, "text": "daPavKa8 resigned."},
        {"system": True, "user": None, "text": "arnav2007 claimed the win."},
        {"system": False, "user": "arnav2007", "text": "ggs"},
    ]
    out = align_chat(chat, moves)
    hi, why, resign, claim, ggs = out[1], out[3], out[4], out[5], out[6]
    assert (hi["after_ply"], hi["before_ply"], hi["before_time"]) == (None, 1, "13:31:22")
    assert (why["after_time"], why["before_time"]) == ("13:31:22", "13:43:48")
    assert resign["after_ply"] == resign["before_ply"] == 3
    assert claim["after_ply"] == 4
    assert (ggs["after_time"], ggs["before_ply"]) == ("13:43:51", None)


# --- storage ---------------------------------------------------------------------------

import json

import pytest

from scraper import game_scraping as gs
from tests.test_pgn4 import REAL_GAME_113036725

REAL_HEADERS = """[GameNr "113036725"]
[TimeControl "1+5"]
[Variant "FFA"]
[RuleVariants "CaptureTheKing EnPassant PointsForMate=3 PromoteTo=R"]
[Red "OragamiMaster22300"]
[RedElo "1624"]
[Blue "griffinlc"]
[BlueElo "1773"]
[Yellow "chikentenders1"]
[YellowElo "1499"]
[Green "justinberner"]
[GreenElo "1500"]
[Termination "(Blue +3)"]
"""


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "DATA", tmp_path)
    return tmp_path


def real_game_data():
    pgn = REAL_HEADERS + REAL_GAME_113036725
    data = {**game_summary(parse_headers(pgn)), "pgn4": pgn, "moves": [], "chat": []}
    data["moves"] = [{"move": "x", "time": None}] * 46
    return data


def test_tc_slug():
    assert gs.tc_slug("1 | 5") == "1+5"
    assert gs.tc_slug("6s | 3") == "6s+3"
    assert gs.tc_slug("30 sec") == "30s"
    assert gs.tc_slug(None) == "unknown"


def test_placement_and_signature():
    pts = {"a": 13, "b": 25, "c": 0, "d": 13}
    assert [gs.placement(pts, n) for n in "abcd"] == [2, 1, 4, 2]
    row = {"date": "Nov 20 2:14 PM", "players": [{"name": "Zed"}, {"name": "amy"}]}
    assert gs.row_signature(row) == "Nov 20 2:14 PM|amy,zed"


def test_store_game_saves_game_record_and_player_index(data_dir):
    row = {"time_control": "1 | 5", "date": "Nov 20 3:19 PM",
           "players": [{"name": "griffinlc", "rating_change": "+20.1"}, {"name": "OragamiMaster22300"},
                       {"name": "chikentenders1"}, {"name": "justinberner"}]}
    status = gs.store_game(real_game_data(), row, player="griffinlc")
    assert status == "ok"
    assert (data_dir / "games" / "113036725.json").exists()
    rec = json.loads((data_dir / "records" / "113036725.json").read_text())
    assert rec["result"]["scores"] == {"R": 13, "B": 25, "Y": 0, "G": 10}
    entries = gs.read_index("griffinlc", "1 | 5")
    assert len(entries) == 1
    e = entries[0]
    assert (e["seat"], e["rating"], e["points"], e["place"], e["rating_change"]) == ("Blue", 1773, 25, 1, "+20.1")
    assert (data_dir / "players" / "griffinlc" / "1+5" / "summary.json").exists()
    # storing again does not duplicate the index entry, and the row is now "known"
    gs.store_game(real_game_data(), row, player="griffinlc")
    assert len(gs.read_index("griffinlc", "1 | 5")) == 1
    assert gs.row_signature(row) in gs.known_signatures("griffinlc")
    # a different time control goes to its own folder
    gs.add_to_player_index("griffinlc", {**real_game_data(), "game_id": "1"}, {**row, "time_control": "30 sec"})
    assert len(gs.read_index("griffinlc", "30 sec")) == 1


def test_bad_pgn_is_logged_not_raised(data_dir):
    data = {**real_game_data(), "pgn4": "1. d2-d5"}
    assert gs.store_game(data).startswith("saved, record failed")
    assert "record:" in (data_dir / "failed.jsonl").read_text()
