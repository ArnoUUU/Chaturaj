"""The play server (gui/play.py): seats, turns, clocks and the HTTP API.

Bots here are UniformEvaluator searches with a few simulations, so the tests stay fast and
use no trained model.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from chaturaji.constants import TURN_ORDER
from gui import play


def wait_for(predicate, timeout=20.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def bank():
    return play.EngineBank()


def make_session(bank, seats, time_control="1|5", **kw):
    kw.setdefault("save_dir", None)
    kw.setdefault("max_sims_per_move", 8)
    kw.setdefault("armed", True)   # the page normally arms the clocks; tests do it directly
    return play.Session(bank, seats, time_control, **kw)


def first_legal(snap):
    start, end = snap["legal"][0]
    return start, end


def test_parse_seats_accepts_human_uniform_and_rejects_unknown(monkeypatch):
    seats = play.parse_seats({"R": "human", "B": "uniform", "Y": "", "G": "human"})
    assert seats == {"R": None, "B": "uniform", "Y": None, "G": None}
    with pytest.raises(ValueError):
        play.parse_seats({"R": "no-such-model"})


def test_board_view_puts_seat_at_bottom():
    for seat in TURN_ORDER:
        view = play.board_view(seat)
        assert sorted(view) == list(range(64))


def test_human_moves_then_uniform_bot_replies(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        snap = s.snapshot()
        assert snap["t"] == "R" and snap["legal"]
        start, end = first_legal(snap)
        assert s.human_move("R", start, end) is None
        assert wait_for(lambda: s.snapshot()["ply"] >= 2), "the bot never moved"
        snap = s.snapshot()
        assert "B" in snap["engine"] and snap["engine"]["B"]["bot"] == "Uniform"
        assert snap["t"] != "B"
    finally:
        s.close()


def test_human_move_errors(bank):
    s = make_session(bank, {"R": None, "B": None, "Y": None, "G": None})
    try:
        snap = s.snapshot()
        start, end = first_legal(snap)
        assert s.human_move("B", start, end) == "not your turn"   # Red moves first
        assert s.human_move("R", 0, 0) == "illegal move"
        assert s.human_move("R", start, end) is None
    finally:
        s.close()


def test_resign_removes_a_person_and_only_people_can_resign(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        assert s.resign("B") == "only a person still in the game can resign"
        assert s.resign("R") is None
        snap = s.snapshot()
        assert "R" in snap["e"] and snap["result"] is None   # three sides still play on
        assert s.resign("R") == "only a person still in the game can resign"
    finally:
        s.close()


def test_flag_fall_eliminates_the_side_to_move(bank):
    # 0.01 minutes = 0.6 seconds on the clock; nobody moves, so Red's clock runs out
    s = make_session(bank, {"R": None, "B": None, "Y": None, "G": None}, time_control="0.01|0")
    try:
        assert wait_for(lambda: "R" in s.flags)
        snap = s.snapshot()
        assert "R" in snap["e"] and snap["flags"] == ["R"]
        assert snap["clock"]["R"] == 0
    finally:
        s.close()


def test_clocks_wait_for_the_page_when_people_play(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"}, armed=False)
    try:
        assert s.snapshot()["running"] is None
        s.arm()
        assert s.snapshot()["running"] == "R"
    finally:
        s.close()


def test_api_setup_state_and_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(play, "SAVE_DIR", tmp_path / "played")
    bank = play.EngineBank()
    app = play.App(bank, play.make_assets("builtin"), max_sims=8)
    server = ThreadingHTTPServer(("127.0.0.1", 0), play.make_handler(app))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        code, bots = call("/api/bots")
        assert code == 200 and bots["options"][0]["value"] == "human"
        assert any(o["value"] == "uniform" for o in bots["options"])
        code, assets = call("/api/assets")
        assert code == 200 and set(assets["views"]) == set(TURN_ORDER) and assets["sprites"]

        assert call("/api/state") == (200, {"idle": True})
        code, err = call("/api/new", {"seats": {"R": "human", "B": "no-such-bot"}, "time": "1|5"})
        assert code == 400 and "unknown bot" in err["error"]
        code, err = call("/api/new", {"seats": {"R": "human", "B": "uniform", "Y": "uniform", "G": "uniform"},
                                      "time": "soon"})
        assert code == 400
        code, ok = call("/api/new", {"seats": {"R": "human", "B": "uniform", "Y": "uniform", "G": "uniform"},
                                     "time": "1|5"})
        assert code == 200 and ok == {"ok": True}

        code, state = call("/api/state")
        assert code == 200 and state["t"] == "R" and state["humans"] == ["R"]
        assert [s["bot"] for s in state["seats"]] == [None, "uniform", "uniform", "uniform"]
        start, end = state["legal"][0]
        assert call("/api/move", {"color": "R", "from": start, "to": 0}) == (200, {"error": "illegal move"})
        assert call("/api/move", {"color": "R", "from": start, "to": end}) == (200, {"ok": True})
        assert wait_for(lambda: call("/api/state")[1]["ply"] >= 2)

        assert call("/api/resign", {"color": "R"}) == (200, {"ok": True})
        assert call("/api/close", {}) == (200, {"ok": True})
        assert call("/api/state") == (200, {"idle": True})
        assert call("/api/move", {"color": "R", "from": 0, "to": 1})[0] == 409
    finally:
        app.stop()
        server.shutdown()
        server.server_close()


def test_pause_stops_clock_and_bots_then_resumes(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        move = first_legal(s.snapshot())                       # take the move before pausing
        assert s.set_paused(True) is None
        snap = s.snapshot()
        assert snap["paused"] and snap["running"] is None and snap["legal"] == []
        assert s.human_move("R", *move) == "the game is paused"
        time.sleep(0.6)
        assert s.snapshot()["ply"] == 0                       # no bot move while paused
        assert s.set_paused(False) is None
        assert s.snapshot()["running"] == "R"
        assert s.set_paused(False) is None                    # resuming twice is harmless
    finally:
        s.close()


def test_pausing_a_bot_game_halts_it(bank):
    # uniform bots play a fixed game (it ends at ply 37), so pause early and check it finishes
    s = make_session(bank, {"R": "uniform", "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        assert wait_for(lambda: s.snapshot()["ply"] >= 3)
        s.set_paused(True)
        ply = s.snapshot()["ply"]
        time.sleep(0.8)
        assert s.snapshot()["ply"] == ply
        s.set_paused(False)
        assert wait_for(lambda: s.snapshot()["result"] is not None), "the game did not finish after resuming"
    finally:
        s.close()


def test_change_seat_mid_game_hands_the_side_to_a_person(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        assert s.set_seat("B", None) is None                  # Blue becomes a person
        assert s.set_seat("R", "no-such-bot") == "unknown bot for Red: no-such-bot"
        assert s.set_seat("X", None) == "unknown side"
        snap = s.snapshot()
        assert snap["humans"] == ["R", "B"] and snap["seats"][1]["human"]
        assert s.set_seat("R", "uniform") is None             # Red becomes a bot
        snap = s.snapshot()
        assert snap["humans"] == ["B"]
        assert wait_for(lambda: s.snapshot()["ply"] >= 1), "the new bot never moved"
    finally:
        s.close()


def test_untimed_people_never_flag_and_show_no_clock(bank):
    s = make_session(bank, {"R": None, "B": None, "Y": None, "G": None},
                     time_control="0.01|0", human_clock=False)
    try:
        time.sleep(1.2)                                       # past the 0.6 s the clock would have had
        snap = s.snapshot()
        assert snap["flags"] == [] and snap["result"] is None
        assert snap["clock"] == {"R": None, "B": None, "Y": None, "G": None}
        start, end = first_legal(snap)
        assert s.human_move("R", start, end) is None
    finally:
        s.close()


def test_review_shows_earlier_positions_and_refuses_moves(bank):
    s = make_session(bank, {"R": None, "B": "uniform", "Y": "uniform", "G": "uniform"})
    try:
        start, end = first_legal(s.snapshot())
        assert s.human_move("R", start, end) is None
        assert wait_for(lambda: s.snapshot()["ply"] >= 2)
        live = s.snapshot()
        assert live["view"] is None and len(live["moves"]) == live["ply"]
        start_pos = s.snapshot(0)
        assert start_pos["view"] == 0 and start_pos["b"] != live["b"]
        assert start_pos["legal"] == [] and start_pos["last"] is None
        one = s.snapshot(1)
        assert one["view"] == 1 and one["last"] is not None
        assert s.snapshot(live["ply"])["view"] is None       # the latest ply is the live game
        assert s.snapshot(live["ply"])["b"] == live["b"]
    finally:
        s.close()


def test_api_pause_seat_and_review(tmp_path, monkeypatch):
    bank = play.EngineBank()
    app = play.App(bank, play.make_assets("builtin"), max_sims=8)
    server = ThreadingHTTPServer(("127.0.0.1", 0), play.make_handler(app))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        seats = {"R": "human", "B": "uniform", "Y": "uniform", "G": "uniform"}
        assert call("/api/new", {"seats": seats, "time": "1|5", "human_clock": False})[0] == 200
        state = call("/api/state")[1]
        assert state["human_clock"] is False and state["clock"]["R"] is None
        assert call("/api/pause", {"paused": True}) == (200, {"ok": True})
        assert call("/api/state")[1]["paused"] is True
        assert call("/api/pause", {"paused": False}) == (200, {"ok": True})
        assert call("/api/seat", {"color": "B", "value": "human"}) == (200, {"ok": True})
        assert call("/api/state")[1]["humans"] == ["R", "B"]
        assert call("/api/seat", {"color": "B", "value": "nope"}) == (200, {"error": "unknown bot for Blue: nope"})
        code, review = call("/api/state?ply=0")                 # no moves yet: ply 0 is the live game
        assert code == 200 and review["view"] is None and review["ply"] == 0
        assert call("/api/state?ply=abc")[0] == 400
        assert call("/api/close", {}) == (200, {"ok": True})
    finally:
        app.stop()
        server.shutdown()
        server.server_close()
