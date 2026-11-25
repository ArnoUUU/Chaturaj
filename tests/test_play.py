import json
import time
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from bot.mcts import MCTS, Node, UniformEvaluator
from bot.timing import Clock, move_budget, parse_time_control
from chaturaji.moves import legal_moves
from chaturaji.state import ChaturajiState
from gui import play


def test_parse_time_control():
    assert parse_time_control("1|5") == (60.0, 5.0)
    assert parse_time_control("0.5+2") == (30.0, 2.0)
    assert parse_time_control("3") == (180.0, 0.0)


def test_move_budget_stays_inside_the_clock():
    assert move_budget(60, 5) == pytest.approx(60 / 20 + 4)
    for left in (0.05, 0.3, 1.0, 2.0, 5.0):
        assert move_budget(left, 5) < left


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_clock_charges_time_and_adds_increment(monkeypatch):
    fake = FakeTime()
    clock = Clock.parse("1|5")
    monkeypatch.setattr(clock, "now", fake)
    clock.start("R")
    fake.t += 7.5
    assert clock.left("R") == pytest.approx(52.5)
    assert clock.press("R") == pytest.approx(7.5)
    assert clock.remaining["R"] == pytest.approx(57.5)
    clock.start("B")
    fake.t += 61
    assert clock.flagged() == "B"


def test_timed_search_respects_time_limit():
    mcts = MCTS(UniformEvaluator(), num_simulations=None, max_turn=None)
    t = time.monotonic()
    root = mcts.search(Node(ChaturajiState.initial()), time_limit=0.2, early_stop=False)
    assert 0.15 < time.monotonic() - t < 0.5
    assert root.child_n.sum() == mcts.last_simulations > 1


def test_tree_reuse_returns_searched_child():
    mcts = MCTS(UniformEvaluator(), num_simulations=50, max_turn=None)
    root = mcts.search(Node(ChaturajiState.initial()))
    i = mcts.choose(root)
    assert MCTS.advance(root, root.moves[i]) is root.children[i]


def test_board_view_puts_each_seat_at_the_bottom():
    s = ChaturajiState.initial()
    for seat in "RBYG":
        bottom = [play.board_view(seat)[56 + c] for c in range(8)]
        assert sum(1 for i in bottom if (s.board[i // 8][i % 8] or " ")[0] == seat) == 4


def _session(**kw):
    return play.Session(UniformEvaluator(), save_dir=None, **kw)


def test_human_and_engine_alternate_on_the_clock():
    s = _session(humans={"R"}, time_control="0.1|1")
    try:
        s.arm()
        m = legal_moves(s.game.state)[0]
        assert s.human_move("B", 0, 1) == "not your turn"
        assert s.human_move("R", m.start[0] * 8 + m.start[1], m.end[0] * 8 + m.end[1]) is None
        deadline = time.monotonic() + 15
        while s.game.state.current_player() != "R" and time.monotonic() < deadline:
            time.sleep(0.05)
        snap = s.snapshot()
        assert snap["t"] == "R" and snap["ply"] == 4 and snap["legal"]
        assert snap["clock"]["R"] > 6  # 6 s base + 1 s increment, minus a little
        assert set(snap["engine"]) == {"B", "Y", "G"}
    finally:
        s.close()


def test_human_flag_falls_and_seat_is_eliminated():
    s = _session(humans={"R"}, time_control="0.005|0")  # 0.3 s
    try:
        s.arm()
        deadline = time.monotonic() + 5
        while "R" not in s.flags and time.monotonic() < deadline:
            time.sleep(0.05)
        assert s.flags[:1] == ["R"] and "R" in s.game.state.eliminated
        assert s.game.history[0]["event"] == "timeout"
    finally:
        s.close()


def test_clocks_wait_for_the_page():
    s = _session(humans={"R"}, time_control="0.005|0")
    try:
        time.sleep(0.5)
        assert not s.flags and s.clock.running is None
    finally:
        s.close()


def test_server_serves_page_and_state(monkeypatch):
    monkeypatch.setattr(play, "sprite_set", lambda: ({}, "test"))
    s = _session(humans={"R"}, time_control="1|5")
    server = ThreadingHTTPServer(("127.0.0.1", 0), play.make_handler(s, play.make_page(s)))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Chaturaji" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["t"] == "R" and state["running"] == "R" and len(state["b"]) == 128
        a, b = state["legal"][0]
        req = urllib.request.Request(base + "/api/move", json.dumps({"color": "R", "from": a, "to": b}).encode(),
                                     {"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True}
    finally:
        server.shutdown()
        s.close()
