"""Pool growth, probing, climbers, drop-outs, and the top-20 fallback. Mocked prober only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from helpers import MockClient, load  # noqa: E402
from leaderboards import pool as pool_mod  # noqa: E402
from leaderboards import update  # noqa: E402
from leaderboards.fetch import parse_watch  # noqa: E402

TOP = 5
D = [f"2026-12-{i:02d}" for i in range(1, 6)]


def row(uid, name, rating):
    return (uid, name, rating)


def make_truth(day_lists: dict) -> dict:
    return day_lists


def run_day(tmp, client, day, **kw):
    client.day = day
    return update.run(tmp, client, date=day, top_n=TOP, seed_min=0, **kw)


# --- pool growth and climbers ------------------------------------------------------------

def climber_truth():
    base = [row(f"u{i}", f"P{i}", 2500 - i * 20) for i in range(1, 9)]   # P1..P8
    day1 = list(base)                                                 # P8 is rank 8
    day2 = [base[0], base[1], base[7]] + [b for b in base[2:7]]        # P8 climbs to rank 3
    day3 = list(day2)
    return {D[0]: day1, D[1]: day2, D[2]: day3}


def test_climber_outside_top_n_is_in_pool_but_not_in_history(tmp_path):
    client = MockClient(climber_truth(), leader_n=1)
    run_day(tmp_path, client, D[0])
    players = load(tmp_path / "players.json")
    assert "u8" not in players                       # rank 8 > TOP: no history
    pool = load(tmp_path / "pool.json")
    assert pool["u8"]["in_pool"] is True and pool["u8"]["last_rank"] == 8


def test_climber_enters_when_probed_into_top_n(tmp_path):
    client = MockClient(climber_truth(), leader_n=1)
    run_day(tmp_path, client, D[0])
    summary = run_day(tmp_path, client, D[1])
    assert "P8" in summary["entered"]
    p = load(tmp_path / "players.json")["u8"]
    assert [h[1] for h in p["history"]] == [3]        # first on-board day is rank 3
    assert p["times_entered"] == 0 and p["currently_on_board"] is True


def test_pool_grows_from_watch_list_without_leaders(tmp_path):
    # the server's leaders show only P1; everyone else is found through the watch list
    client = MockClient(climber_truth(), leader_n=1)
    summary = run_day(tmp_path, client, D[0])
    assert summary["probes_sent"] == 8
    pool = load(tmp_path / "pool.json")
    assert sorted(pool) == sorted(f"u{i}" for i in range(1, 9))
    assert all(v["in_pool"] for v in pool.values())


def test_rank_above_250_leaves_the_pool(tmp_path):
    # 300 players; P260 is ranked 260 and must be dropped from the pool
    truth = [row(f"u{i}", f"P{i}", 3000 - i) for i in range(1, 301)]
    client = MockClient({D[0]: truth}, leader_n=1)
    run_day(tmp_path, client, D[0])
    pool = load(tmp_path / "pool.json")
    assert pool["u150"]["in_pool"] is True          # inside 250
    assert pool["u260"]["in_pool"] is False         # outside 250: kept for the record, not probed as a member
    assert "u260" not in [r["uid"] for r in load(tmp_path / "players.json").values()]


def test_dropout_from_pool_pauses_history(tmp_path):
    base = [row(f"u{i}", f"P{i}", 2500 - i * 20) for i in range(1, 7)]
    gone = [r for r in base if r[0] != "u2"]
    day2 = gone[:1] + [row("u2", "P2", 2400)] + gone[1:]   # P2 at rank 2 on D1 later
    truth = {D[0]: base, D[1]: gone, D[2]: gone, D[3]: day2}
    client = MockClient(truth, leader_n=1)
    for d in D[:4]:
        run_day(tmp_path, client, d)
    p2 = load(tmp_path / "players.json")["u2"]
    assert [h[0] for h in p2["history"]] == [D[0], D[3]]
    assert p2["times_entered"] == 1 and p2["left_date"] is None
    assert p2["change_1d"] is None                 # first day back


# --- fallback ----------------------------------------------------------------------------

def test_probe_failure_publishes_top20_and_leaves_stats_alone(tmp_path):
    truth = climber_truth()
    client = MockClient(truth, leader_n=1, fail_probe_days=[D[2]])
    run_day(tmp_path, client, D[0])
    run_day(tmp_path, client, D[1])
    players_before = (tmp_path / "players.json").read_text(encoding="utf-8")
    pool_before = (tmp_path / "pool.json").read_text(encoding="utf-8")
    records_before = (tmp_path / "records.json").read_text(encoding="utf-8")

    summary = run_day(tmp_path, client, D[2])
    assert summary["coverage"] == "top20"
    assert summary["warnings"] and "pool probe failed" in summary["warnings"][0]

    assert (tmp_path / "players.json").read_text(encoding="utf-8") == players_before
    assert (tmp_path / "pool.json").read_text(encoding="utf-8") == pool_before
    assert (tmp_path / "records.json").read_text(encoding="utf-8") == records_before
    assert not (tmp_path / "daily" / f"{D[2]}.json").exists()
    assert load(tmp_path / "index.json")["latest"] == D[1]

    today = load(tmp_path / "today.json")
    assert today["coverage"] == "top20" and today["stats_updated"] is False
    assert today["count"] == 1 and today["players"][0]["username"] == "P1"


def test_pool_run_after_fallback_on_same_day_is_pool(tmp_path):
    truth = climber_truth()
    failing = MockClient(truth, leader_n=1, fail_probe_days=[D[0]])
    run_day(tmp_path, failing, D[0])
    ok = MockClient(truth, leader_n=1)
    summary = run_day(tmp_path, ok, D[0])
    assert summary["coverage"] == "pool"
    assert load(tmp_path / "today.json")["coverage"] == "pool"
    assert (tmp_path / "daily" / f"{D[0]}.json").exists()


def test_watch_failure_does_not_stop_the_run(tmp_path):
    class NoWatch(MockClient):
        def watch(self):
            raise TimeoutError("no watch_games reply")

    client = NoWatch(climber_truth(), leader_n=3)
    summary = run_day(tmp_path, client, D[0])
    assert summary["coverage"] == "pool"
    assert any("watch list unavailable" in w for w in summary["warnings"])


def test_short_pool_writes_with_warning(tmp_path):
    client = MockClient(climber_truth(), leader_n=1)
    summary = run_day(tmp_path, client, D[0])
    today = load(tmp_path / "today.json")
    assert today["rows_expected"] == TOP and today["count"] == 5
    assert summary["warnings"] == []   # 5 of 5 found: nothing to warn about


# --- pure helpers -------------------------------------------------------------------------

def test_gather_skips_recently_probed_saved_game_players():
    pool = {"9": {"uid": "9", "username": "old", "last_probed": "2026-12-01", "in_pool": False, "last_rank": None},
            "8": {"uid": "8", "username": "stale", "last_probed": "2026-11-01", "in_pool": False, "last_rank": None}}
    saved = [{"uid": "9", "username": "old", "rating": 2500},
             {"uid": "8", "username": "stale", "rating": 2400}]
    cands = pool_mod.gather_candidates(pool, [], [], [], saved, "2026-12-03", seed_min=2000, max_new=10)
    assert [c["uid"] for c in cands] == ["8"]


def test_gather_caps_new_saved_game_players():
    saved = [{"uid": str(i), "username": f"s{i}", "rating": 2000 + i} for i in range(10)]
    cands = pool_mod.gather_candidates({}, [], [], [], saved, "2026-12-03", seed_min=2000, max_new=3)
    assert len(cands) == 3
    assert cands[0]["username"] == "s9"   # highest rating first


def test_gather_order_puts_pool_before_watch():
    pool = {"5": {"uid": "5", "username": "member", "last_rank": 50, "in_pool": True, "last_probed": "2026-12-02"}}
    watch = [{"uid": "7", "username": "w", "rating": 2600.0}]
    cands = pool_mod.gather_candidates(pool, [], [], watch, [], "2026-12-03", seed_min=2000, max_new=0)
    assert [(c["uid"], c["source"]) for c in cands] == [("5", "pool"), ("7", "watch")]


def test_build_board_uses_leaders_then_probes_and_respects_top_n():
    results = [{"uid": "1", "username": "a", "rank": 1, "rating": 2900},   # overridden by leader row below
               {"uid": "2", "username": "b", "rank": 2, "rating": 2800},
               {"uid": "3", "username": "c", "rank": 150, "rating": 2400}, # beyond top_n: excluded
               {"uid": "4", "username": "d", "rank": None, "rating": None}]
    leaders = [{"rank": 1, "uid": "1", "username": "a", "rating": 2950}]
    rows = pool_mod.build_board(results, leaders, top_n=100)
    assert [(r["rank"], r["username"], r["rating"]) for r in rows] == [(1, "a", 2950), (2, "b", 2800)]


def test_parse_watch_keeps_highest_rating_per_player():
    games = [{"players": [{"playerId": 5, "username": "x", "rating": "2100"},
                          {"playerId": 6, "username": "y", "rating": "1900"}]},
             {"players": [{"playerId": 5, "username": "x", "rating": "2300"}]}]
    out = parse_watch(games)
    assert {r["uid"]: r["rating"] for r in out} == {"5": 2300.0, "6": 1900.0}


def test_board_rank_is_server_rank_not_rating_sort(tmp_path):
    # a lower-rated player with the better server rank is listed first
    truth = [row("u1", "low", 2100), row("u2", "high", 2400)]
    client = MockClient({D[0]: truth}, leader_n=2)
    run_day(tmp_path, client, D[0])
    today = load(tmp_path / "today.json")
    assert [p["username"] for p in today["players"]] == ["low", "high"]
