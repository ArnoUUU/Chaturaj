"""Stats and daily-run tests over a five-day scenario, through the mock client.

The board here is the top 4 (update.run takes top_n), so the schedule stays readable.
Days: D1..D5 = 2026-11-01..05.

  D1: A 2000 (#1), B 1900, X 1800, C 1700
  D2: X 1850 (#1), A 2010, B 1900, C 1690
  D3: A 2020 (#1), B 1905, C 1700, D 1650      X drops off the board
  D4: A 2020 (#1), B 1880, C 1700, D 1660      X still off
  D5: A 2030 (#1), B 1880, X 1700, D 1660      X comes back at #3; C drops off
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from helpers import DATES, UIDS, MockClient, five_day_truth, load  # noqa: E402
from leaderboards import stats, update  # noqa: E402

D1, D2, D3, D4, D5 = DATES
TOP = 4


def play(tmp_path: Path, dates=tuple(DATES), truth=None, **kw) -> list[dict]:
    truth = truth or five_day_truth()
    client = MockClient(truth, leader_n=1)
    out = []
    for d in dates:
        client.day = d
        out.append(update.run(tmp_path, client, date=d, top_n=TOP, seed_min=0, **kw))
    return out


@pytest.fixture
def five_days(tmp_path):
    return tmp_path, play(tmp_path)


# --- the five-day run and files ----------------------------------------------------------

def test_five_day_run_writes_all_files(five_days):
    tmp, summaries = five_days
    assert len(list((tmp / "daily").glob("*.json"))) == 5
    assert load(tmp / "index.json") == {"dates": DATES, "latest": D5}
    today = load(tmp / "today.json")
    assert today["date"] == D5 and today["count"] == 4 and today["coverage"] == "pool"
    assert [p["username"] for p in today["players"]] == ["A", "B", "X", "D"]
    assert summaries[0]["prev_date"] is None and summaries[4]["prev_date"] == D4


def test_snapshot_schema(five_days):
    tmp, _ = five_days
    snap = load(tmp / "daily" / f"{D1}.json")
    assert set(snap) >= {"date", "fetched_at", "board", "top_n", "coverage", "count", "players"}
    assert set(snap["players"][0]) == {"rank", "uid", "username", "rating"}


# --- drop-out, off-board days, re-entry (the tracking-window rule) -----------------------

def test_dropout_pauses_tracking(tmp_path):
    play(tmp_path, dates=(D1, D2, D3, D4))
    x = load(tmp_path / "players.json")[UIDS["X"]]
    assert x["currently_on_board"] is False
    assert x["left_date"] == D3
    assert x["left_rating"] == 1850
    assert x["rank"] is None and x["rating"] is None
    assert [h[0] for h in x["history"]] == [D1, D2]


def test_off_board_days_accrue_nothing(tmp_path):
    play(tmp_path, dates=(D1, D2, D3, D4))
    x = load(tmp_path / "players.json")[UIDS["X"]]
    assert x["days_on_board"] == 2
    assert x["times_entered"] == 0
    assert x["longest_streak_days"] == 2
    assert x["current_streak_days"] == 0
    assert x["days_in_top10"] == 2


def test_reentry_resumes_same_history(five_days):
    tmp, _ = five_days
    x = load(tmp / "players.json")[UIDS["X"]]
    assert [h[0] for h in x["history"]] == [D1, D2, D5]
    assert x["times_entered"] == 1
    assert x["current_streak_days"] == 1
    assert x["longest_streak_days"] == 2
    assert x["currently_on_board"] is True and x["left_date"] is None
    assert x["rank"] == 3 and x["rating"] == 1700


def test_reentry_first_day_has_no_1d_change(five_days):
    tmp, _ = five_days
    x = load(tmp / "players.json")[UIDS["X"]]
    assert x["change_1d"] is None
    assert x["rank_change_1d"] is None
    assert x["change_7d"] is None   # only 5 days of data


def test_peak_persists_across_dropout(five_days):
    tmp, _ = five_days
    x = load(tmp / "players.json")[UIDS["X"]]
    assert x["peak_rating"] == 1850 and x["peak_rating_date"] == D2
    assert x["peak_rank"] == 1 and x["peak_rank_date"] == D2
    assert x["first_seen"] == D1


def test_dropout_detected_and_listed(tmp_path):
    summaries = play(tmp_path, dates=(D1, D2, D3))
    assert summaries[2]["left"] == ["X"]
    assert summaries[2]["entered"] == ["D"]
    assert summaries[1]["left"] == [] and summaries[1]["entered"] == []


def test_reentry_listed_as_entered(five_days):
    _, summaries = five_days
    assert summaries[4]["entered"] == ["X"]
    assert summaries[4]["left"] == ["C"]   # C was on D4 and is not on D5


# --- peaks, streaks, reigns, records ----------------------------------------------------

def test_peak_rating_updates(five_days):
    tmp, _ = five_days
    a = load(tmp / "players.json")[UIDS["A"]]
    assert a["peak_rating"] == 2030 and a["peak_rating_date"] == D5
    assert a["peak_rank"] == 1 and a["peak_rank_date"] == D1


def test_streaks_for_continuous_player(five_days):
    tmp, _ = five_days
    a = load(tmp / "players.json")[UIDS["A"]]
    assert a["days_on_board"] == 5
    assert a["times_entered"] == 0
    assert a["longest_streak_days"] == 5 and a["current_streak_days"] == 5


def test_number_one_reign_and_days_at_top(five_days):
    tmp, _ = five_days
    records = load(tmp / "records.json")
    assert records["longest_reign"] == {"uid": UIDS["A"], "username": "A", "start": D3, "end": D5, "days": 3}
    assert records["number_one"]["username"] == "A" and records["number_one"]["days"] == 3
    assert records["most_days_at_number_one"] == {"uid": UIDS["A"], "username": "A", "days": 4}
    assert records["most_days_in_top10"]["days"] == 5


def test_all_time_high(five_days):
    tmp, _ = five_days
    assert load(tmp / "records.json")["all_time_high"] == {
        "uid": UIDS["A"], "username": "A", "rating": 2030, "date": D5}


def test_biggest_moves_1d(five_days):
    tmp, _ = five_days
    records = load(tmp / "records.json")
    assert records["biggest_gain_1d"] == {"uid": UIDS["X"], "username": "X", "rating": 50, "date": D2}
    assert records["biggest_drop_1d"] == {"uid": UIDS["B"], "username": "B", "rating": -25, "date": D4}
    x = load(tmp / "players.json")[UIDS["X"]]
    assert x["biggest_gain_1d"] == {"rating": 50, "date": D2}
    assert x["biggest_drop_1d"] is None


def test_avg_series_has_one_row_per_day(five_days):
    tmp, _ = five_days
    series = load(tmp / "records.json")["avg_series"]
    assert [row[0] for row in series] == DATES
    assert series[0][1] == 2000
    assert series[4][1] == 2030


def test_movers_today(five_days):
    tmp, _ = five_days
    movers = load(tmp / "records.json")["movers"]
    assert movers["up"] == [{"uid": UIDS["A"], "username": "A", "rank": 1, "rating": 2030, "change": 10}]
    assert movers["down"] == []


def test_change_1d_is_on_board_day_to_day(five_days):
    tmp, _ = five_days
    a = load(tmp / "players.json")[UIDS["A"]]
    assert a["change_1d"] == 10 and a["rank_change_1d"] == 0
    b = load(tmp / "players.json")[UIDS["B"]]
    assert b["change_1d"] == 0 and b["rank_change_1d"] == 0


# --- idempotency and file handling -------------------------------------------------------

def test_reapply_same_day_is_identical(five_days):
    tmp, _ = five_days
    before = (tmp / "players.json").read_text(encoding="utf-8")
    records_before = (tmp / "records.json").read_text(encoding="utf-8")
    client = MockClient(five_day_truth(), leader_n=1)
    client.day = D5
    update.run(tmp, client, date=D5, top_n=TOP, seed_min=0)
    assert (tmp / "players.json").read_text(encoding="utf-8") == before
    assert (tmp / "records.json").read_text(encoding="utf-8") == records_before
    assert load(tmp / "index.json")["dates"] == DATES


def test_apply_snapshot_is_idempotent_in_memory():
    truth = five_day_truth()
    players: dict = {}
    prev = None
    for d in DATES:
        rows = [{"rank": i, "uid": u, "username": n, "rating": r}
                for i, (u, n, r) in enumerate(truth[d], 1)]
        players, _ = stats.apply_snapshot(players, {"players": rows}, d, prev)
        prev = d
    rows5 = [{"rank": i, "uid": u, "username": n, "rating": r}
             for i, (u, n, r) in enumerate(truth[D5], 1)]
    once, _ = stats.apply_snapshot(json.loads(json.dumps(players)), {"players": rows5}, D5, D4)
    twice, _ = stats.apply_snapshot(json.loads(json.dumps(once)), {"players": rows5}, D5, D4)
    assert once == twice


def test_existing_pool_snapshot_kept_without_force(tmp_path):
    play(tmp_path, dates=(D1,))
    shifted = five_day_truth()
    shifted[D1] = [(u, n, r + 1) for (u, n, r) in shifted[D1]]
    client = MockClient(shifted, leader_n=1)
    client.day = D1
    summary = update.run(tmp_path, client, date=D1, top_n=TOP, seed_min=0)
    assert summary["status"] == "kept existing snapshot"
    assert load(tmp_path / "daily" / f"{D1}.json")["players"][0]["rating"] == 2000


def test_force_overwrites_snapshot(tmp_path):
    play(tmp_path, dates=(D1,))
    shifted = five_day_truth()
    shifted[D1] = [(u, n, r + 1) for (u, n, r) in shifted[D1]]
    client = MockClient(shifted, leader_n=1)
    client.day = D1
    update.run(tmp_path, client, date=D1, top_n=TOP, seed_min=0, force=True)
    assert load(tmp_path / "daily" / f"{D1}.json")["players"][0]["rating"] == 2001


def test_leaderboard_with_no_rows_fails_and_writes_nothing(tmp_path):
    client = MockClient(five_day_truth(), leader_n=0)
    client.day = D1
    with pytest.raises(update.DataError):
        update.run(tmp_path, client, date=D1, top_n=TOP, seed_min=0)
    assert list(tmp_path.iterdir()) == []


def test_bad_leader_rank_rejected():
    rows = [{"rank": 1, "uid": "1", "username": "a", "rating": 1000},
            {"rank": 5, "uid": "2", "username": "b", "rating": 990}]
    with pytest.raises(update.DataError):
        update.validate_leaders(rows)
