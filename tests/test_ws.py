"""WebSocket scraper: message -> saved game conversion (no network)."""

import json
import re

import pytest

from scraper import game_scraping as gs
from scraper import ws
from tests.test_game_scraping import REAL_HEADERS
from tests.test_pgn4 import REAL_GAME_113036725

T0 = 1787801197000  # ms


def with_comments(movetext: str, start_ms: int = T0, step_ms: int = 2000) -> str:
    """Add the server's { date= clock= } comment after every move, like archive_game does."""
    n = 0

    def add(m):
        nonlocal n
        tok = m.group(0)
        if re.fullmatch(r"\d+\.|\.\.", tok) or tok.startswith("["):
            return tok
        stamp = ws.iso_ms(start_ms + n * step_ms)
        n += 1
        return f"{tok} {{ date={stamp} clock={60000 - n} }}"
    head, _, body = movetext.partition("\n")
    return head + "\n" + re.sub(r"\S+", add, body)


def archive_game(chat=None):
    pgn = REAL_HEADERS.replace('[RuleVariants', '[StartFen4 "R-0\nx,x/\nx,x"]\n[RuleVariants') \
        + with_comments(REAL_GAME_113036725)
    return {
        "gameNr": 113036725, "date": "2026-11-20T23:19:00.000Z", "rated": True,
        "timeControlInitial": 1, "timeControlIncrement": 5, "q": {"timeControl": "1 | 5",
                                                                    "title": "Chaturaji", "variantUrl": "chaturaji", "typeName": "FFA"},
        "standings": [2, 1, 4, 3], "duration": 9, "plies": 46,
        "username1": "OragamiMaster22300", "username2": "griffinlc",
        "username3": "chikentenders1", "username4": "justinberner",
        "uid1": 1, "uid2": 2, "uid3": 3, "uid4": 4,
        "rating1": 1624, "rating2": 1773, "rating3": 1499, "rating4": 1500,
        "ratingDiff1": 3.5, "ratingDiff2": 20.1, "ratingDiff3": -19, "ratingDiff4": -4.6,
        "points1": 13, "points2": 25, "points3": 0, "points4": 10,
        "ip1": "aaaa", "ip2": "bbbb", "ip3": "cccc", "ip4": "dddd",
        "pgn4": pgn,
        "chat": chat if chat is not None else [
            {"time": T0 - 5000, "playerId": 0, "username": "", "message": "Game started.", "type": "info"},
            {"time": T0 + 3000, "playerId": 2, "username": "griffinlc", "message": "hi", "type": "",
             "seat": 1, "fullMoveNr": 1},
        ],
    }


def test_page_user_reads_embedded_context():
    html = '<script>\n context = {"userId":"userId","user":{"id":5,"username":"me","isGuest":false}};\n</script>'
    assert ws.page_user(html) == {"id": 5, "username": "me", "isGuest": False}
    assert ws.page_user("<html>no context</html>") is None


def test_parse_moves_times_and_clocks():
    moves = ws.parse_moves('[GameNr "1"]\n1. e5-e6 { date=2026-08-27T03:26:37.554Z clock=61105 } '
                           ' .. e10-f10 { date=2026-08-27T03:26:39.187Z clock=63367 } \n2. R .. T')
    assert [m["move"] for m in moves] == ["e5-e6", "e10-f10", "R", "T"]
    assert moves[1]["seconds_since_prev"] == 1.633
    assert moves[0]["clock_ms"] == 61105 and moves[2]["clock_ms"] is None
    assert moves[3]["seconds_since_prev"] is None


def test_clean_pgn4_joins_split_header_values():
    assert ws.clean_pgn4('[StartFen4 "a\nb\nc"]\n1. e5-e6 { date=x }') == '[StartFen4 "abc"]\n1. e5-e6 { date=x }'


def test_time_control_labels():
    assert ws.time_control({"q": {"timeControl": "1 | 5"}}) == "1 | 5"
    assert ws.time_control({"timeControlInitial": 0.1, "timeControlIncrement": 3}) == "6s | 3"
    assert ws.time_control({"timeControlInitial": 0.5, "timeControlIncrement": 0}) == "30 sec"


def test_seat_players_places_from_standings():
    players = ws.seat_players(archive_game())
    assert [(p["seat"], p["name"], p["place"]) for p in players] == [
        ("Red", "OragamiMaster22300", 2), ("Blue", "griffinlc", 1),
        ("Yellow", "chikentenders1", 4), ("Green", "justinberner", 3)]
    assert players[1]["rating_change"] == "+20.1"
    assert all("ip" not in p for p in players)


def test_game_data_places_chat_by_timestamp():
    data = ws.game_data(archive_game())
    assert data["game_id"] == "113036725" and data["time_control"] == "1 | 5"
    assert "\n" not in data["headers"]["StartFen4"]
    assert len(data["moves"]) == 46 and data["moves"][0]["time"] == ws.iso_ms(T0)
    start, hi = data["chat"]
    assert start["system"] and (start["after_ply"], start["before_ply"]) == (None, 0)
    # sent at T0+3s: after ply 1 (T0+2s), before ply 2 (T0+4s)
    assert (hi["user"], hi["seat"], hi["after_ply"], hi["before_ply"]) == ("griffinlc", 1, 1, 2)
    assert "ip1" not in json.dumps(data)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "DATA", tmp_path)
    return tmp_path


def test_ws_game_stores_like_selenium(data_dir):
    raw = archive_game()
    row = ws.archive_row(raw, searched="griffinlc")
    assert row["players"][1]["searched"] and ws.is_chaturaji(raw)
    assert gs.store_game(ws.game_data(raw), row, player="griffinlc") == "ok"
    rec = json.loads((data_dir / "records" / "113036725.json").read_text())
    assert rec["result"]["scores"] == {"R": 13, "B": 25, "Y": 0, "G": 10}
    e = gs.read_index("griffinlc", "1 | 5")[0]
    assert (e["seat"], e["points"], e["place"], e["rating_change"], e["chat_messages"]) == ("Blue", 25, 1, "+20.1", 1)
    assert ws.known_game_ids("griffinlc") == {"113036725"}


class FakeSocket:
    """Stands in for VariantsSocket: one archive page and its games."""

    def __init__(self, rows):
        self.rows, self.fetched = rows, []

    def find_player(self, name):
        return 2

    def iter_archive(self, uid, days, max_pages=None):
        yield from self.rows

    def game(self, gid):
        self.fetched.append(str(gid))
        return {**archive_game(), "gameNr": int(gid)}


def test_scrape_player_filters_and_skips_known(data_dir):
    base = archive_game()
    rows = [{**base, "gameNr": 1},
            {**base, "gameNr": 2, "q": {**base["q"], "timeControl": "30 sec"}},
            {**base, "gameNr": 3, "q": {**base["q"], "variantUrl": "fogofwar", "title": "Fog of War"}},
            {**base, "gameNr": 4}, {**base, "gameNr": 5}]
    sock = FakeSocket(rows)
    assert ws.scrape_player(sock, "griffinlc", ["1 | 5"], max_per_tc=2, delay=0) == {"1+5": 2}
    assert sock.fetched == ["1", "4"]
    # a second run skips games already indexed
    sock2 = FakeSocket(rows)
    assert ws.scrape_player(sock2, "griffinlc", ["1 | 5"], max_per_tc=5, delay=0) == {"1+5": 1}
    assert sock2.fetched == ["5"]
    # another player in the same game is indexed from the saved copy, without a download
    sock3 = FakeSocket(rows[:1])
    assert ws.scrape_player(sock3, "justinberner", None, max_per_tc=5, delay=0) == {"1+5": 1}
    assert sock3.fetched == [] and gs.read_index("justinberner", "1 | 5")[0]["place"] == 3


def test_scrape_player_min_seek(data_dir):
    base = archive_game()
    rows = [{**base, "gameNr": 1, "q": {**base["q"], "minRating": 2200, "maxRating": 4000}},
            {**base, "gameNr": 2, "q": {**base["q"], "minRating": "1500"}},
            {**base, "gameNr": 3},                                   # no minimum
            {**base, "gameNr": 4, "q": {**base["q"], "minRating": "2000"}}]
    assert ws.seek_range(rows[0]) == (2200, 4000) and ws.seek_range(rows[2]) == (None, None)
    sock = FakeSocket(rows)
    assert ws.scrape_player(sock, "griffinlc", None, max_per_tc=10, delay=0, min_seek=2000) == {"1+5": 2}
    assert sock.fetched == ["1", "4"]
    assert [e["seek_min"] for e in gs.read_index("griffinlc", "1 | 5")] == [2200, 2000]


def test_scrape_player_rapid_and_opponent_rating(data_dir):
    base = archive_game()   # griffinlc (1773) vs 1624 / 1499 / 1500
    strong = {"rating1": 2100, "rating3": 1950, "rating4": 1900}
    rows = [{**base, "gameNr": 1, **strong, "q": {**base["q"], "timeControlType": "rapid"}},
            {**base, "gameNr": 2, **strong, "q": {**base["q"], "timeControlType": "blitz"}},
            {**base, "gameNr": 3, "q": {**base["q"], "timeControlType": "rapid"}},   # weak opponents
            {**base, "gameNr": 4, **strong, "rating4": 1899, "q": {**base["q"], "timeControlType": "rapid"}}]
    row = ws.archive_row(rows[0], searched="griffinlc")
    assert row["tc_type"] == "rapid" and ws.min_opponent_rating(row, "griffinlc") == 1900
    # game 2 was indexed earlier, before tc_type was recorded
    gs.store_game(ws.game_data({**rows[1], "gameNr": 2}), {**ws.archive_row(rows[1]), "tc_type": None},
                  player="griffinlc")
    sock = FakeSocket(rows)
    assert ws.scrape_player(sock, "griffinlc", None, max_per_tc=10, delay=0,
                            tc_types=["Rapid"], min_opp_rating=1900) == {"1+5": 1}
    assert sock.fetched == ["1"]
    by_id = {e["game_id"]: e for e in gs.read_index("griffinlc", "1 | 5")}
    assert by_id["1"]["tc_type"] == "rapid" and by_id["2"]["tc_type"] == "blitz"   # back-filled


def test_scrape_player_skips_games_without_the_player(data_dir):
    base = archive_game()
    other = {**base, "gameNr": 9, "username2": "someoneelse"}
    sock = FakeSocket([other, {**base, "gameNr": 1}])
    assert ws.scrape_player(sock, "griffinlc", None, max_per_tc=10, delay=0) == {"1+5": 1}
    assert sock.fetched == ["1"]
