import json

import pytest

from chaturaji import Game
from chaturaji.simulate import random_game
from scraper.pgn4 import PGN4Error, find_pgn_candidates, from_game, parse_headers, to_game, tokens

SAMPLE = """[Variant "Chaturaji"]
[Red "alice"]
[Blue "bob"]
[Yellow "carol"]
[Green "dave"]

1. d2-d3 .. Na7-c8 .. e7-e6 .. Kh4-g5 {a comment}
2. Kd1-e1 .. Ka5-b4 .. Ng8-e7 .. Nh2-f3+
"""


def test_parse_sample():
    assert parse_headers(SAMPLE)["Red"] == "alice"
    assert tokens(SAMPLE)[:4] == ["d2-d3", "Na7-c8", "e7-e6", "Kh4-g5"]
    g = to_game(SAMPLE)
    assert [h["san"] for h in g.history] == ["d3", "Nc8", "e6", "Kg5", "Ke1+", "Kb4", "Ne7", "Nf3+"]
    # Ke1+: with d2 empty, the c1 bishop sees Green's king on g5
    assert g.metadata["headers"]["Blue"] == "bob"


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random_games(seed):
    g = random_game(seed)
    g2 = to_game(from_game(g))
    assert [h["uci"] for h in g2.history] == [h["uci"] for h in g.history]
    assert g2.state.to_dict() == g.state.to_dict()


def test_resignation_token():
    text = "1. d2-d3 .. R .. e7-e6 .. h4-g5\n2. d3-d4 .. e6-e5"
    g = to_game(text)
    assert "B" in g.state.eliminated
    assert g.state.board[3][0] == "XK"  # Blue's king (a5) is dead, not captured
    assert [h["san"] for h in g.history] == ["d3", "R", "e6", "Kg5", "d4+", "e5"]
    assert g.move_table().splitlines()[2].split() == ["2.", "d4+", "--", "e5"]
    # survives save/load
    assert Game.from_dict(json.loads(json.dumps(g.to_dict()))).state.to_dict() == g.state.to_dict()


def test_illegal_move_reports_context():
    with pytest.raises(PGN4Error, match="illegal for Red"):
        to_game("1. d2-d5")


def test_offset_for_larger_board_coordinates():
    # same opening as SAMPLE but shifted 3 files / 3 ranks, as on a 14x14 board
    shifted = "1. g5-g6 .. Nd10-f11 .. h10-h9 .. Kk7-j8"
    g = to_game(shifted, offset=(3, 3))
    assert [h["san"] for h in g.history] == ["d3", "Nc8", "e6", "Kg5"]
    with pytest.raises(PGN4Error):  # without the offset, g5-g6 is not a Red move
        to_game(shifted)
    with pytest.raises(PGN4Error, match="outside"):
        to_game("1. k7-k8")


def test_find_candidates_in_nested_json():
    payload = json.dumps({"game": {"id": 1, "meta": {"pgn": SAMPLE}}, "other": "1.5 stars"})
    assert list(find_pgn_candidates(payload)) == [SAMPLE.strip()]


REAL_GAME_113036725 = """[Result "OragamiMaster22300: 13 - griffinlc: 25 - chikentenders1: 0 - justinberner: 10"]
1. e5-e6 .. e8-f8 .. h10-h9 .. j7-i7
2. g5-g6 .. Bd9-e8+ .. Kh11-g11 .. i7-h7
3. g6xh7 .. e10-f10+ .. Kg11-h11 .. j6-i6
4. Bf4-g5+ .. Kd8-d9 .. Bi11-h10+ .. i6xh7
5. e6-e7 .. Be8xKh11# .. Nk5-i6
6. e7xf8 .. e9xf8 .. Ni6xBg5
7. Ne4xNg5 .. Nd10-e8 .. Bk6xh9
8. Ng5xh7 .. Kd9-e9 .. j5-i5
9. Kg4-h5 .. Bh11-f9 .. Bh9xd5
10. Nh7xi5 .. Bf9xRk4 .. Bd5-f7+
11. Rd4-d8 .. Rd11xRd8 .. Bf7xNe8+
12. Ni5xBk4 .. Rd8xBe8 .. j4-i4+
13. Kh5xi4 .. Re8-e7+ .. T
14. f5-f6 .. Re7xKk7
15. R
"""


def test_real_chesscom_game_scores_match():
    """chess.com game 113036725 (14x14 coordinates): final scores 13 / 25 / 0 / 10."""
    g = to_game(REAL_GAME_113036725, offset=(3, 3))
    assert g.state.scores == {"R": 13, "B": 25, "Y": 0, "G": 10}
    assert g.result()["reason"] == "last_king"
