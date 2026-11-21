import json

import pytest

from chaturaji import ChaturajiState, Game, Move, apply_move, game_over_reason, legal_moves, winners
from chaturaji.encoding import (
    NUM_ACTIONS, PLANES, action_to_move, encode_state, legal_action_mask, move_to_action,
)
from chaturaji.moves import is_attacked_by
from chaturaji.notation import move_table
from chaturaji.simulate import random_game
from chaturaji.state import board_key, square_to_index


def make_state(pieces: dict[str, str], turn: int = 0, eliminated=()) -> ChaturajiState:
    board = [[None] * 8 for _ in range(8)]
    for sq, p in pieces.items():
        r, c = square_to_index(sq)
        board[r][c] = p
    s = ChaturajiState(board=board, turn=turn, eliminated=set(eliminated))
    s.position_counts[board_key(board)] = 1
    return s


def dests(state, square):
    start = square_to_index(square)
    return {m.uci[2:] for m in legal_moves(state) if m.start == start}


def play(state, uci):
    return apply_move(state, Move.from_uci(uci, state.board))


# --- setup ---------------------------------------------------------------------------

def test_initial_position():
    s = ChaturajiState.initial()
    pieces = [p for row in s.board for p in row if p]
    assert len(pieces) == 32
    for color in "RBYG":
        own = [p for p in pieces if p[0] == color]
        assert sorted(p[1] for p in own) == sorted("KBNRPPPP")
    assert s.turn == 0 and s.current_player() == "R"
    assert s.board[7][0] == "RR" and s.board[7][3] == "RK"  # a1, d1: Red on bottom row
    assert square_to_index("a1") == (7, 0) and square_to_index("h8") == (0, 7)


# --- movement ------------------------------------------------------------------------

def test_piece_movement_on_open_board():
    s = make_state({"d4": "RR", "a1": "RK"})
    assert len(dests(s, "d4")) == 14
    s = make_state({"d4": "RB", "a8": "RK"})
    assert len(dests(s, "d4")) == 13
    s = make_state({"d4": "RN", "a8": "RK"})
    assert dests(s, "d4") == {"b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"}
    s = make_state({"d4": "RK"})
    assert len(dests(s, "d4")) == 8


def test_sliders_blocked_and_capture():
    s = make_state({"d4": "RR", "d6": "YP", "d2": "RP", "a1": "RK"})
    d = dests(s, "d4")
    assert "d6" in d and "d7" not in d and "d2" not in d and "d3" in d


@pytest.mark.parametrize("color,start,forward,caps", [
    ("R", "d4", "d5", {"c5", "e5"}),
    ("Y", "d4", "d3", {"c3", "e3"}),
    ("B", "d4", "e4", {"e3", "e5"}),
    ("G", "d4", "c4", {"c3", "c5"}),
])
def test_pawn_directions(color, start, forward, caps):
    turn = "RBYG".index(color)
    enemy = "R" if color != "R" else "Y"
    pieces = {start: color + "P"}
    pieces.update({sq: enemy + "N" for sq in caps})
    s = make_state(pieces, turn=turn)
    assert dests(s, start) == {forward} | caps
    # blocked forward, no double step
    s = make_state({start: color + "P", forward: enemy + "P"}, turn=turn)
    assert dests(s, start) == set()


@pytest.mark.parametrize("color,start,end", [
    ("R", "c7", "c8"), ("Y", "c2", "c1"), ("B", "g3", "h3"), ("G", "b3", "a3"),
])
def test_promotion_to_rook(color, start, end):
    s = make_state({start: color + "P"}, turn="RBYG".index(color))
    (m,) = legal_moves(s)
    assert m.promotion == "R"
    s2, _ = apply_move(s, m)
    r, c = square_to_index(end)
    assert s2.board[r][c] == color + "R"


def test_king_can_walk_into_attack():
    s = make_state({"d4": "RK", "e8": "YR"})
    assert "e4" in dests(s, "d4")
    assert is_attacked_by(s.board, square_to_index("e4"), "Y")


# --- scoring / elimination -----------------------------------------------------------

@pytest.mark.parametrize("ptype,points", [("P", 1), ("N", 3), ("B", 5), ("R", 5)])
def test_capture_points(ptype, points):
    s = make_state({"a1": "RR", "a5": "Y" + ptype, "h8": "YK", "h1": "RK"})
    s2, info = play(s, "a1a5")
    assert s2.scores["R"] == points and info.points == points


def test_king_capture_eliminates_and_kills_pieces():
    s = make_state({"a1": "RR", "a5": "BK", "c6": "BN", "b7": "BP", "h8": "YK", "h2": "GK",
                    "e1": "RK"})
    s2, info = play(s, "a1a5")
    assert info.eliminated == "B" and "B" in s2.eliminated
    assert s2.scores["R"] == 3
    assert s2.board[2][2] == "XN" and s2.board[1][1] == "XP"
    assert s2.current_player() == "Y" and s2.turn == 2  # Blue's turn skipped
    # dead pieces never move and are worth 0
    assert all(s2.board[m.start[0]][m.start[1]][0] != "X" for m in legal_moves(s2))


def test_dead_piece_capture_worth_zero():
    s = make_state({"c8": "YR", "c3": "XN", "h8": "YK", "e1": "RK", "h2": "GK"},
                   turn=2, eliminated={"B"})
    s2, info = play(s, "c8c3")
    assert info.points == 0 and s2.scores["Y"] == 0


def test_check_bonuses():
    # Red rook to d5 attacks Blue king d8 and Yellow king h5 (double check +1)
    s = make_state({"d1": "RR", "d8": "BK", "h5": "YK", "a1": "GK", "e1": "RK"})
    s2, info = play(s, "d1d5")
    assert info.checked == {"B", "Y"} and info.check_bonus == 1 and s2.scores["R"] == 1
    # Triple check: rook on d5 also hits Green king a5 (+5)
    s = make_state({"d1": "RR", "d8": "BK", "h5": "YK", "a5": "GK", "e1": "RK"})
    s2, info = play(s, "d1d5")
    assert info.checked == {"B", "Y", "G"} and info.check_bonus == 5
    # Single check: no bonus
    s = make_state({"d1": "RR", "d8": "BK", "h7": "YK", "a1": "GK", "e1": "RK"})
    _, info = play(s, "d1d5")
    assert info.checked == {"B"} and info.check_bonus == 0


def test_no_bonus_without_new_check():
    # Red already double-checks; a quiet move elsewhere gives no bonus.
    s = make_state({"d5": "RR", "d8": "BK", "h5": "YK", "a1": "GK", "e1": "RK"})
    _, info = play(s, "e1f1")
    assert len(info.checked) == 2 and info.check_bonus == 0


# --- game end ------------------------------------------------------------------------

def test_last_king_standing_and_points_win():
    s = make_state({"a1": "RR", "a5": "BK", "h8": "YK", "h1": "GK", "e1": "RK"},
                   eliminated=set())
    s.eliminated = {"Y", "G"}
    s.board[0][7] = None
    s.board[7][7] = None
    s.scores = {"R": 0, "B": 0, "Y": 20, "G": 0}
    s2, _ = play(s, "a1a5")
    assert game_over_reason(s2) == "last_king"
    assert winners(s2) == ["Y"]  # points decide, not survival


def test_threefold_repetition():
    s = make_state({"a1": "RK", "h8": "BK", "a8": "YK", "h1": "GK"})
    shuffle = [("a1a2", "h8h7", "a8b8", "h1g1"), ("a2a1", "h7h8", "b8a8", "g1h1")]
    for _ in range(2):
        for half in shuffle:
            for uci in half:
                s, _ = play(s, uci)
    assert game_over_reason(s) == "repetition"
    assert winners(s) == ["R", "B", "Y", "G"]


# --- notation / record ---------------------------------------------------------------

def test_san_and_table():
    g = Game()
    g.play_uci("c2c3")  # Red pawn
    g.play_uci("a7c8")  # Blue knight
    g.play_uci("e7e6")  # Yellow pawn
    g.play_uci("h4g5")  # Green king
    assert [h["san"] for h in g.history] == ["c3", "Nc8", "e6", "Kg5"]
    table = g.move_table().splitlines()
    assert table[0].split() == ["Red", "Blue", "Yellow", "Green"]
    assert table[1].split() == ["1.", "c3", "Nc8", "e6", "Kg5"]


def test_san_capture_disambiguation_and_elimination():
    s = make_state({"a1": "RR", "h4": "RR", "a4": "BN", "c8": "BK", "e8": "YK", "h1": "GK",
                    "e1": "RK"})
    g = Game(s)
    g.play_uci("h4a4")
    assert g.history[0]["san"] == "Rhxa4"
    s = make_state({"c2": "RP", "d3": "BK", "e8": "YK", "h1": "GK", "e1": "RK"})
    g = Game(s)
    g.play_uci("c2d3")
    assert g.history[0]["san"] == "cxd3#"
    s = make_state({"b6": "BP", "c7": "YN", "e8": "YK", "h1": "GK", "e1": "RK", "a5": "BK"},
                   turn=1)
    g = Game(s)
    g.play_uci("b6c7")
    assert g.history[0]["san"] == "6xc7"


def test_skipped_turns_in_table():
    hist = [{"turn": 0, "san": "a"}, {"turn": 2, "san": "b"}, {"turn": 3, "san": "c"},
            {"turn": 4, "san": "d"}]
    assert move_table(hist).splitlines()[1].split() == ["1.", "a", "--", "b", "c"]


def test_record_roundtrip(tmp_path):
    g = random_game(seed=3)
    path = tmp_path / "g.json"
    g.save(path)
    g2 = Game.load(path)
    assert g2.history == g.history
    assert g2.state.to_dict() == g.state.to_dict()
    assert len(g2.states()) == len(g.history) + 1
    json.loads(path.read_text())


# --- encoding / simulation -----------------------------------------------------------

def test_encoding():
    s = ChaturajiState.initial()
    x = encode_state(s)
    assert x.shape == (PLANES, 8, 8)
    assert x[:25].sum() == 32
    mask = legal_action_mask(s)
    assert mask.shape == (NUM_ACTIONS,) and mask.sum() == len(legal_moves(s))
    for m in legal_moves(s):
        assert action_to_move(move_to_action(m), s) == m


@pytest.mark.parametrize("seed", range(20))
def test_random_games_finish(seed):
    g = random_game(seed)
    assert g.result()["reason"] in ("last_king", "repetition")
    s = g.state
    assert sum(1 for row in s.board for p in row if p and p[1] == "K") == len(s.active_players())


# --- scoring rules observed on chess.com ----------------------------------------------

def test_dead_king_capture_worth_three():
    from chaturaji.game import eliminate
    s = make_state({"a1": "RR", "a5": "BK", "e8": "YK", "h1": "GK", "e1": "RK"}, turn=1)
    s = eliminate(s, "B")  # Blue resigns on its turn: king becomes XK
    assert s.board[3][0] == "XK" and s.current_player() == "Y"
    s.turn = 4  # Red to move
    s2, info = play(s, "a1a5")
    assert info.points == 3 and s2.scores["R"] == 3


def test_last_standing_collects_dead_kings():
    from chaturaji.game import eliminate
    s = make_state({"a1": "RR", "a5": "BK", "e8": "YK", "h1": "GK", "e1": "RK"})
    s = eliminate(s, "B")
    s = eliminate(s, "Y")
    assert s.scores["R"] == 0 and s.scores["G"] == 0
    s = eliminate(s, "G")  # Red is last: +3 for each of the 3 dead kings
    assert game_over_reason(s) == "last_king"
    assert s.scores["R"] == 9
    assert eliminate(s, "R").scores == s.scores  # nothing happens after the game ends


def test_repetition_gives_remaining_players_two():
    s = make_state({"a1": "RK", "h8": "YK"}, eliminated={"B", "G"})
    for _ in range(2):
        for uci in ("a1a2", "h8h7", "a2a1", "h7h8"):
            s, info = play(s, uci)
    assert game_over_reason(s) == "repetition"
    assert s.scores == {"R": 2, "B": 0, "Y": 2, "G": 0}  # both remaining players +2
