"""Bridge to the compiled rules (engine/src/rules.rs) and the check that they match Python.

The Python package `chaturaji/` stays the reference. `check` replays games in both engines
in lockstep and compares, at every ply: the board, turn, scores, eliminated seats, game-over
reason, repetition count, the legal moves (same order), the network's 16 global features,
and what each move did (points, capture, check bonus, elimination). At the end it unmakes
every move and compares with the start, which tests make/unmake too.

    python -m bot.rules_core check                       # every scraped game + 200 random games
    python -m bot.rules_core check --random 2000 --limit 0
    python -m bot.rules_core bench                       # speed: Python vs compiled

Moves in the compiled engine are integers from_square * 64 + to_square, square = row * 8 + col.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chaturaji.constants import TURN_ORDER
from chaturaji.game import apply_move, eliminate, game_over_reason, is_game_over
from chaturaji.moves import Move, legal_moves
from chaturaji.state import ChaturajiState, board_key

from .mlp import global_features
from .nnue import board_codes

try:
    import chaturaji_engine as _engine
except ImportError:
    _engine = None

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "scraper" / "data" / "records"
SEAT = {c: i for i, c in enumerate(TURN_ORDER)}


def require_engine():
    if _engine is None:
        raise SystemExit("chaturaji_engine is not installed: build it (engine/README.md)")
    return _engine


def position_from_state(state: ChaturajiState):
    """The compiled Position for a Python state (undo history starts empty)."""
    e = require_engine()
    return e.Position.from_parts(board_codes(state.board), state.turn,
                                 [state.scores[c] for c in TURN_ORDER],
                                 sorted(SEAT[c] for c in state.eliminated),
                                 list(state.position_counts.items()))


def move_code(move: Move) -> int:
    (sr, sc), (er, ec) = move.start, move.end
    return (sr * 8 + sc) * 64 + er * 8 + ec


def move_from_code(code: int, board) -> Move:
    s, e = divmod(int(code), 64)
    uci = f"{chr(97 + s % 8)}{8 - s // 8}{chr(97 + e % 8)}{8 - e // 8}"
    return Move.from_uci(uci, board)


def differences(state: ChaturajiState, pos) -> list[str]:
    """Everything that differs between a Python state and a compiled Position."""
    out = []
    if not np.array_equal(pos.board, board_codes(state.board)):
        out.append("board")
    if pos.turn != state.turn:
        out.append(f"turn {pos.turn} vs {state.turn}")
    if pos.scores != [state.scores[c] for c in TURN_ORDER]:
        out.append(f"scores {pos.scores} vs {[state.scores[c] for c in TURN_ORDER]}")
    if pos.eliminated != sorted(SEAT[c] for c in state.eliminated):
        out.append(f"eliminated {pos.eliminated} vs {sorted(state.eliminated)}")
    if pos.game_over != game_over_reason(state):
        out.append(f"game over {pos.game_over} vs {game_over_reason(state)}")
    if pos.repetitions != state.position_counts.get(board_key(state.board), 0):
        out.append("repetitions")
    if not is_game_over(state):
        if pos.current_player != SEAT[state.current_player()]:
            out.append("current player")
        if pos.legal_moves() != [move_code(m) for m in legal_moves(state)]:
            out.append("legal moves")
    for k in range(4):
        if not np.array_equal(pos.globals(k), global_features(state, k)):
            out.append(f"globals from seat {k}")
            break
    return out


def check_game(initial: ChaturajiState, entries: list[dict], name: str) -> list[str]:
    """Replay (uci or elimination) entries in both engines; returns problems found."""
    state, pos = initial.copy(), position_from_state(initial)
    start = board_codes(initial.board)
    problems = []
    for ply, h in enumerate(entries):
        diff = differences(state, pos)
        if diff:
            return [f"{name} ply {ply}: {', '.join(diff)}"]
        if h.get("uci") is None:
            state = eliminate(state, h["color"])
            pos.eliminate(SEAT[h["color"]])
            continue
        move = Move.from_uci(h["uci"], state.board)
        state, info = apply_move(state, move)
        points, captured, check_bonus, elim = pos.push(move_code(move))
        cap_code = 0 if info.captured is None else int(_piece_code(info.captured))
        got, want = (points, captured, check_bonus, elim), (
            info.points, cap_code, info.check_bonus, None if info.eliminated is None else SEAT[info.eliminated])
        if got != want:
            return [f"{name} ply {ply} ({h['uci']}): move info {got} vs {want}"]
    diff = differences(state, pos)
    if diff:
        return [f"{name} end: {', '.join(diff)}"]
    while pos.depth:
        pos.pop()
    if not np.array_equal(pos.board, start) or pos.turn != initial.turn or differences(initial, pos):
        problems.append(f"{name}: unmaking every move does not return to the start")
    return problems


_OWNER = {o: i for i, o in enumerate("RBYGX")}
_TYPE = {t: i for i, t in enumerate("KBNRP")}


def _piece_code(piece: str) -> int:
    return 1 + _OWNER[piece[0]] * 5 + _TYPE[piece[1]]


def scraped(records: Path, limit: int | None):
    paths = sorted(records.glob("*.json"))
    for p in paths[:limit] if limit else paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        yield p.stem, ChaturajiState.from_dict(d["initial"]), d["moves"]


def random_games(n: int, seed: int = 0, max_plies: int = 500):
    rng = np.random.default_rng(seed)
    for g in range(n):
        s = ChaturajiState.initial()
        entries = []
        while not is_game_over(s) and len(entries) < max_plies:
            if rng.random() < 0.004 and len(s.active_players()) > 1:   # a resignation now and then
                c = TURN_ORDER[int(rng.integers(4))]
                entries.append({"uci": None, "color": c})
                s = eliminate(s, c)
                continue
            ms = legal_moves(s)
            m = ms[int(rng.integers(len(ms)))]
            entries.append({"uci": m.uci})
            s, _ = apply_move(s, m)
        yield f"random-{g}", ChaturajiState.initial(), entries


def check(records: Path = RECORDS, limit: int | None = None, n_random: int = 200, seed: int = 0) -> dict:
    require_engine()
    t0 = time.time()
    games = plies = 0
    problems = []
    sources = [scraped(records, limit)] if records and Path(records).is_dir() else []
    sources.append(random_games(n_random, seed))
    for src in sources:
        for name, initial, entries in src:
            problems += check_game(initial, entries, name)
            games += 1
            plies += len(entries)
    return {"games": games, "plies": plies, "problems": len(problems), "first_problems": problems[:10],
            "seconds": round(time.time() - t0, 1)}


def bench(n_games: int = 20, seed: int = 1) -> dict:
    """Random playouts: legal moves + make move per ply, Python vs compiled."""
    e = require_engine()
    rng = np.random.default_rng(seed)
    t, plies = time.perf_counter(), 0
    for _ in range(n_games):
        s = ChaturajiState.initial()
        while not is_game_over(s) and s.turn < 400:
            ms = legal_moves(s)
            s, _ = apply_move(s, ms[int(rng.integers(len(ms)))])
            plies += 1
    py = (time.perf_counter() - t) / plies
    rng = np.random.default_rng(seed)
    t, plies2 = time.perf_counter(), 0
    for _ in range(n_games):
        p = e.Position.initial()
        while p.game_over is None and p.turn < 400:
            ms = p.legal_moves()
            p.push(ms[int(rng.integers(len(ms)))])
            plies2 += 1
    rs = (time.perf_counter() - t) / plies2
    return {"plies": plies, "python_us_per_ply": round(1e6 * py, 1), "compiled_us_per_ply": round(1e6 * rs, 2),
            "speedup": round(py / rs, 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--records", default=str(RECORDS))
    c.add_argument("--limit", type=int, default=0, help="scraped games to check (0 = all)")
    c.add_argument("--random", type=int, default=200, help="random games with resignations")
    c.add_argument("--seed", type=int, default=0)
    sub.add_parser("bench")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        res = check(Path(args.records), args.limit or None, args.random, args.seed)
        print(json.dumps(res, indent=1))
        if res["problems"]:
            raise SystemExit(1)
    else:
        print(json.dumps(bench()))


if __name__ == "__main__":
    main()
