"""Parse chess.com 4-player PGN ("PGN4") movetext into validated `chaturaji` Game records.

Expected shape (chess.com 4-player export; verify against a real downloaded game):

    [Variant "Chaturaji"]
    [Red "alice"] [Blue "bob"] [Yellow "carol"] [Green "dave"]
    1. d2-d3 .. b7-c7 .. e7-e6 .. g4-f4
    2. Nb1-c3 .. Ba6xc4+ .. R .. h4-g5 ...

Each numbered move holds up to four slots separated by "..". A slot is a long-algebraic
move (optional piece letter, from-square, '-' or 'x', optional captured-piece letter,
to-square, optional '=R', '+', '#'), or an elimination token: R (resign), T (timeout),
S (stalemate), "#" alone (checkmated). Comments in {} are ignored.

The parser does not rely on slot positions: the engine knows whose turn it is (it skips
eliminated players), so slots are applied in order. Every move is checked for legality;
a mismatch raises PGN4Error, which also catches a wrong assumed starting layout.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from chaturaji.moves import Move, legal_moves
from chaturaji.notation import COLOR_NAMES
from chaturaji.record import Game
from chaturaji.state import index_to_square, square_to_index

HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\b\d+\.(?!\d)")
SQUARE_RE = re.compile(r"([a-n])(1[0-4]|[1-9])")
ELIMINATION_TOKENS = {"R": "resign", "T": "timeout", "S": "stalemate", "#": "checkmate"}


class PGN4Error(ValueError):
    pass


def parse_headers(text: str) -> dict[str, str]:
    return {k: v for k, v in HEADER_RE.findall(text)}


def movetext(text: str) -> str:
    body = HEADER_RE.sub(" ", text)
    return COMMENT_RE.sub(" ", body)


def tokens(text: str) -> list[str]:
    """Flat list of slot tokens in play order."""
    out: list[str] = []
    for chunk in MOVE_NUM_RE.split(movetext(text)):
        for slot in chunk.split(".."):
            slot = slot.strip()
            if not slot or slot in ("*", "1-0", "0-1"):
                continue
            # a slot normally holds one token; tolerate extra whitespace-separated tokens
            out.extend(slot.split())
    return out


def _squares(token: str, offset: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for f, r in SQUARE_RE.findall(token):
        col = ord(f) - ord("a") - offset[0]
        rank = int(r) - offset[1]
        if not (0 <= col < 8 and 1 <= rank <= 8):
            raise PGN4Error(f"square {f}{r} is outside the 8x8 board (offset={offset})")
        out.append(square_to_index(f"{chr(ord('a') + col)}{rank}"))
    return out


def to_game(text: str, offset: tuple[int, int] = (0, 0), metadata: dict | None = None) -> Game:
    """Replay PGN4 text through the engine. `offset` = (files, ranks) to subtract if the
    export uses coordinates of a larger board."""
    headers = parse_headers(text)
    game = Game(metadata={"source": "chess.com", "headers": headers, **(metadata or {})})
    for i, tok in enumerate(tokens(text)):
        color = game.state.current_player()
        bare = tok.rstrip("+#")
        if tok in ELIMINATION_TOKENS or bare in ELIMINATION_TOKENS:
            key = tok if tok in ELIMINATION_TOKENS else bare
            game.eliminate(color, ELIMINATION_TOKENS[key])
            continue
        sq = _squares(tok, offset)
        if len(sq) < 2:
            raise PGN4Error(f"token {i} {tok!r}: cannot find from/to squares")
        start, end = sq[0], sq[-1]
        legal = {(m.start, m.end): m for m in legal_moves(game.state)}
        move = legal.get((start, end))
        if move is None:
            piece = game.state.board[start[0]][start[1]]
            raise PGN4Error(
                f"token {i} {tok!r}: illegal for {COLOR_NAMES[color]} at turn "
                f"{game.state.turn} (piece on {index_to_square(*start)} is {piece})")
        game.play(move)
        if game.result()["reason"] is not None:
            break
    return game


def from_game(game: Game) -> str:
    """Export a Game as PGN4 text (long algebraic), mainly for round-trip tests."""
    lines = ['[Variant "Chaturaji"]']
    for k, v in game.metadata.get("headers", {}).items():
        if k != "Variant":
            lines.append(f'[{k} "{v}"]')
    slots: dict[int, str] = {}
    states = game.states()
    for h, before in zip(game.history, states):
        if h.get("uci") is None:
            text = h["san"]
        else:
            mv = Move.from_uci(h["uci"], before.board)
            piece = before.board[mv.start[0]][mv.start[1]]
            capture = before.board[mv.end[0]][mv.end[1]] is not None
            text = ("" if piece[1] == "P" else piece[1]) + h["uci"][:2] + ("x" if capture else "-") \
                + h["uci"][2:] + ("=R" if mv.promotion else "")
        slots[h["turn"]] = (slots[h["turn"]] + " " + text) if h["turn"] in slots else text
    if slots:
        rounds = []
        for rnd in range(max(slots) // 4 + 1):
            cells = [slots.get(rnd * 4 + i, "") for i in range(4)]
            rounds.append(f"{rnd + 1}. " + " .. ".join(c for c in cells if c))
        lines.append("\n".join(rounds))
    return "\n".join(lines) + "\n"


# --- finding PGN4 inside captured network data -------------------------------------------

def _looks_like_pgn4(s: str) -> bool:
    return ("1." in s and ".." in s and SQUARE_RE.search(s) is not None
            and ("[" in s or "-" in s))


def find_pgn_candidates(obj: Any) -> Iterable[str]:
    """Yield strings in nested JSON (or raw text) that look like PGN4 movetext."""
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                yield from find_pgn_candidates(json.loads(s))
                return
            except ValueError:
                pass
        if _looks_like_pgn4(s):
            yield s
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from find_pgn_candidates(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_pgn_candidates(v)
