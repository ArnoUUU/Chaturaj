"""Live data for the daily job: one variants-socket session, read-only.

  get.leaderboard <board>            -> the server's top 20 (ranks 1-20)
  subscribe-watch <filter> / unsubscribe-watch   -> live games with player ids and ratings
  player-popup {playerId, username}  -> one player's rank and rating (the profile card)

Nothing here seeks, joins or starts a game. Saved-game players are read from the local
scraper folder, which is gitignored, so they exist on the local machine only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import BOARD, LEADER_ROWS, PROBE_DELAY, WATCH_FILTER

SCRAPER_GAMES = Path(__file__).resolve().parent.parent / "scraper" / "data" / "games"
WATCH_WAIT = 10.0


class FetchError(RuntimeError):
    pass


def parse_leaders(reply: dict, n: int = LEADER_ROWS) -> list[dict]:
    """A `leaderboard` reply -> [{rank, uid, username, rating}] for the first n rows."""
    leaders = reply.get("leaders") or []
    rows = []
    for rank, p in enumerate(leaders[:n], 1):
        try:
            rows.append({"rank": rank, "uid": str(int(p["uid"])), "username": str(p["username"]),
                         "rating": int(p["rating"])})
        except (KeyError, TypeError, ValueError) as e:
            raise FetchError(f"leader row {rank} is malformed: {p!r} ({e})") from e
    return rows


def parse_watch(games: list) -> list[dict]:
    """A `watch_games` reply -> unique [{uid, username, rating}] over all listed players."""
    best: dict[str, dict] = {}
    for g in games or []:
        for p in g.get("players") or []:
            uid = p.get("playerId")
            if not uid or not p.get("username"):
                continue
            try:
                rating = float(p.get("rating"))
            except (TypeError, ValueError):
                rating = 0.0
            key = str(uid)
            if key not in best or rating > best[key]["rating"]:
                best[key] = {"uid": key, "username": str(p["username"]), "rating": rating}
    return list(best.values())


def scraper_players(path: Path = SCRAPER_GAMES) -> list[dict]:
    """Unique [{uid, username, rating}] from saved chaturaji games (local only; [] if absent)."""
    if not path.is_dir():
        return []
    best: dict[str, dict] = {}
    for f in path.glob("*.json"):
        try:
            game = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for p in game.get("players") or []:
            uid = p.get("uid")
            if not uid or not p.get("name"):
                continue
            try:
                rating = float(p.get("rating") or 0)
            except (TypeError, ValueError):
                rating = 0.0
            key = str(uid)
            if key not in best or rating > best[key]["rating"]:
                best[key] = {"uid": key, "username": str(p["name"]), "rating": rating}
    return list(best.values())


class LiveClient:
    """Socket-backed data source. Use as a context manager; each method is one request."""

    def __init__(self, cookies=None, timeout: float = 20.0, board: str = BOARD):
        self.cookies = cookies
        self.timeout = timeout
        self.board = board
        self.sock = None
        self.requests = 0

    def __enter__(self):
        from scraper import ws
        kwargs = {} if self.cookies is None else {"cookies": self.cookies}
        self.sock = ws.VariantsSocket(timeout=self.timeout, **kwargs).connect()
        return self

    def __exit__(self, *exc):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def leaders(self) -> list[dict]:
        self.sock.send("get.leaderboard", self.board)
        self.requests += 1
        _, reply = self.sock.wait(lambda m, d: m == "leaderboard" and isinstance(d, dict)
                                  and d.get("leaderboardType") == self.board)
        return parse_leaders(reply)

    def watch(self) -> list[dict]:
        self.sock.send("subscribe-watch", WATCH_FILTER)
        self.requests += 1
        try:
            _, games = self.sock.wait(lambda m, d: m == "watch_games", timeout=WATCH_WAIT)
        finally:
            self.sock.send("unsubscribe-watch", WATCH_FILTER)
        return parse_watch(games)

    def scraper(self) -> list[dict]:
        return scraper_players()

    def probe(self, candidates: list[dict], delay: float = PROBE_DELAY):
        from .prober import probe_many
        self.requests += len(candidates)
        return probe_many(self.sock, candidates, delay=delay, board=self.board)
