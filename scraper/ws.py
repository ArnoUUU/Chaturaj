"""Scrape chess.com Chaturaji games over the variants WebSocket, without a browser.

The chess.com variants app gets all its data from one WebSocket,
wss://variants.gcp-prod.chess.com. Every message is JSON: the client sends
{"action", "data", "windowId"} and the server answers with {"mutation", "data"}.
The messages used here:

  connect-user          {...window.context.user, clientVersion}  -> user_connected
  autocomplete-username {username}                              -> set_username_choices
  search-archive        {days, playerId, page, queryId, ...}    -> set/add_games_search_result
  game                  {gameNr, action: "join"}                -> archive_game
  pong                  {timestamp, playerId}                   (answer to the server's ping)

`connect-user` needs the logged-in user object that chess.com puts in each page's HTML
(`context = {...}`). We get it with a plain HTTP request that sends the cookies saved by
`python -m scraper.game_scraping login`, so a browser is only needed to log in.

Compared with the Selenium scraper, this is much faster (about 0.1 s per archive page
and 0.4 s per game, with no page rendering) and gets more data: every move has a
millisecond timestamp and the clock left, and every chat line has its own timestamp and
seat, so chat is placed between moves exactly.

Pure conversion helpers (`game_data`, `archive_row`, `parse_moves`, ...) are unit-tested
in tests/test_ws.py.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .game_scraping import COOKIES, game_summary, parse_headers

WS_URL = "wss://variants.gcp-prod.chess.com"
PAGE_URL = "https://www.chess.com/variants/chaturaji/archive"
CLIENT_VERSION = "0.0.176"   # variants.js client version this was written against
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
SEATS = ("Red", "Blue", "Yellow", "Green")   # username1..4 / rating1..4 / ... order


class NotLoggedIn(RuntimeError):
    pass


# --- pure helpers ------------------------------------------------------------------------

CONTEXT_RE = re.compile(r'\bcontext\s*=\s*(?=\{"userId")')
MOVE_RE = re.compile(r"(?P<move>[^\s{}]+)\s*(?:\{(?P<comment>[^}]*)\})?")
MOVE_NUM_RE = re.compile(r"^\d+\.$")


def page_user(html: str) -> dict | None:
    """The `context.user` object chess.com embeds in its pages (None if not found)."""
    m = CONTEXT_RE.search(html)
    if not m:
        return None
    try:
        ctx, _ = json.JSONDecoder().raw_decode(html, m.end())
    except ValueError:
        return None
    return ctx.get("user") or None


def iso_ms(ms: float | int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    """ISO date ('2026-08-27T03:26:37.554Z') -> epoch milliseconds."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def clean_pgn4(pgn4: str) -> str:
    """Join header values the server splits over lines (StartFen4) so the PGN4 matches
    what the game page shows; movetext and its {date= clock=} comments are kept."""
    return re.sub(r'\[(\w+) "([^"]*)"\]', lambda m: f'[{m.group(1)} "{m.group(2).replace(chr(10), "")}"]', pgn4)


def parse_moves(pgn4: str) -> list[dict]:
    """Moves with their server timestamp and clock, from the PGN4 movetext comments
    ('e5-e6 { date=2026-08-27T03:26:37.554Z clock=61105 }')."""
    body = re.sub(r'\[\w+ "[^"]*"\]', "", pgn4)
    out, prev = [], None
    for m in MOVE_RE.finditer(body):
        tok = m.group("move")
        if MOVE_NUM_RE.match(tok) or tok == "..":
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", m.group("comment") or ""))
        ms = parse_iso(fields.get("date"))
        gap = round((ms - prev) / 1000, 3) if ms is not None and prev is not None else None
        out.append({"move": tok, "time": fields.get("date"), "ply": len(out),
                    "seconds_since_prev": gap,
                    "clock_ms": int(fields["clock"]) if fields.get("clock", "").lstrip("-").isdigit() else None})
        if ms is not None:
            prev = ms
    return out


def place_chat(chat: list[dict], moves: list[dict]) -> list[dict]:
    """Chat lines with exact [after_ply, before_ply] windows from their timestamps.

    after_ply is the last move made at or before the message (None if it came before the
    first move), before_ply the first move after it (None if after the last move).
    """
    times = [parse_iso(m["time"]) for m in moves]
    out = []
    for c in chat:
        t = c.get("time")
        after = max((i for i, mt in enumerate(times) if mt is not None and t is not None and mt <= t), default=None)
        before = min((i for i, mt in enumerate(times) if mt is not None and t is not None and mt > t), default=None)
        system = c.get("type") == "info" or not c.get("playerId")
        out.append({
            "system": system,
            "user": None if system else c.get("username"),
            "text": c.get("message", ""),
            "time": iso_ms(t),
            "seat": c.get("seat"),
            "full_move": c.get("fullMoveNr"),
            "after_ply": after, "before_ply": before,
            "after_time": moves[after]["time"] if after is not None else None,
            "before_time": moves[before]["time"] if before is not None else None,
        })
    return out


def time_control(d: dict) -> str | None:
    """Time control as the archive shows it ('1 | 5', '6s | 3', '30 sec')."""
    tc = (d.get("q") or {}).get("timeControl")
    if tc:
        return tc
    ini, inc = d.get("timeControlInitial"), d.get("timeControlIncrement")
    if ini is None:
        return None
    ini_s = f"{round(ini * 60)}s" if ini < 1 else f"{ini:g}"
    if not inc:
        return f"{round(ini * 60)} sec" if ini < 1 else f"{ini:g} min"
    return f"{ini_s} | {inc:g}"


def _fmt_diff(v) -> str | None:
    if v is None:
        return None
    return f"{v:+g}" if isinstance(v, (int, float)) else str(v)


def seat_players(d: dict) -> list[dict]:
    """Per-seat name, id, rating, rating change, points and place (no IP data).

    `standings` lists 1-based seat numbers in finishing order ([3, 4, 2, 1]: Yellow won).
    """
    standings = d.get("standings") or []
    place_of = {seat: rank for rank, seat in enumerate(standings, 1)}
    out = []
    for i, color in enumerate(SEATS, 1):
        name = d.get(f"username{i}")
        if not name:
            continue
        out.append({"seat": color, "name": name, "uid": d.get(f"uid{i}"),
                    "rating": d.get(f"rating{i}"), "rating_change": _fmt_diff(d.get(f"ratingDiff{i}")),
                    "points": d.get(f"points{i}"),
                    "place": place_of.get(i)})
    return out


def archive_row(row: dict, searched: str | None = None) -> dict:
    """An archive search result in the shape game_scraping uses for its index."""
    q = row.get("q") or {}
    return {
        "game_id": str(row.get("gameNr")),
        "time_control": time_control(row),
        "mode": q.get("typeName"),
        "variant": q.get("title"),
        "variant_url": q.get("variantUrl"),
        "rated": row.get("rated"),
        "date": row.get("date"),
        "duration_min": row.get("duration"),
        "plies": row.get("plies"),
        "termination": row.get("termination"),
        "players": [{**p, "searched": bool(searched) and p["name"].lower() == searched.lower()}
                    for p in seat_players(row)],
    }


def is_chaturaji(row: dict) -> bool:
    q = row.get("q") or {}
    return (q.get("variantUrl") or "").lower() == "chaturaji" or (q.get("title") or "").lower() == "chaturaji"


def game_data(d: dict) -> dict:
    """An `archive_game` message -> the same game dict the Selenium scraper saves."""
    pgn = clean_pgn4(d.get("pgn4") or "")
    headers = parse_headers(pgn)
    moves = parse_moves(pgn)
    summary = game_summary(headers)
    gid = str(d.get("gameNr") or summary["game_id"])
    return {
        **summary,
        "game_id": gid,
        "time_control": time_control(d) or summary["time_control"],
        "url": f"https://www.chess.com/variants/chaturaji/game/{gid}",
        "headers": headers,
        "pgn4": pgn,
        "moves": moves,
        "chat": place_chat(d.get("chat") or [], moves),
        "players": seat_players(d),
        "rated": d.get("rated"),
        "start_date": d.get("date"),
        "end_date": d.get("endDate"),
        "duration_min": d.get("duration"),
        "tournament": d.get("tNr"),
        "source": "websocket",
        "scraped_at": time.time(),
    }


# --- the socket --------------------------------------------------------------------------

class VariantsSocket:
    """A logged-in connection to the chess.com variants server.

    with VariantsSocket() as s:
        uid = s.find_player("arnav2007")
        rows, last = s.search_archive(uid, page=0)
        game = s.game(rows[0]["gameNr"])
    """

    def __init__(self, cookies: Path = COOKIES, timeout: float = 15.0):
        self.cookies_path = Path(cookies)
        self.timeout = timeout
        self.ws = None
        self.user: dict | None = None
        self.window_id = f"py-{time.time():.0f}"

    # connection
    def _http(self):
        import requests
        if not self.cookies_path.exists():
            raise NotLoggedIn(f"no saved login at {self.cookies_path}; run "
                              "`python -m scraper.game_scraping login` first")
        s = requests.Session()
        s.headers["User-Agent"] = UA
        for c in json.loads(self.cookies_path.read_text(encoding="utf-8")):
            s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
        return s

    def connect(self) -> "VariantsSocket":
        import websocket
        http = self._http()
        r = http.get(PAGE_URL, timeout=self.timeout)
        r.raise_for_status()
        user = page_user(r.text)
        if not user or not user.get("id") or user.get("isGuest"):
            raise NotLoggedIn("the saved chess.com login has expired; run "
                              "`python -m scraper.game_scraping login` again")
        self.user = user
        self.ws = websocket.create_connection(
            WS_URL, timeout=self.timeout, origin="https://www.chess.com",
            header=[f"User-Agent: {UA}"],
            cookie="; ".join(f"{c.name}={c.value}" for c in http.cookies))
        self.send("connect-user", {**user, "clientVersion": CLIENT_VERSION})
        self.wait(lambda m, d: m == "user_connected")
        return self

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # messaging
    def send(self, action: str, data=None) -> None:
        self.ws.send(json.dumps({"action": action, "data": data, "windowId": self.window_id}))

    def recv(self) -> tuple[str, object]:
        msg = json.loads(self.ws.recv())
        data = msg.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                pass
        mutation = msg.get("mutation", "")
        if mutation == "ping":
            self.send("pong", {"timestamp": data, "playerId": (self.user or {}).get("id")})
        return mutation, data

    def wait(self, pred, timeout: float | None = None):
        """Read messages until pred(mutation, data) is true; return (mutation, data)."""
        end = time.time() + (timeout or self.timeout)
        while time.time() < end:
            mutation, data = self.recv()
            if pred(mutation, data):
                return mutation, data
        raise TimeoutError("no matching message from the variants server")

    def _retry(self, fn):
        """Run fn once, and again on a fresh connection if the socket dropped."""
        import websocket
        try:
            return fn()
        except (websocket.WebSocketConnectionClosedException, ConnectionError, OSError):
            self.close()
            self.connect()
            return fn()

    # requests
    def find_player(self, username: str) -> int:
        def go():
            self.send("autocomplete-username", {"username": username})
            _, d = self.wait(lambda m, d: m == "set_username_choices")
            return d
        choices = self._retry(go)
        players = choices.get("players", []) if isinstance(choices, dict) else choices or []
        for p in players:
            if p.get("username", "").lower() == username.lower():
                return p["uid"]
        raise LookupError(f"no chess.com variants player named {username!r}")

    def search_archive(self, player_id: int, page: int = 0, days: str = "0-9999",
                       **filters) -> tuple[list[dict], bool]:
        """One page (20 games) of a player's archive, newest first. Returns (rows, last_page)."""
        def go():
            qid = int(time.time() * 1000)
            q = {"days": days, "playerId": player_id, "gameType": "", "gameNr": "",
                 "ratingType": "", "title": "", "queryId": qid, **filters}
            if page:
                q["page"] = page
            self.send("search-archive", q)
            _, d = self.wait(lambda m, d: m in ("set_games_search_result", "add_games_search_result")
                             and isinstance(d, dict) and d.get("queryId") == qid)
            return d
        d = self._retry(go)
        return d.get("games", []), bool(d.get("lastPage"))

    def iter_archive(self, player_id: int, days: str = "0-9999", page_delay: float = 0.5,
                     max_pages: int | None = None):
        page = 0
        while max_pages is None or page < max_pages:
            rows, last = self.search_archive(player_id, page, days)
            yield from rows
            if last or not rows:
                return
            page += 1
            time.sleep(page_delay)

    def game(self, game_nr: int | str) -> dict:
        """The full stored game (PGN4 with move times and clocks, chat, ratings)."""
        nr = int(game_nr)

        def go():
            self.send("game", {"gameNr": nr, "action": "join"})
            _, d = self.wait(lambda m, d: m == "archive_game" and isinstance(d, dict)
                             and int(d.get("gameNr") or 0) == nr)
            self.send("game", {"gameNr": nr, "action": "exit"})
            return d
        return self._retry(go)


# --- scraping entry points (same storage as the Selenium scraper) ------------------------

def known_game_ids(name: str) -> set[str]:
    from .game_scraping import players_dir
    base = players_dir() / name.lower()
    ids = set()
    for idx in base.glob("*/index.jsonl") if base.exists() else []:
        for line in idx.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(str(json.loads(line).get("game_id")))
    return ids


def _load_saved(gid: str) -> dict | None:
    from .game_scraping import games_dir
    path = games_dir() / f"{gid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scrape_player(sock: VariantsSocket, name: str, time_controls: list[str] | None = None,
                  max_per_tc: int = 20, days: str = "0-9999", delay: float = 1.0,
                  refetch: bool = False, max_pages: int | None = None) -> dict[str, int]:
    """Scrape a player's Chaturaji games, saved per time control (like game_scraping's
    Selenium `scrape_player`). Pages through the whole archive in one search.

    Games already in this player's index are skipped by id. Games already saved for
    another player are added to this player's index without downloading them again.
    Returns {time_control folder (e.g. '1+5'): games added this run}.
    """
    from collections import defaultdict
    from .game_scraping import _pause, add_to_player_index, log_failure, store_game, tc_slug
    wanted = {tc_slug(t) for t in time_controls} if time_controls else None
    saved: dict[str, int] = defaultdict(int)
    known = set() if refetch else known_game_ids(name)
    uid = sock.find_player(name)
    rows_seen = 0
    from .game_scraping import records_dir
    for raw in sock.iter_archive(uid, days, max_pages=max_pages):
        if wanted and all(saved[t] >= max_per_tc for t in wanted):
            break
        rows_seen += 1
        if not is_chaturaji(raw):
            continue
        row = archive_row(raw, searched=name)
        tc, gid = row["time_control"], row["game_id"]
        slug = tc_slug(tc)
        if (wanted and slug not in wanted) or gid in known or saved[slug] >= max_per_tc:
            continue
        data = None if refetch else _load_saved(gid)
        if data is not None:   # downloaded before (e.g. for an opponent): just index it
            add_to_player_index(name, data, row,
                                record_ok=(records_dir() / f"{gid}.json").exists())
            status = "indexed (already saved)"
        else:
            try:
                data = game_data(sock.game(gid))
            except (TimeoutError, LookupError) as e:
                log_failure(gid, f"ws: {e}")
                print(f"  {tc_slug(tc)}: {gid} skipped: {e}", flush=True)
                continue
            status = store_game(data, row, player=name)
            _pause(delay)
        known.add(gid)
        saved[slug] += 1
        print(f"  {slug}: {gid} ({len(data.get('moves', []))} plies) {status}"
              f" [{saved[slug]}/{max_per_tc}]", flush=True)
    print(f"[{name}] {rows_seen} archive rows read")
    return dict(saved)


def scrape_games(sock: VariantsSocket, ids: list[str], delay: float = 1.0, refetch: bool = False,
                 player: str | None = None) -> None:
    from .game_scraping import _pause, games_dir, log_failure, store_game
    ids = list(dict.fromkeys(ids))
    for n, gid in enumerate(ids, 1):
        if not refetch and (games_dir() / f"{gid}.json").exists():
            print(f"[{n}/{len(ids)}] {gid} already saved")
            continue
        try:
            data = game_data(sock.game(gid))
        except TimeoutError as e:
            log_failure(gid, f"ws: {e}")
            print(f"[{n}/{len(ids)}] {gid} skipped: {e}")
            continue
        print(f"[{n}/{len(ids)}] {gid}: {store_game(data, player=player)}")
        _pause(delay)
