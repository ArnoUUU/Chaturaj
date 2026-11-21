"""Scrape chess.com Chaturaji games with Selenium, by player and by time control.

Main entry point: `scrape_player(driver, name, time_controls=...)`. It opens the variants
archive, searches the player, walks through date ranges (the archive shows at most ~20
games per search), and for each game in the requested time controls:
  - opens the game page and records the PGN4, players, seats, ratings, rating changes,
    points, time control, anonymity, termination, every move with its timestamp, the
    full chat (each message placed between timed game events) and the observer count
  - replays the PGN4 through the rules engine and saves a validated `chaturaji` record
  - adds the game to that player's index for that time control

Storage layout (see scraper/README.md):
  scraper/data/games/<game_id>.json                    full scraped game (one copy per game)
  scraper/data/records/<game_id>.json                  engine record (viewer / training)
  scraper/data/players/<name>/<time_control>/index.jsonl   one line per game for this player
  scraper/data/players/<name>/<time_control>/summary.json  totals for this player + control
  scraper/data/failed.jsonl                            games that could not be scraped/converted

Usage (from the repo root):
  python -m scraper.game_scraping login
  python -m scraper.game_scraping player arnav2007 --time-control "1 | 5" --max-per-tc 20
  python -m scraper.game_scraping player arnav2007                 # every time control
  python -m scraper.game_scraping games 113032695 113031547
  python -m scraper.game_scraping archive --url "<archive URL with p=<username>>"
  python -m scraper.game_scraping convert                          # rebuild records/

Chess.com's Terms of Service restrict automated access, so use your own account and keep
the default delays.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / ".selenium-profile"
DATA = HERE / "data"
ARCHIVE_URL = "https://www.chess.com/variants/chaturaji/archive"
GAME_URL = "https://www.chess.com/variants/chaturaji/game/{id}"
GAME_ID_RE = re.compile(r"(?:game/)?(\d{6,})")
CHESSCOM_OFFSET = (3, 3)  # chess.com uses a 14x14 board; the 8x8 area starts at d4
# Archive date ranges, newest first. Each search shows ~20 games, so walking these
# reaches further back in a player's history.
DATE_WINDOWS = ("0-1", "0-7", "0-14", "14-31", "30-92", "90-183", "180-276", "274-366", "365-9999")

# --- in-page extraction (plain JS, run with execute_script) ---------------------------

ARCHIVE_JS = r"""
const q = (s, r) => r.querySelector(s);
return [...document.querySelectorAll('.archive-results-game-container')].map((c, index) => {
  const text = c.innerText;
  const lines = text.split('\n').map(s => s.trim()).filter(Boolean);
  const players = [...c.querySelectorAll('.archive-results-game-player')].map(p => ({
    name: (q('.archive-results-user-name', p) || p).innerText.trim(),
    rating: ((q('.archive-results-rating', p) || {}).innerText || '').replace(/[()]/g, '').trim() || null,
    rating_change: ((q('.archive-results-rating-diff', p) || {}).innerText || '').trim() || null,
    points: ((q('.archive-results-points', p) || {}).innerText || '').trim() || null,
    searched: !!p.querySelector('.playerWasSearched'),
  }));
  const date = (text.match(/[A-Z][a-z]{2} \d{1,2}(?:, \d{4})? \d{1,2}:\d{2} [AP]M/) || [null])[0];
  const duration = (text.match(/•\s*(\d+)'/) || [])[1] || null;
  const plies = (text.match(/•\s*(\d+)\s*\n?\s*p\b/) || [])[1] || null;
  return {index, time_control: lines[0] || null, mode: lines[1] || null, variant: lines[2] || null,
          date, duration_min: duration && +duration, plies: plies && +plies, players, text};
});
"""

GAME_JS = r"""
const q = (s, r=document) => r.querySelector(s);
const qa = (s, r=document) => [...r.querySelectorAll(s)];
const m = location.href.match(/game\/(\d+)/);
const moves = qa('.moves-pointer').map(e => {
  const t = e.getAttribute('title') || '';
  const i = t.lastIndexOf(' • ');
  return i < 0 ? {move: t || e.innerText.trim(), time: null} : {move: t.slice(0, i), time: t.slice(i + 3)};
});
const chat = []; let last = null;
for (const row of qa('.chat-message-row')) {
  const sys = !!q('.chat-info-msg', row);
  const userEl = q('.chat-username-f', row);
  let text = (q('.chat-chat-content-f', row) || row).innerText.trim();
  let user = null;
  if (!sys) {
    if (userEl) { user = userEl.innerText.replace(/:\s*$/, '').trim(); last = user; }
    else { user = last; }
    if (userEl && text.startsWith(userEl.innerText)) text = text.slice(userEl.innerText.length).trim();
  }
  chat.push({system: sys, user, text});
}
const players = qa('.playerbox-component').map(b => {
  const t = ((q('.playerbox-player', b) || {}).innerText || '').replace(/\n/g, ' ');
  return {
    name: (t.match(/^\s*([^\s(]+)/) || [])[1] || null,
    rating: (t.match(/\((\d+)\??\)/) || [])[1] || null,
    points: ((q('.playerbox-points', b) || {}).innerText || '').trim() || null,
    clock: ((q('.playerbox-clock', b) || {}).innerText || '').trim() || null,
  };
});
return {url: location.href, id: m && m[1], pgn: (q('.pgn4-pgn-box') || {}).value || null,
        moves, chat, players,
        observers: ((q('.chat-observer-count') || {}).innerText || '').replace(/\s+/g, ' ').trim() || null};
"""

# --- pure helpers (unit-tested in tests/test_game_scraping.py) ---------------------

HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
EVENT_RE = re.compile(r"^(?P<who>\S+) (?P<what>checkmated!|resigned\.|timed out\.|claimed the win\.|stalemated!)")


def parse_headers(pgn: str | None) -> dict[str, str]:
    return dict(HEADER_RE.findall(pgn or ""))


def game_summary(headers: dict[str, str]) -> dict:
    """Players, ratings, time control, anonymity and result from PGN4 headers."""
    seats = {}
    for color in ("Red", "Blue", "Yellow", "Green"):
        if color in headers:
            elo = headers.get(color + "Elo")
            seats[color] = {"name": headers[color], "rating": int(elo) if elo and elo.isdigit() else None}
    result = {}
    for part in headers.get("Result", "").split(" - "):
        if ": " in part:
            name, pts = part.rsplit(": ", 1)
            result[name] = int(pts) if pts.lstrip("-").isdigit() else pts
    rules = headers.get("RuleVariants", "").split()
    return {
        "game_id": headers.get("GameNr"),
        "time_control": headers.get("TimeControl"),
        "variant": headers.get("Variant"),
        "rule_variants": rules,
        "anonymity": next((r for r in rules if "Anonymous" in r), None),
        "seats": seats,
        "final_points": result,
        "termination": headers.get("Termination"),
        "date": headers.get("Date"),
    }


def _seconds(hms: str | None) -> int | None:
    if not hms:
        return None
    parts = [int(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def add_move_timing(moves: list[dict]) -> list[dict]:
    """Add ply index and seconds since the previous move (handles midnight wrap)."""
    out, prev = [], None
    for i, mv in enumerate(moves):
        sec = _seconds(mv.get("time"))
        gap = None
        if sec is not None and prev is not None:
            gap = (sec - prev) % 86400
        out.append({**mv, "ply": i, "seconds_since_prev": gap})
        if sec is not None:
            prev = sec
    return out


def _event_ply(moves: list[dict], kind: str, used: set[int]) -> int | None:
    """Ply of the first unused move matching a chat event kind."""
    def match(mv: str) -> bool:
        if kind == "checkmated!":
            return "#" in mv
        if kind in ("resigned.", "claimed the win."):
            return mv.strip() == "R"
        if kind == "timed out.":
            return mv.strip() == "T"
        if kind == "stalemated!":
            return mv.strip() == "S" or mv.endswith("S")
        return False
    for i, mv in enumerate(moves):
        if i not in used and match(mv["move"]):
            return i
    return None


def align_chat(chat: list[dict], moves: list[dict]) -> list[dict]:
    """Give each chat line a [after_ply, before_ply] window using timed system events.

    System lines such as "X checkmated!" or "X resigned." correspond to moves ('#', 'R',
    'T') that have timestamps. A message between two such lines in the chat was sent
    between those two moves. Lines before the first event get after_ply=None, and lines
    after the last event get before_ply=None (i.e. "after the game reached that point").
    """
    used: set[int] = set()
    anchors: list[tuple[int, int]] = []  # (chat index, ply)
    for ci, line in enumerate(chat):
        if not line["system"]:
            continue
        m = EVENT_RE.match(line["text"])
        if m:
            ply = _event_ply(moves, m.group("what"), used)
            if ply is not None:
                used.add(ply)
                anchors.append((ci, ply))
    out = []
    for ci, line in enumerate(chat):
        before = [p for a, p in anchors if a < ci]
        after = [p for a, p in anchors if a > ci]
        lo = max(before) if before else None
        hi = min(after) if after else None
        own = next((p for a, p in anchors if a == ci), None)
        window = {
            "after_ply": own if own is not None else lo,
            "before_ply": own if own is not None else hi,
        }
        for key, ply in (("after_time", window["after_ply"]), ("before_time", window["before_ply"])):
            window[key] = moves[ply]["time"] if ply is not None else None
        out.append({**line, **window})
    return out


# --- browser -----------------------------------------------------------------------------

def make_driver(headless: bool = False):
    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    PROFILE.mkdir(parents=True, exist_ok=True)
    opts.add_argument(f"--user-data-dir={PROFILE}")
    opts.add_argument("--window-size=1400,1000")
    if headless:
        opts.add_argument("--headless=new")
    return webdriver.Chrome(options=opts)


def wait_for(driver, js_condition: str, timeout: float = 20.0, poll: float = 0.4) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if driver.execute_script(f"return !!({js_condition});"):
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def scrape_game_page(driver, gid: str | None = None) -> dict:
    """Scrape the currently open game page (wait until its PGN4 is loaded)."""
    cond = ("(() => { const v = (document.querySelector('.pgn4-pgn-box')||{}).value || '';"
            " const m = location.href.match(/game\\/(\\d+)/);"
            " return m && v.includes('[GameNr \"' + m[1] + '\"]') && /\\n1\\. /.test(v); })()")
    if not wait_for(driver, cond, timeout=25):
        raise TimeoutError(f"game page did not load its PGN4 ({driver.current_url})")
    time.sleep(1.0)  # let the chat and move list finish rendering
    raw = driver.execute_script(GAME_JS)
    headers = parse_headers(raw["pgn"])
    moves = add_move_timing(raw["moves"])
    return {
        **game_summary(headers),
        "url": raw["url"],
        "headers": headers,
        "pgn4": raw["pgn"],
        "moves": moves,
        "chat": align_chat(raw["chat"], moves),
        "player_boxes": raw["players"],
        "observers": raw["observers"],
        "scraped_at": time.time(),
    }


# --- storage -------------------------------------------------------------------------

def games_dir() -> Path:
    return DATA / "games"


def records_dir() -> Path:
    return DATA / "records"


def players_dir() -> Path:
    return DATA / "players"


def tc_slug(time_control: str | None) -> str:
    """Folder-safe name for a time control: '1 | 5' -> '1+5', '6s | 3' -> '6s+3', '30 sec' -> '30s'."""
    s = (time_control or "").strip().lower().replace(" ", "").replace("|", "+")
    s = re.sub(r"sec(onds?)?$", "s", s)
    s = re.sub(r"[^0-9a-z+]", "", s)
    return s or "unknown"


def player_tc_dir(name: str, time_control: str) -> Path:
    return players_dir() / name.lower() / tc_slug(time_control)


def row_signature(row: dict) -> str:
    """Identify an archive row before opening it (date + the four player names)."""
    names = sorted(p["name"].lower() for p in row.get("players", []))
    return f"{row.get('date')}|{','.join(names)}"


def placement(final_points: dict[str, int], name: str) -> int | None:
    """1-4 place of `name` by final points (ties share the better place)."""
    key = next((k for k in final_points if k.lower() == name.lower()), None)
    if key is None or not isinstance(final_points[key], int):
        return None
    return 1 + sum(1 for v in final_points.values() if isinstance(v, int) and v > final_points[key])


def read_index(name: str, time_control: str) -> list[dict]:
    path = player_tc_dir(name, time_control) / "index.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def known_signatures(name: str) -> set[str]:
    base = players_dir() / name.lower()
    sigs = set()
    for idx in base.glob("*/index.jsonl") if base.exists() else []:
        for line in idx.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sigs.add(json.loads(line).get("signature"))
    return sigs


def save_game_data(data: dict) -> Path:
    games_dir().mkdir(parents=True, exist_ok=True)
    path = games_dir() / f"{data['game_id']}.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def save_record(data: dict) -> tuple[Path | None, str | None]:
    """Replay the PGN4 through the engine; save a chaturaji Game record or return the error."""
    from .pgn4 import PGN4Error, to_game
    try:
        game = to_game(data["pgn4"], CHESSCOM_OFFSET, {
            "chesscom_id": data["game_id"], "time_control": data.get("time_control"),
            "seats": data.get("seats"), "final_points": data.get("final_points")})
    except (PGN4Error, KeyError, TypeError) as e:
        log_failure(data.get("game_id"), f"record: {e}")
        return None, str(e)
    records_dir().mkdir(parents=True, exist_ok=True)
    path = records_dir() / f"{data['game_id']}.json"
    game.save(path)
    return path, None


def log_failure(game_id, error: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "failed.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": game_id, "error": error, "at": time.time()}) + "\n")


def add_to_player_index(name: str, data: dict, row: dict | None = None,
                        record_ok: bool = False) -> bool:
    """Append this game to the player's index for its time control. Returns False if
    the game is already listed or the player is not in it."""
    seat = next((c for c, s in data.get("seats", {}).items() if s["name"].lower() == name.lower()), None)
    if seat is None:
        return False
    tc = (row or {}).get("time_control") or data.get("time_control") or "unknown"
    if any(e["game_id"] == data["game_id"] for e in read_index(name, tc)):
        return False
    me = next((p for p in (row or {}).get("players", []) if p["name"].lower() == name.lower()), {})
    entry = {
        "game_id": data["game_id"],
        "date": data.get("date"),
        "archive_date": (row or {}).get("date"),
        "time_control": tc,
        "seat": seat,
        "rating": data["seats"][seat].get("rating"),
        "rating_change": me.get("rating_change"),
        "points": next((v for k, v in data.get("final_points", {}).items() if k.lower() == name.lower()), None),
        "place": placement(data.get("final_points", {}), name),
        "opponents": [s["name"] for c, s in data["seats"].items() if c != seat],
        "termination": data.get("termination"),
        "anonymity": data.get("anonymity"),
        "plies": len(data.get("moves", [])),
        "chat_messages": sum(1 for c in data.get("chat", []) if not c["system"]),
        "record": record_ok,
        "signature": row_signature(row) if row else None,
    }
    folder = player_tc_dir(name, tc)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    write_summary(name, tc)
    return True


def write_summary(name: str, time_control: str) -> dict:
    rows = read_index(name, time_control)
    places = [r["place"] for r in rows if r.get("place")]
    points = [r["points"] for r in rows if isinstance(r.get("points"), int)]
    summary = {
        "player": name, "time_control": time_control, "games": len(rows),
        "wins": sum(1 for p in places if p == 1),
        "average_place": round(sum(places) / len(places), 2) if places else None,
        "average_points": round(sum(points) / len(points), 2) if points else None,
        "ratings": [r["rating"] for r in rows if r.get("rating")],
        "updated_at": time.time(),
    }
    (player_tc_dir(name, time_control) / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def store_game(data: dict, row: dict | None = None, player: str | None = None) -> str:
    """Save the scraped game, its engine record, and (optionally) the player's index entry."""
    if row is not None:
        data["archive_row"] = row
    save_game_data(data)
    record_path, err = save_record(data)
    if player:
        add_to_player_index(player, data, row, record_ok=record_path is not None)
    return "ok" if err is None else f"saved, record failed: {err}"


# --- archive navigation ---------------------------------------------------------------

def _pause(delay: float) -> None:
    time.sleep(delay + random.uniform(0, delay / 2))


def _run_archive_search(driver, username: str) -> None:
    """The archive search only runs when a username suggestion is clicked."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    box = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Username or Game #"]')
    box.click()
    box.send_keys(Keys.CONTROL, "a")
    box.send_keys(username)
    target = re.compile(rf"^{re.escape(username)} \(\d+\)$", re.I)
    end = time.time() + 10
    while time.time() < end:
        for el in driver.find_elements(By.XPATH, "//*[not(*)][contains(text(), '(')]"):
            try:
                if target.match(el.text.strip()) and el.is_displayed():
                    el.click()
                    return
            except Exception:
                pass
        time.sleep(0.4)
    raise TimeoutError(f"no archive suggestion appeared for {username!r}")


def _set_date_window(driver, window: str) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    for el in driver.find_elements(By.CSS_SELECTOR, "select.archive-search-small, .archive-search-days select"):
        if window in [o.get_attribute("value") for o in Select(el).options]:
            Select(el).select_by_value(window)
            return


def _click_show_more(driver, times: int) -> None:
    from selenium.webdriver.common.by import By
    for _ in range(times):
        buttons = [b for b in driver.find_elements(
            By.CSS_SELECTOR, ".archive-results-results-wrapper button") if b.text.strip() == "Show more"]
        if not buttons:
            return
        before = driver.execute_script("return document.querySelectorAll('.archive-results-game-container').length;")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", buttons[-1])
        buttons[-1].click()
        time.sleep(2.0)
        after = driver.execute_script("return document.querySelectorAll('.archive-results-game-container').length;")
        if after <= before:
            return


def open_archive(driver, username: str, window: str = "0-9999", show_more: int = 3) -> list[dict]:
    """Search `username` in the Chaturaji archive for one date window; return the rows."""
    driver.get(ARCHIVE_URL)
    if not wait_for(driver, "document.querySelector('input[placeholder=\"Username or Game #\"]')", 20):
        raise TimeoutError("archive page did not load (logged in?)")
    _set_date_window(driver, window)
    _run_archive_search(driver, username)
    # wait for results, or for the counter to settle on 0 (empty window)
    wait_for(driver, "document.querySelector('.archive-results-game-container') ||"
                     " (document.querySelector('.archive-search-row-count')||{}).innerText === '0'", 15)
    time.sleep(1.5)
    _click_show_more(driver, show_more)
    rows = driver.execute_script(ARCHIVE_JS)
    return [r for r in rows if (r.get("variant") or "").lower() == "chaturaji"]


def _open_row(driver, index: int) -> None:
    driver.execute_script(
        "const c=document.querySelectorAll('.archive-results-game-container')[arguments[0]];"
        " c.scrollIntoView({block:'center'});"
        " (c.querySelector('.archive-results-game-title')||c).click();", index)


# --- scraping entry points --------------------------------------------------------------

def scrape_player(driver, name: str, time_controls: list[str] | None = None,
                  max_per_tc: int = 20, windows=DATE_WINDOWS, show_more: int = 3,
                  delay: float = 4.0, refetch: bool = False) -> dict[str, int]:
    """Scrape a player's Chaturaji games, grouped and saved per time control.

    time_controls: controls as the archive shows them, e.g. ["1 | 5", "30 sec"];
                   None means every control.
    max_per_tc:    new games to save per time control in this run.
    Returns {time_control: games saved this run}.
    """
    saved: dict[str, int] = defaultdict(int)
    seen = set() if refetch else known_signatures(name)

    def full(tc: str) -> bool:
        return saved[tc] >= max_per_tc

    for window in windows:
        rows = open_archive(driver, name, window, show_more)
        todo = [r for r in rows
                if (not time_controls or r["time_control"] in time_controls)
                and row_signature(r) not in seen]
        print(f"[{name}] window {window}: {len(rows)} rows, {len(todo)} new in requested controls")
        for row in todo:
            tc = row["time_control"]
            if full(tc):
                continue
            present = driver.execute_script(
                "return document.querySelectorAll('.archive-results-game-container').length;")
            if present <= row["index"]:
                open_archive(driver, name, window, show_more)
            _open_row(driver, row["index"])
            try:
                data = scrape_game_page(driver)
            except TimeoutError as e:
                log_failure(None, f"{name} {window} row {row['index']}: {e}")
                continue
            status = store_game(data, row, player=name)
            seen.add(row_signature(row))
            saved[tc] += 1
            print(f"  {tc_slug(tc)}: {data['game_id']} ({len(data['moves'])} plies) {status}")
            _pause(delay)
        if time_controls and all(full(tc) for tc in time_controls):
            break
    return dict(saved)


def scrape_games(driver, ids: list[str], delay: float = 4.0, refetch: bool = False,
                 player: str | None = None) -> None:
    for n, gid in enumerate(dict.fromkeys(ids), 1):
        if not refetch and (games_dir() / f"{gid}.json").exists():
            print(f"[{n}/{len(ids)}] {gid} already saved")
            continue
        driver.get(GAME_URL.format(id=gid))
        try:
            data = scrape_game_page(driver, gid)
        except TimeoutError as e:
            log_failure(gid, str(e))
            print(f"[{n}/{len(ids)}] {gid} skipped: {e}")
            continue
        print(f"[{n}/{len(ids)}] {gid}: {store_game(data, player=player)}")
        _pause(delay)


def convert_all() -> tuple[int, int]:
    """Rebuild records/ from every saved game (e.g. after an engine or parser change)."""
    files = sorted(games_dir().glob("*.json"))
    ok = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        ok += save_record(data)[0] is not None
    return ok, len(files)


# --- CLI ---------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login", help="open Chrome to log in; the session is kept")

    p = sub.add_parser("player", help="scrape a player's games, saved per time control")
    p.add_argument("name")
    p.add_argument("--time-control", action="append", dest="time_controls",
                   help='as shown in the archive, e.g. "1 | 5" (repeatable; default: all)')
    p.add_argument("--max-per-tc", type=int, default=20)
    p.add_argument("--windows", nargs="+", default=list(DATE_WINDOWS))
    p.add_argument("--show-more", type=int, default=3)

    g = sub.add_parser("games", help="scrape specific games by URL or id")
    g.add_argument("games", nargs="+")
    g.add_argument("--player", help="also add them to this player's index")

    a = sub.add_parser("archive", help="scrape the rows of an archive URL (must contain p=<username>)")
    a.add_argument("--url", required=True)
    a.add_argument("--time-control", action="append", dest="time_controls")
    a.add_argument("--max-per-tc", type=int, default=20)

    sub.add_parser("convert", help="rebuild engine records from saved games")

    for sp in (p, g, a):
        sp.add_argument("--delay", type=float, default=4.0, help="seconds between games")
        sp.add_argument("--refetch", action="store_true")
        sp.add_argument("--headless", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "convert":
        ok, total = convert_all()
        print(f"converted {ok}/{total}")
        return
    driver = make_driver(getattr(args, "headless", False))
    try:
        if args.cmd == "login":
            driver.get("https://www.chess.com/login")
            input("Log in to chess.com in the Chrome window, then press Enter here... ")
        elif args.cmd == "player":
            print(scrape_player(driver, args.name, args.time_controls, args.max_per_tc,
                                args.windows, args.show_more, args.delay, args.refetch))
        elif args.cmd == "games":
            ids = [m.group(1) for m in map(GAME_ID_RE.search, args.games) if m]
            scrape_games(driver, ids, args.delay, args.refetch, args.player)
        elif args.cmd == "archive":
            m = re.search(r"[?&]p=([^&]+)", args.url)
            if not m:
                raise SystemExit("the archive URL must contain p=<username>")
            window = (re.search(r"daysBack=([\d-]+)", args.url) or [None, "0-9999"])[1]
            print(scrape_player(driver, m.group(1), args.time_controls, args.max_per_tc,
                                (window,), delay=args.delay, refetch=args.refetch))
    finally:
        driver.quit()


if __name__ == "__main__":
    sys.exit(main())
