"""Play a timed game against bots in your browser (bot/PLAN.md milestone M7).

    python -m gui.play                                # you are Red; bots play the other seats
    python -m gui.play --human R B                    # two people take turns at one screen
    python -m gui.play --seat R=human B=uniform Y=models/opponent_pool/BEST_taran0915_model.pt G=human
    python -m gui.play --human R --no-human-clock     # people play without a clock
    python -m gui.play --human none --no-open         # bots only, played out in the terminal

The page opens on a setup screen: a dropdown for each side (Human, or a bot found in models/)
and a time control (1|5 means 1 minute plus a 5-second increment per move). During a game
you can change a side's player between moves, pause the clocks and the bots' thinking, and
step back and forward through the moves to review them while the game goes on. Clocks work
like chess.com: each side has its own clock, the increment is added after each move, and a
side that runs out of time is eliminated on time (its pieces turn grey). Bots choose how long
to think from their own clock (bot/timing.py). Finished games are saved to games/played/ and
open in the replay viewer: python -m gui.viewer games/played

Everything runs on this machine: a small standard-library web server on 127.0.0.1. Piece
sprites come from gui/sprites.py: by default chess.com's artwork (downloaded once and
cached in gui/.cache, never committed); --sprites builtin uses the simple built-in pieces.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from bot.features import rotate_square
from bot.mcts import MCTS, Node, UniformEvaluator
from bot.timing import Clock, move_budget
from chaturaji.constants import TURN_ORDER
from chaturaji.game import is_game_over
from chaturaji.moves import Move, Resign, legal_moves
from chaturaji.record import Game
from chaturaji.state import index_to_square, square_to_index

from .sprites import COLORS as SPRITE_COLORS, PIECE_TEMPLATES, fallback_svg, sprite_set
from .viewer import _NAMES

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "games" / "played"
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"play.css": "text/css", "play.js": "application/javascript"}
UNIFORM = "uniform"
HUMAN = "human"
PLAYERS_DIR = ROOT / "scraper" / "data" / "players"
FALLBACK_CLOCKS = ["1|5", "1|0", "3|2", "5|0", "10|5", "0.5|2"]


# --- bots -------------------------------------------------------------------------------

def _index_rows() -> list[dict]:
    """Rows of models/index.jsonl whose weights can still be found."""
    from bot.registry import find_model
    index = ROOT / "models" / "index.jsonl"
    rows = []
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                find_model(row["label"])
            except (ValueError, KeyError, FileNotFoundError):
                continue
            rows.append(row)
    return rows


def best_indexed_model() -> Path | None:
    """The model with the best recorded score share in models/index.jsonl."""
    from bot.registry import find_model
    best, best_share = None, -1.0
    for row in _index_rows():
        share = (row.get("evaluation") or {}).get("score_share") or 0.0
        if share > best_share:
            best, best_share = find_model(row["label"]), share
    return best


def default_bot() -> str:
    """The benchmark model if one is set, else the best evaluated model, else uniform."""
    from bot import benchmark
    from bot.registry import find_model
    bench = benchmark.current()
    if bench:
        try:
            return str(find_model(bench["model"]))
        except FileNotFoundError:
            pass
    best = best_indexed_model()
    return str(best) if best else UNIFORM


def list_bots() -> dict:
    """Everything a side can be set to (Human, the bots) and which one is the default."""
    from bot import benchmark
    from bot.registry import find_model
    options = [{"value": HUMAN, "label": "Human"}]
    bench = benchmark.current()
    bench_path = find_model(bench["model"]).resolve() if bench else None
    for path in sorted((ROOT / "models" / "opponent_pool").glob("*.pt"), key=lambda p: p.name):
        label = path.stem.replace("BEST_", "")
        if path.name.startswith("BEST_"):
            label += " (best)"
        if path.resolve() == bench_path:
            label += " (benchmark)"
        options.append({"value": f"opponent_pool/{path.name}", "label": f"Pool: {label}"})
    for row in _index_rows():
        options.append({"value": row["label"], "label": f"Model: {row['label']}"})
    options.append({"value": UNIFORM, "label": "Uniform (no model, weak)"})
    # the default bot is a file path or 'uniform'; pick the option that names the same file
    default_value = UNIFORM
    target = default_bot()
    if target != UNIFORM:
        target_path = Path(target).resolve()
        for opt in options:
            if opt["value"] in (HUMAN, UNIFORM):
                continue
            try:
                if find_model(opt["value"]).resolve() == target_path:
                    default_value = opt["value"]
                    break
            except FileNotFoundError:
                continue
    return {"options": options, "default": default_value}


class EngineBank:
    """Loads each bot once and shares it between games."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._evals: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, ref: str):
        with self._lock:
            if ref not in self._evals:
                if ref == UNIFORM:
                    self._evals[ref] = UniformEvaluator()
                else:
                    from bot.evaluate import load_evaluator
                    from bot.registry import find_model
                    self._evals[ref] = load_evaluator(find_model(ref), self.device)
            return self._evals[ref]

    def name(self, ref: str) -> str:
        if ref == UNIFORM:
            return "Uniform"
        from bot.evaluate import model_name
        from bot.registry import find_model
        return model_name(find_model(ref))


def parse_seat_value(value, color: str) -> str | None:
    """'human' or '' -> None (a person); 'uniform' or a model reference -> that reference."""
    v = str(HUMAN if value is None else value).strip()
    if v in ("", HUMAN):
        return None
    if v != UNIFORM:
        from bot.registry import find_model
        try:
            find_model(v)
        except FileNotFoundError:
            raise ValueError(f"unknown bot for {_NAMES[color]}: {v}") from None
    return v


def parse_seats(raw: dict) -> dict:
    """{'R': 'human' | 'uniform' | model ref, ...} -> {'R': None | ref}. Raises ValueError."""
    return {c: parse_seat_value(raw.get(c), c) for c in TURN_ORDER}


# --- time controls from the scrape ----------------------------------------------------

def _slug_clock(slug: str) -> tuple[str, str] | None:
    """A scraper folder name -> (clock value 'minutes|increment', archive label), or None.

    Folder names come from scraper/game_scraping.py tc_slug: '1+5' = 1 min + 5 s, '6s+3' = 6 s + 3 s,
    '15s' = 15 s with no increment, '4min' = 4 min. Delay controls ('6s+6d') are skipped."""
    import re
    m = re.fullmatch(r"(\d+(?:\.\d{1,3})?)(s|sec|min)?(?:\+(\d+))?", slug)
    if not m:
        return None
    base, unit, inc = m.group(1), m.group(2), m.group(3)
    n = float(base)
    seconds_unit = unit in ("s", "sec")
    minutes = n / 60 if seconds_unit else n
    if minutes <= 0 or minutes > 60:
        return None
    inc_s = int(inc) if inc else 0
    value = f"{minutes:g}|{inc_s}"
    if inc:   # the archive's own style: "1 | 5", "6s | 3"
        label = f"{n:g}s | {inc_s}" if seconds_unit else f"{n:g} | {inc_s}"
    else:
        label = f"{n:g} sec" if seconds_unit else f"{n:g} min"
    return value, label


def scraped_time_controls() -> list[dict]:
    """Time controls the scraper has recorded, most games first: [{value, label, games}]."""
    import json as _json
    games: dict[str, int] = {}
    labels: dict[str, str] = {}
    if PLAYERS_DIR.exists():
        for summary in PLAYERS_DIR.glob("*/*/summary.json"):
            parsed = _slug_clock(summary.parent.name)
            if parsed is None:
                continue
            value, label = parsed
            try:
                count = int(_json.loads(summary.read_text(encoding="utf-8")).get("games") or 0)
            except (ValueError, OSError):
                continue
            games[value] = games.get(value, 0) + count
            labels.setdefault(value, label)
    rows = [{"value": v, "label": labels[v], "games": n} for v, n in games.items() if n > 0]
    rows.sort(key=lambda r: (-r["games"], r["value"]))
    return rows or [{"value": v, "label": v, "games": 0} for v in FALLBACK_CLOCKS]


# --- a game -----------------------------------------------------------------------------

def board_view(seat: str) -> list[int]:
    """Display cell -> board index, turning the board so `seat` sits at the bottom."""
    k = TURN_ORDER.index(seat)
    view = [0] * 64
    for r in range(8):
        for c in range(8):
            dr, dc = rotate_square(r, c, k)
            view[dr * 8 + dc] = r * 8 + c
    return view


def _move_squares(uci: str | None) -> list[int] | None:
    """'e2e4' -> [board index of e2, board index of e4], or None."""
    if not uci or len(uci) < 4:
        return None
    a, b = square_to_index(uci[:2]), square_to_index(uci[2:4])
    return [a[0] * 8 + a[1], b[0] * 8 + b[1]]


class Session:
    """One timed game with a person or a bot in each seat.

    `seats` maps each colour to None (a person) or a bot reference. Seats can change during
    the game (set_seat) and the game can be paused (set_paused). All game and clock changes
    happen under `self.lock`. A bot searches without the lock and throws its search away
    when the position, its seat, or the pause changes under it.
    """

    def __init__(self, bank: EngineBank, seats: dict, time_control: str = "1|5",
                 save_dir: Path | None = SAVE_DIR, max_sims_per_move: int | None = None,
                 verbose: bool = False, armed: bool | None = None, human_clock: bool = True):
        self.bank = bank
        self.seats = {c: seats.get(c) for c in TURN_ORDER}
        self.human_clock = human_clock
        self.save_dir = save_dir
        self.max_sims = max_sims_per_move
        self.verbose = verbose
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.closed = False
        self.paused = False
        self._names: dict[str, str] = {}
        self.clock = Clock.parse(time_control)
        self._sync_clock()
        # with people playing, the clocks wait until the page is open
        self.armed = (not self.humans()) if armed is None else armed
        self._searchers: dict[str, MCTS] = {}   # engine thread only
        self.version = 0
        self.root: Node | None = None
        self.root_owner: str | None = None
        self.engine_info: dict[str, dict] = {}
        self.flags: list[str] = []
        self.saved: str | None = None
        self.last: tuple[int, int] | None = None
        self.game = Game(metadata={
            "time_control": self.clock.label,
            "seats": self._seat_metadata(),
            "human_clock": human_clock,
            "source": "gui.play",
            "started": datetime.now().isoformat(timespec="seconds"),
        })
        if self.armed:
            self.clock.start(self.game.state.current_player())
        self._threads = [threading.Thread(target=self._engine_loop, daemon=True),
                         threading.Thread(target=self._ticker, daemon=True)]
        for t in self._threads:
            t.start()

    # --- seats and clocks -----------------------------------------------------------

    def humans(self) -> set[str]:
        return {c for c in TURN_ORDER if self.seats[c] is None}

    def bots(self) -> dict[str, str]:
        return {c: self.seats[c] for c in TURN_ORDER if self.seats[c] is not None}

    def _seat_name(self, c: str) -> str:
        if self.seats[c] is None:
            return "You" if len(self.humans()) == 1 else f"Player ({_NAMES[c]})"
        ref = self.seats[c]
        if ref not in self._names:
            self._names[ref] = self.bank.name(ref)
        return self._names[ref]

    def _seat_metadata(self) -> dict:
        return {_NAMES[c]: {"name": self._seat_name(c), "human": self.seats[c] is None,
                            "bot": self.seats[c]} for c in TURN_ORDER}

    def _sync_clock(self) -> None:
        """People have no clock when the clock option is off; bots always do."""
        self.clock.untimed = set() if self.human_clock else set(self.humans())

    def set_seat(self, color: str, value) -> str | None:
        """Change who plays `color` (a person, or a bot). Takes effect at once: a bot that
        was thinking for this side gives up its search."""
        if color not in TURN_ORDER:
            return "unknown side"
        try:
            ref = parse_seat_value(value, color)
        except ValueError as e:
            return str(e)
        with self.lock:
            if self.over():
                return "the game is over"
            if self.seats[color] == ref:
                return None
            self.seats[color] = ref
            self.root = None
            self.version += 1          # stops a search that was running for this side
            self._sync_clock()
            self.game.metadata["seats"] = self._seat_metadata()
            self.changed.notify_all()
            return None

    def set_paused(self, paused: bool) -> str | None:
        """Pause or resume: the clocks stop and bots stop thinking, then pick up where they were."""
        with self.lock:
            if self.over():
                return "the game is over"
            if paused == self.paused:
                return None
            self.paused = paused
            self.version += 1          # stops any bot search in progress
            if paused:
                self.clock.stop()
            elif self.armed:
                self.clock.start(self.game.state.current_player())
            self.changed.notify_all()
            return None

    # --- game state -----------------------------------------------------------------

    def arm(self) -> None:
        """Start the clocks (the first time the page asks for the game)."""
        with self.lock:
            if not self.armed:
                self.armed = True
                if not self.over() and not self.paused:
                    self.clock.start(self.game.state.current_player())
                self.changed.notify_all()

    def over(self) -> bool:
        return is_game_over(self.game.state)

    def _after_change(self) -> None:
        """Call under the lock after any move or elimination."""
        self.version += 1
        if self.over():
            self.clock.stop()
            self._save()
        elif self.armed and not self.paused:
            self.clock.start(self.game.state.current_player())
        self.changed.notify_all()

    def _play(self, color: str, move: Move, extra: dict | None = None) -> None:
        used = self.clock.press(color)
        self.game.play(move)
        entry = self.game.history[-1]
        entry.update({"think": round(used, 2), **(extra or {})})
        if color not in self.clock.untimed:
            entry["clock"] = round(self.clock.remaining[color], 2)
        self.last = (move.start[0] * 8 + move.start[1], move.end[0] * 8 + move.end[1])
        self.root = MCTS.advance(self.root, move)
        if self.verbose:
            print(f"{entry['turn']:>4} {entry.get('san', ''):<12} {used:5.2f}s used"
                  + (f", {extra['sims']} sims" if extra else ""), flush=True)
        self._after_change()

    def human_move(self, color: str, start: int, end: int) -> str | None:
        """Returns an error message, or None if the move was played."""
        with self.lock:
            if self.over():
                return "the game is over"
            if self.paused:
                return "the game is paused"
            if color not in self.humans() or color != self.game.state.current_player():
                return "not your turn"
            if self.clock.flagged():
                self._tick()
                return "out of time"
            uci = index_to_square(start // 8, start % 8) + index_to_square(end // 8, end % 8)
            try:
                move = Move.from_uci(uci, self.game.state.board)
            except (ValueError, KeyError, IndexError):
                return "illegal move"
            if move not in legal_moves(self.game.state):
                return "illegal move"
            self._play(color, move)
            return None

    def resign(self, color: str) -> str | None:
        with self.lock:
            if self.over():
                return "the game is over"
            if color not in self.humans() or color in self.game.state.eliminated:
                return "only a person still in the game can resign"
            if color == self.clock.running:
                self.clock.stop()
            self.game.eliminate(color, "resign")
            self.root = None
            self._after_change()
            return None

    def _tick(self) -> None:
        """Eliminate the side to move if its time has run out (under the lock)."""
        color = self.clock.flagged()
        if color is None or self.over():
            return
        self.clock.stop()
        self.clock.remaining[color] = 0.0
        self.game.eliminate(color, "timeout")
        self.flags.append(color)
        self.root = None
        if self.verbose:
            print(f"     {_NAMES[color]} lost on time", flush=True)
        self._after_change()

    def _ticker(self) -> None:
        while not self.closed:
            with self.lock:
                self._tick()
            time.sleep(0.05)

    def _save(self) -> None:
        if self.save_dir is None or self.saved:
            return
        self.game.metadata["clock_left"] = self.clock.snapshot()
        self.game.metadata["flagged"] = list(self.flags)
        self.game.metadata["seats"] = self._seat_metadata()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        self.game.save(path)
        self.saved = str(path)
        if self.verbose:
            print(f"saved {path}", flush=True)

    # --- bots -----------------------------------------------------------------------

    def _searcher(self, ref: str) -> MCTS:
        if ref not in self._searchers:
            self._searchers[ref] = MCTS(self.bank.get(ref), num_simulations=self.max_sims,
                                        max_turn=None)
        return self._searchers[ref]

    def _engine_loop(self) -> None:
        while not self.closed:
            job = None
            with self.lock:
                if self.armed and not self.paused and not self.over():
                    color = self.game.state.current_player()
                    ref = self.seats.get(color)
                    if ref is not None:
                        reuse = self.root is not None and self.root_owner == ref
                        job = {"color": color, "ref": ref, "version": self.version,
                               "state": self.game.state, "root": self.root if reuse else None,
                               "budget": move_budget(self.clock.left(color), self.clock.increment)}
                if job is None:
                    self.changed.wait(0.5)
                    continue
            # loading a model can take a while: keep it outside the lock
            mcts = self._searcher(job["ref"])
            root = job["root"] or Node(job["state"])
            started = time.monotonic()
            mcts.search(root, time_limit=job["budget"],
                        stop=lambda: self.closed or self.paused or self.version != job["version"])
            with self.lock:
                if self.closed or self.paused or self.version != job["version"] or self.over():
                    continue  # the position, the seats or the pause changed while thinking
                i = mcts.choose(root)
                n = root.child_n[i]
                q = root.child_w[i] / n if n else root.value
                self.engine_info[job["color"]] = {
                    "bot": self._seat_name(job["color"]),
                    "sims": MCTS.real_visits(root), "new_sims": mcts.last_simulations,
                    "seconds": round(time.monotonic() - started, 2),
                    "budget": round(job["budget"], 2),
                    "values": [round(float(v), 3) for v in q]}
                self.root, self.root_owner = root, job["ref"]
                if isinstance(root.moves[i], Resign):
                    self._engine_resign(job["color"], {"sims": int(mcts.last_simulations)})
                else:
                    self._play(job["color"], root.moves[i], {"sims": int(mcts.last_simulations)})

    def _engine_resign(self, color: str, extra: dict | None = None) -> None:
        """A bot resigns for `color` (under the lock): charged like a move, logged as
        "<Color> resigned", and the seat is eliminated as for a person."""
        used = self.clock.press(color)
        self.game.eliminate(color, "resign")
        entry = self.game.history[-1]
        entry.update({"think": round(used, 2), **(extra or {})})
        if color not in self.clock.untimed:
            entry["clock"] = round(self.clock.remaining[color], 2)
        self.root = None
        if self.verbose:
            print(f"{entry['turn']:>4} {_caption(entry):<12} {used:5.2f}s used", flush=True)
        self._after_change()

    # --- for the page ---------------------------------------------------------------

    def _clocks_after(self, ply: int) -> dict[str, float | None]:
        """Clocks as they were recorded after `ply` moves (used when reviewing)."""
        clocks = {c: (None if c in self.clock.untimed else self.clock.remaining.get(c))
                  for c in TURN_ORDER}
        for h in self.game.history[:ply]:
            if h.get("clock") is not None:
                clocks[h["color"]] = h["clock"]
        return clocks

    def snapshot(self, ply: int | None = None) -> dict:
        """The page's view of the game. `ply` (0 .. number of moves) shows an earlier position
        for review; None, or the latest ply, is the live game."""
        with self.lock:
            total = len(self.game.history)
            reviewing = ply is not None and ply < total
            if reviewing:
                shown = max(0, ply)
                s = self.game.states()[shown]
                turn, legal, running = None, [], None
                last = _move_squares(self.game.history[shown - 1].get("uci")) if shown else None
                clocks = self._clocks_after(shown)
                over = False
            else:
                shown = total
                s = self.game.state
                over = self.over()
                turn = None if over else s.current_player()
                legal = []
                if turn in self.humans() and not self.paused:
                    legal = [[m.start[0] * 8 + m.start[1], m.end[0] * 8 + m.end[1]]
                             for m in legal_moves(s)]
                last = list(self.last) if self.last else None
                clocks = self.clock.snapshot()
                running = self.clock.running
            seats = self._seat_metadata()
            return {
                "b": "".join(p or ".." for row in s.board for p in row),
                "s": [s.scores[c] for c in TURN_ORDER],
                "e": "".join(sorted(s.eliminated)),
                "t": turn,
                "clock": clocks, "running": running,
                "tc": self.clock.label, "human_clock": self.human_clock,
                "paused": self.paused,
                "seats": [{**seats[_NAMES[c]], "color": c} for c in TURN_ORDER],
                "humans": sorted(self.humans(), key=TURN_ORDER.index),
                "legal": legal, "last": last,
                "moves": [{"n": i + 1, "color": h["color"], "san": h.get("san") or "",
                           "event": h.get("event"), "points": h.get("points", 0),
                           "clock": h.get("clock")}
                          for i, h in enumerate(self.game.history)],
                "ply": total, "view": shown if reviewing else None,
                "engine": self.engine_info,
                "result": self.game.result() if over else None,
                "saved": self.saved, "flags": self.flags,
            }

    def wait_until_over(self, poll: float = 0.2) -> dict:
        while True:
            with self.lock:
                if self.over():
                    return self.game.result()
            time.sleep(poll)

    def close(self) -> None:
        self.closed = True
        with self.lock:
            self.changed.notify_all()


class App:
    """The server's state: the bot bank, the sprites, and the one current game."""

    def __init__(self, bank: EngineBank, assets: dict, max_sims: int | None = None,
                 verbose: bool = False):
        self.bank = bank
        self.assets = assets
        self.max_sims = max_sims
        self.verbose = verbose
        self._lock = threading.Lock()
        self.session: Session | None = None

    def start(self, seats: dict, time_control: str, armed: bool = True,
              human_clock: bool = True) -> Session:
        Clock.parse(time_control)  # raises ValueError for a bad time control
        with self._lock:
            if self.session is not None:
                self.session.close()
            self.session = Session(self.bank, seats, time_control,
                                   max_sims_per_move=self.max_sims, verbose=self.verbose,
                                   armed=armed, human_clock=human_clock)
            return self.session

    def current(self) -> Session | None:
        with self._lock:
            return self.session

    def stop(self) -> None:
        with self._lock:
            if self.session is not None:
                self.session.close()
                self.session = None


def builtin_sprites() -> dict[str, str]:
    """Simple built-in pieces (no download): a coloured disc with the piece letter."""
    return {o + p: fallback_svg(p, c) for o, c in SPRITE_COLORS.items() for p in PIECE_TEMPLATES}


def make_assets(sprites: str = "chesscom") -> dict:
    if sprites == "builtin":
        pieces, source = builtin_sprites(), "built-in"
    else:
        pieces, source = sprite_set()
    return {"sprites": pieces, "sprite_source": source,
            "views": {c: board_view(c) for c in TURN_ORDER}, "clocks": scraped_time_controls()}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html")
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                if name in STATIC_FILES:
                    return self._send(200, (STATIC_DIR / name).read_bytes(), STATIC_FILES[name])
                return self._json(404, {"error": "not found"})
            if path == "/api/assets":
                return self._json(200, app.assets)
            if path == "/api/bots":
                return self._json(200, list_bots())
            if path == "/api/state":
                session = app.current()
                if session is None:
                    return self._json(200, {"idle": True})
                session.arm()
                ply = parse_qs(url.query).get("ply", [""])[0]
                try:
                    ply = int(ply) if ply else None
                except ValueError:
                    return self._json(400, {"error": "ply must be a number"})
                return self._json(200, session.snapshot(ply))
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            if self.path == "/api/new":
                try:
                    seats = parse_seats(body.get("seats") or {})
                    app.start(seats, str(body.get("time") or "1|5"), armed=True,
                              human_clock=bool(body.get("human_clock", True)))
                except (ValueError, TypeError, AttributeError) as e:
                    return self._json(400, {"error": str(e) or "bad game setup"})
                return self._json(200, {"ok": True})
            if self.path == "/api/close":
                app.stop()
                return self._json(200, {"ok": True})
            session = app.current()
            if session is None:
                return self._json(409, {"error": "no game is running"})
            if self.path == "/api/move":
                try:
                    err = session.human_move(str(body["color"]), int(body["from"]), int(body["to"]))
                except (KeyError, TypeError, ValueError):
                    err = "bad move request"
            elif self.path == "/api/resign":
                err = session.resign(str(body.get("color")))
            elif self.path == "/api/seat":
                err = session.set_seat(str(body.get("color")), body.get("value"))
            elif self.path == "/api/pause":
                err = session.set_paused(bool(body.get("paused")))
            else:
                return self._json(404, {"error": "not found"})
            self._json(200, {"error": err} if err else {"ok": True})

    return Handler


def parse_seat_args(items: list[str], base: dict) -> dict:
    seats = dict(base)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--seat takes COLOR=VALUE, got {item!r}")
        c, v = item.split("=", 1)
        seats[c.strip().upper()[:1]] = v.strip()
    return seats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="+", default=["R"],
                    help="seat(s) played by people: R B Y G, or 'none' (default R)")
    ap.add_argument("--seat", nargs="+", default=[], metavar="COLOR=VALUE",
                    help="set each side: human, uniform, or a model (overrides --human and --engine)")
    ap.add_argument("--engine", default=None,
                    help="bot for the other seats: models/<label>, a model.pt, a pool file, or 'uniform' "
                         "(default: the benchmark, else the best model in models/index.jsonl)")
    ap.add_argument("--time", default="1|5", help='time control "minutes|increment" (default "1|5")')
    ap.add_argument("--no-human-clock", action="store_true", help="people play without a clock")
    ap.add_argument("--max-sims", type=int, default=None, help="optional cap on simulations per move")
    ap.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for the bots")
    ap.add_argument("--sprites", choices=["chesscom", "builtin"], default="chesscom",
                    help="piece artwork: chess.com's (downloaded once, cached) or built-in")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)

    if [h.lower() for h in args.human] == ["none"]:
        humans = set()
    else:
        humans = {h.upper()[:1] for h in args.human}
    if not humans <= set(TURN_ORDER):
        ap.error("--human takes R, B, Y, G or none")
    try:
        Clock.parse(args.time)
    except ValueError:
        ap.error(f"bad time control {args.time!r}; use minutes|increment, e.g. 1|5")
    bot = args.engine or default_bot()
    base = {c: (HUMAN if c in humans else bot) for c in TURN_ORDER}
    try:
        seats = parse_seats(parse_seat_args(args.seat, base))
    except ValueError as e:
        ap.error(str(e))
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    bank = EngineBank()
    assets = make_assets(args.sprites)
    app = App(bank, assets, max_sims=args.max_sims, verbose=not humans)
    # with people at the screen, the clocks start when the page opens (see /api/state)
    session = app.start(seats, args.time, armed=not humans, human_clock=not args.no_human_clock)
    names = ", ".join(f"{_NAMES[c]}={session._seat_name(c) if seats[c] else 'human'}" for c in TURN_ORDER)
    print(f"seats: {names}   time control: {args.time}")

    if not humans and args.no_open:  # bots only, in the terminal
        try:
            res = session.wait_until_over()
        finally:
            app.stop()
        print(json.dumps({**res, "flagged": session.flags, "clock_left": session.clock.snapshot(),
                          "saved": session.saved}))
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"playing at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
