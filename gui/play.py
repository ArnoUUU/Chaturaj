"""Play a timed game against the engine in your browser (bot/PLAN.md milestone M7).

    python -m gui.play                          # you are Red, the engine plays the rest, 1 | 5
    python -m gui.play --human B                # play Blue (the board turns so Blue is at the bottom)
    python -m gui.play --human R Y --time "3|2" # two people at one screen, engine plays Blue and Green
    python -m gui.play --engine models/<label>  # pick the model (default: best evaluated in models/)
    python -m gui.play --human none --no-open   # engine against itself on the clock, in the terminal

Starts a small local web server (standard library only, 127.0.0.1) and opens the page in
your browser. Click a piece, then a highlighted square, to move. Clocks work like
chess.com: each seat has its own clock, the increment is added after each move, and a
seat whose clock runs out is eliminated on time (its pieces turn grey, like a
resignation). The engine plays every seat that is not --human, choosing how long to think
from its own clock (bot/timing.py). Finished games are saved to games/played/ and open in
the replay viewer: python -m gui.viewer games/played

Everything runs on this machine. Nothing connects to chess.com except the one-time
download of the piece sprites (gui/sprites.py), which falls back to built-in sprites.
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

import numpy as np

from bot.features import rotate_square
from bot.mcts import MCTS, Node
from bot.timing import Clock, move_budget
from chaturaji.constants import TURN_ORDER
from chaturaji.game import is_game_over
from chaturaji.moves import Move, legal_moves
from chaturaji.record import Game
from chaturaji.state import index_to_square

from .sprites import sprite_set
from .viewer import _NAMES, _caption

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "games" / "played"


def default_model() -> Path | None:
    """The saved model with the best recorded score share in models/index.jsonl."""
    index = ROOT / "models" / "index.jsonl"
    best, best_share = None, -1.0
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            share = (row.get("evaluation") or {}).get("score_share") or 0.0
            path = ROOT / "models" / row["label"]
            if share > best_share and (path / "model.pt").exists():
                best, best_share = path, share
    return best


def board_view(seat: str) -> list[int]:
    """Display cell -> board index, turning the board so `seat` sits at the bottom."""
    k = TURN_ORDER.index(seat)
    view = [0] * 64
    for r in range(8):
        for c in range(8):
            dr, dc = rotate_square(r, c, k)
            view[dr * 8 + dc] = r * 8 + c
    return view


class Session:
    """One timed game: the record, the clocks, and the engine thread.

    All game and clock changes happen under `self.lock`. The engine searches without
    the lock and aborts when the position changes under it (a resignation, a flag).
    """

    def __init__(self, evaluator, humans: set[str], time_control: str = "1|5",
                 engine_name: str = "engine", save_dir: Path | None = SAVE_DIR,
                 max_sims_per_move: int | None = None, verbose: bool = False):
        self.evaluator = evaluator
        self.humans = set(humans)
        self.time_control = time_control
        self.engine_name = engine_name
        self.save_dir = save_dir
        self.max_sims = max_sims_per_move
        self.verbose = verbose
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.closed = False
        self.armed = not self.humans  # with people playing, clocks wait for the page to open
        self.new_game()
        self._threads = [threading.Thread(target=self._engine_loop, daemon=True),
                         threading.Thread(target=self._ticker, daemon=True)]
        for t in self._threads:
            t.start()

    # --- game state -------------------------------------------------------------------

    def new_game(self) -> None:
        with self.lock:
            self.clock = Clock.parse(self.time_control)
            seats = {_NAMES[c]: {"name": "You" if c in self.humans else "Engine",
                                 "human": c in self.humans} for c in TURN_ORDER}
            if len(self.humans) > 1:
                for c in self.humans:
                    seats[_NAMES[c]]["name"] = f"Player ({_NAMES[c]})"
            self.game = Game(metadata={"time_control": self.clock.label, "seats": seats,
                                       "engine": self.engine_name, "source": "gui.play",
                                       "started": datetime.now().isoformat(timespec="seconds")})
            self.version = getattr(self, "version", 0) + 1
            self.root: Node | None = None
            self.engine_info: dict[str, dict] = {}
            self.flags: list[str] = []
            self.saved: str | None = None
            self.last: tuple[int, int] | None = None
            if self.armed:
                self.clock.start(self.game.state.current_player())
            self.changed.notify_all()

    def arm(self) -> None:
        """Start the clocks (the first time the page asks for the game)."""
        with self.lock:
            if not self.armed:
                self.armed = True
                if not self.over():
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
        else:
            self.clock.start(self.game.state.current_player())
        self.changed.notify_all()

    def _play(self, color: str, move: Move, extra: dict | None = None) -> None:
        used = self.clock.press(color)
        self.game.play(move)
        self.game.history[-1].update({"clock": round(self.clock.remaining[color], 2),
                                      "think": round(used, 2), **(extra or {})})
        self.last = (move.start[0] * 8 + move.start[1], move.end[0] * 8 + move.end[1])
        self.root = MCTS.advance(self.root, move)
        if self.verbose:
            h = self.game.history[-1]
            print(f"{h['turn']:>4} {_caption(h):<28} {used:5.2f}s used, "
                  f"{self.clock.remaining[color]:6.2f}s left"
                  + (f", {extra['sims']} sims" if extra else ""), flush=True)
        self._after_change()

    def human_move(self, color: str, start: int, end: int) -> str | None:
        """Returns an error message, or None if the move was played."""
        with self.lock:
            if self.over():
                return "the game is over"
            if color != self.game.state.current_player() or color not in self.humans:
                return "not your turn"
            if self.clock.flagged():
                self._tick()
                return "out of time"
            uci = index_to_square(start // 8, start % 8) + index_to_square(end // 8, end % 8)
            move = Move.from_uci(uci, self.game.state.board)
            if move not in legal_moves(self.game.state):
                return "illegal move"
            self._play(color, move)
            return None

    def resign(self, color: str) -> str | None:
        with self.lock:
            if self.over():
                return "the game is over"
            if color not in self.humans or color in self.game.state.eliminated:
                return "you can only resign a human seat that is still playing"
            if color == self.clock.running:
                self.clock.stop()
            self.game.eliminate(color, "resign")
            self.root = None
            self._after_change()
            return None

    def _tick(self) -> None:
        """Eliminate the player to move if their flag fell (under the lock)."""
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
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        self.game.save(path)
        self.saved = str(path)
        if self.verbose:
            print(f"saved {path}", flush=True)

    # --- engine -----------------------------------------------------------------------

    def _engine_loop(self) -> None:
        mcts = MCTS(self.evaluator, num_simulations=self.max_sims, max_turn=None)
        while not self.closed:
            with self.lock:
                state = self.game.state
                color = state.current_player()
                if self.over() or color in self.humans or not self.armed:
                    self.changed.wait(0.5)
                    continue
                version = self.version
                left = self.clock.left(color)
                budget = move_budget(left, self.clock.increment)
                root = self.root if self.root is not None else Node(state)
            started = time.monotonic()
            mcts.search(root, time_limit=budget,
                        stop=lambda: self.closed or self.version != version)
            with self.lock:
                if self.version != version or self.over():
                    continue  # position changed while thinking (flag, resignation, new game)
                i = mcts.choose(root)
                n = root.child_n[i]
                q = root.child_w[i] / n if n else root.value
                self.engine_info[color] = {
                    "sims": int(root.child_n.sum()), "new_sims": mcts.last_simulations,
                    "seconds": round(time.monotonic() - started, 2), "budget": round(budget, 2),
                    "values": [round(float(v), 3) for v in q]}
                self.root = root
                self._play(color, root.moves[i], {"sims": int(mcts.last_simulations)})

    # --- for the page -----------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            s = self.game.state
            turn = None if self.over() else s.current_player()
            legal = []
            if turn in self.humans:
                legal = [[m.start[0] * 8 + m.start[1], m.end[0] * 8 + m.end[1]]
                         for m in legal_moves(s)]
            res = self.game.result() if self.over() else None
            return {
                "b": "".join(p or ".." for row in s.board for p in row),
                "s": [s.scores[c] for c in TURN_ORDER],
                "e": "".join(sorted(s.eliminated)),
                "t": turn,
                "clock": self.clock.snapshot(), "running": self.clock.running,
                "tc": self.clock.label, "engine_name": self.engine_name,
                "legal": legal, "last": self.last,
                "humans": sorted(self.humans, key=TURN_ORDER.index),
                "names": [self.game.metadata["seats"][_NAMES[c]]["name"] for c in TURN_ORDER],
                "moves": [_caption(h) for h in self.game.history[-12:]],
                "ply": len(self.game.history),
                "engine": self.engine_info,
                "result": res, "saved": self.saved, "flags": self.flags,
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


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Chaturaji — play</title><style>
body{margin:0;background:#262421;color:#ddd;font:14px system-ui,sans-serif;display:flex;justify-content:center}
main{padding:16px;display:flex;gap:18px;flex-wrap:wrap;justify-content:center}
#left{display:flex;flex-direction:column;align-items:center;gap:10px}
#board{display:grid;grid-template-columns:repeat(8,64px);grid-template-rows:repeat(8,64px);border:4px solid #403d39}
.sq{width:64px;height:64px;background-size:90%;background-position:center;background-repeat:no-repeat;position:relative;cursor:pointer}
.l{background-color:#ebecd0}.d{background-color:#739552}
.sq.last{box-shadow:inset 0 0 0 64px rgba(255,255,80,.35)}.sq.sel{box-shadow:inset 0 0 0 64px rgba(20,85,30,.5)}
.sq.dot::after{content:"";position:absolute;left:24px;top:24px;width:16px;height:16px;border-radius:50%;background:rgba(0,0,0,.25)}
.sq.cap::after{content:"";position:absolute;inset:3px;border-radius:50%;border:5px solid rgba(0,0,0,.25)}
#pts{display:flex;gap:8px}.p{min-width:110px;padding:6px 10px;border-radius:6px;color:#fff;text-align:center;font-weight:600}
.p span{display:block;font-size:22px}.p .ck{font:600 18px ui-monospace,monospace;background:rgba(0,0,0,.35);border-radius:4px;margin-top:4px;padding:2px}
.p.out{opacity:.35}.p.turn{outline:3px solid #fff}.ck.low{color:#ffb4b4}
#side{width:260px;display:flex;flex-direction:column;gap:10px}
#side div.box{background:#302e2b;border-radius:6px;padding:10px}
button{background:#403d39;color:#ddd;border:0;border-radius:4px;padding:6px 10px;font-size:14px;cursor:pointer}
#moves div{padding:1px 0}#status{font-weight:600;min-height:20px}.muted{color:#999;font-size:12px}
</style></head><body><main>
<div id="left"><div id="pts"></div><div id="board"></div><div id="status"></div></div>
<div id="side">
 <div class="box"><div id="tc"></div><div class="muted">Click a piece, then a highlighted square.</div></div>
 <div class="box"><button id="resign">Resign</button> <button id="new">New game</button></div>
 <div class="box"><b>Engine</b><div id="eng" class="muted"></div></div>
 <div class="box"><b>Moves</b><div id="moves"></div></div>
</div></main><script>
const SPRITES=__SPRITES__, VIEW=__VIEW__;
const COLORS={R:"#BF3B43",B:"#4185BF",Y:"#C09526",G:"#4E9161"}, ORDER="RBYG", NAMES={R:"Red",B:"Blue",Y:"Yellow",G:"Green"};
const URL_={}; for(const k in SPRITES) URL_[k]="data:image/svg+xml;base64,"+btoa(SPRITES[k]);
const $=id=>document.getElementById(id), sqs=[];
for(let d=0;d<64;d++){const e=document.createElement("div");e.className="sq "+(((d>>3)+(d&7))%2?"d":"l");
  e.onclick=()=>click(VIEW[d]);$("board").appendChild(e);sqs.push(e);}
let S=null, got=0, sel=null, busy=false;
function fmt(t){t=Math.max(0,t);if(t<10)return t.toFixed(1);const m=Math.floor(t/60),s=Math.floor(t%60);return m+":"+String(s).padStart(2,"0");}
function left(c){let t=S.clock[c];if(S.running===c)t-=(performance.now()-got)/1000;return t;}
function render(){if(!S)return;
  const targets=sel===null?[]:S.legal.filter(m=>m[0]===sel).map(m=>m[1]);
  for(let d=0;d<64;d++){const i=VIEW[d],p=S.b.substr(i*2,2),e=sqs[d];
    e.style.backgroundImage=p===".."?"":`url(${URL_[p]})`;
    e.classList.toggle("last",!!S.last&&(S.last[0]===i||S.last[1]===i));
    e.classList.toggle("sel",sel===i);
    e.classList.toggle("dot",targets.includes(i)&&p==="..");e.classList.toggle("cap",targets.includes(i)&&p!=="..");}
  $("pts").innerHTML=[...ORDER].map((c,i)=>{const t=left(c);return `<div class="p${S.e.includes(c)?" out":""}${S.t===c?" turn":""}" style="background:${COLORS[c]}">${S.names[i]}<span>${S.s[i]}</span><div class="ck${t<10?" low":""}">${fmt(t)}</div></div>`;}).join("");
  $("tc").textContent="Time control "+S.tc+(S.humans.length?" · you: "+S.humans.map(c=>NAMES[c]).join(", "):"");
  let st="";
  if(S.result){const w=S.result.winners.map(c=>NAMES[c]).join(" & ");st=`Game over (${S.result.reason}). Winner: ${w}.`+(S.saved?" Saved.":"");}
  else if(S.humans.includes(S.t))st=`Your move (${NAMES[S.t]})`;else st=`${NAMES[S.t]} (engine) is thinking…`;
  $("status").textContent=st;
  $("moves").innerHTML=S.moves.slice().reverse().map(m=>`<div>${m}</div>`).join("");
  $("eng").innerHTML=`<div style="word-break:break-all">${S.engine_name}</div>`+(Object.entries(S.engine).map(([c,e])=>`${NAMES[c]}: ${e.sims} sims in ${e.seconds}s · top-score chance `+e.values.map((v,i)=>`<span style="color:${COLORS[ORDER[i]]}">${Math.round(v*100)}%</span>`).join(" ")).join("<br>")||"–");
  $("resign").disabled=!!S.result||!S.humans.some(c=>!S.e.includes(c));}
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});return r.json();}
async function poll(){try{const s=await api("/api/state");if(!S||s.ply!==S.ply||s.running!==S.running||!!s.result!==!!S.result)sel=null;S=s;got=performance.now();}catch(e){$("status").textContent="Lost connection to the game server.";}}
async function click(i){if(!S||busy||!S.humans.includes(S.t))return;
  if(sel!==null&&S.legal.some(m=>m[0]===sel&&m[1]===i)){busy=true;const r=await api("/api/move",{color:S.t,from:sel,to:i});busy=false;sel=null;
    if(r.error)$("status").textContent=r.error;await poll();render();return;}
  sel=S.legal.some(m=>m[0]===i)?i:null;render();}
$("resign").onclick=async()=>{const c=S.humans.find(c=>!S.e.includes(c)&&(S.t===c||S.humans.length===1))||S.humans.find(c=>!S.e.includes(c));
  if(c&&confirm("Resign "+NAMES[c]+"?")){await api("/api/resign",{color:c});await poll();render();}};
$("new").onclick=async()=>{if(S.result||confirm("Abandon this game and start a new one?")){await api("/api/new",{});await poll();render();}};
setInterval(poll,250);setInterval(render,100);poll();
</script></body></html>"""


def make_page(session: Session, seat: str | None = None) -> str:
    sprites, _ = sprite_set()
    seat = seat or (sorted(session.humans, key=TURN_ORDER.index) or ["R"])[0]
    return PAGE.replace("__SPRITES__", json.dumps(sprites, separators=(",", ":"))) \
               .replace("__VIEW__", json.dumps(board_view(seat)))


def make_handler(session: Session, page: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html")
            elif self.path == "/api/state":
                session.arm()
                self._send(200, json.dumps(session.snapshot()))
            else:
                self._send(404, '{"error": "not found"}')

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, '{"error": "bad json"}')
            if self.path == "/api/move":
                try:
                    err = session.human_move(str(body["color"]), int(body["from"]), int(body["to"]))
                except (KeyError, TypeError, ValueError):
                    err = "bad move request"
            elif self.path == "/api/resign":
                err = session.resign(str(body.get("color")))
            elif self.path == "/api/new":
                session.new_game()
                err = None
            else:
                return self._send(404, '{"error": "not found"}')
            self._send(200, json.dumps({"error": err} if err else {"ok": True}))

    return Handler


def load_engine(model: str | None):
    """(evaluator, name). No model -> uniform priors and a neutral value."""
    from bot.evaluate import load_evaluator, model_name
    path = model or default_model()
    if path is None or str(path) == "uniform":
        from bot.mcts import UniformEvaluator
        return UniformEvaluator(), "uniform"
    return load_evaluator(path), model_name(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="+", default=["R"],
                    help="seat(s) played by people: R B Y G, or 'none' for engine only (default R)")
    ap.add_argument("--time", default="1|5", help='time control "minutes|increment" (default "1|5")')
    ap.add_argument("--engine", default=None,
                    help="models/<label>, a model.pt, a snapshot, or 'uniform' (default: best in models/index.jsonl)")
    ap.add_argument("--max-sims", type=int, default=None, help="optional cap on simulations per move")
    ap.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for the engine")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)

    humans = set() if [h.lower() for h in args.human] == ["none"] else {h.upper()[0] for h in args.human}
    if not humans <= set(TURN_ORDER):
        ap.error("--human takes R, B, Y, G or none")
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    evaluator, name = load_engine(args.engine)
    print(f"engine: {name}   time control: {args.time}   humans: {''.join(sorted(humans)) or 'none'}")
    session = Session(evaluator, humans, args.time, name, max_sims_per_move=args.max_sims,
                      verbose=not humans)

    if not humans and args.no_open:  # engine-only game in the terminal
        try:
            res = session.wait_until_over()
        finally:
            session.close()
        print(json.dumps({**res, "flagged": session.flags, "clock_left": session.clock.snapshot(),
                          "saved": session.saved}))
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session, make_page(session)))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"playing at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


if __name__ == "__main__":
    main()
