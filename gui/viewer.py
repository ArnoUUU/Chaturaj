"""Minimal game viewer: board, chess.com piece sprites in seat colours, and points.

    python -m gui.viewer scraper/data/records/113028841.json      # one game
    python -m gui.viewer scraper/data/records                     # a folder (newest 50 games)
    python -m gui.viewer scraper/data/records --limit 400

Builds one self-contained HTML page (gui/.cache/viewer.html) and opens it in your default
browser. Nothing else is needed: no server and no packages. Controls: Left/Right arrows,
Home/End, the buttons, and a game picker when several games are loaded.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from chaturaji.constants import TURN_ORDER
from chaturaji.record import Game

from .sprites import sprite_set

OUT = Path(__file__).resolve().parent / ".cache" / "viewer.html"


_NAMES = {"R": "Red", "B": "Blue", "Y": "Yellow", "G": "Green"}


def _caption(h: dict) -> str:
    if h.get("uci") is None:
        return f"{_NAMES[h['color']]} {'timed out' if h.get('event') == 'timeout' else 'resigned'}"
    pts = f"  (+{h['points']})" if h.get("points") else ""
    return f"{_NAMES[h['color']]}: {h['san']}{pts}"


def game_frames(game: Game) -> dict:
    """Compact frames: a 128-char board string (2 chars per square, '..' empty), scores,
    player to move, eliminated players, and the move that led to the frame."""
    frames = []
    for i, s in enumerate(game.states()):
        h = game.history[i - 1] if i else None
        frames.append({
            "b": "".join(p or ".." for row in s.board for p in row),
            "s": [s.scores[c] for c in TURN_ORDER],
            "t": s.current_player(),
            "e": "".join(sorted(s.eliminated)),
            "m": _caption(h) if h else "start",
        })
    seats = game.metadata.get("seats") or {}
    names = [(seats.get(n) or {}).get("name") or n for n in ("Red", "Blue", "Yellow", "Green")]
    gid = game.metadata.get("chesscom_id") or ""
    return {"title": f"{gid} {game.metadata.get('time_control') or ''}".strip() or "game",
            "names": names, "frames": frames}


def load_games(path: Path, limit: int) -> list[dict]:
    files = [path] if path.is_file() else sorted(path.glob("*.json"), key=lambda p: p.name, reverse=True)
    games = []
    for f in files[:limit]:
        try:
            games.append(game_frames(Game.load(f)))
        except Exception as e:
            print(f"skip {f.name}: {e}")
    return games


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Chaturaji viewer</title><style>
body{margin:0;background:#262421;color:#ddd;font:14px system-ui,sans-serif;display:flex;justify-content:center}
main{padding:16px;display:flex;flex-direction:column;align-items:center;gap:10px}
#board{display:grid;grid-template-columns:repeat(8,64px);grid-template-rows:repeat(8,64px);border:4px solid #403d39}
.sq{width:64px;height:64px;background-size:90%;background-position:center;background-repeat:no-repeat}
.l{background-color:#ebecd0}.d{background-color:#739552}
#pts{display:flex;gap:8px}.p{min-width:96px;padding:6px 10px;border-radius:6px;color:#fff;text-align:center;font-weight:600}
.p span{display:block;font-size:22px}.p.out{opacity:.35}.p.turn{outline:3px solid #fff}
button,select{background:#403d39;color:#ddd;border:0;border-radius:4px;padding:6px 10px;font-size:14px}
#info{min-height:20px}
</style></head><body><main>
<select id="game"></select>
<div id="pts"></div>
<div id="board"></div>
<div><button id="first">|&lt;</button> <button id="prev">&lt;</button>
<span id="ply"></span> <button id="next">&gt;</button> <button id="last">&gt;|</button></div>
<div id="info"></div>
</main><script>
const GAMES=__GAMES__, SPRITES=__SPRITES__;
const COLORS={R:"#BF3B43",B:"#4185BF",Y:"#C09526",G:"#4E9161"}, ORDER="RBYG";
const URL_={}; for(const k in SPRITES) URL_[k]="data:image/svg+xml;base64,"+btoa(SPRITES[k]);
const $=id=>document.getElementById(id), board=$("board"), sqs=[];
for(let i=0;i<64;i++){const d=document.createElement("div");d.className="sq "+(((i>>3)+(i&7))%2?"d":"l");board.appendChild(d);sqs.push(d);}
GAMES.forEach((g,i)=>{const o=document.createElement("option");o.value=i;o.textContent=g.title+" — "+g.names.join(", ");$("game").appendChild(o);});
$("game").style.display=GAMES.length>1?"":"none";
let g=0,f=0;
function show(){const G=GAMES[g],F=G.frames[f];
  for(let i=0;i<64;i++){const p=F.b.substr(i*2,2);sqs[i].style.backgroundImage=p===".."?"":`url(${URL_[p]})`;}
  $("pts").innerHTML=[...ORDER].map((c,i)=>`<div class="p${F.e.includes(c)?" out":""}${F.t===c?" turn":""}" style="background:${COLORS[c]}">${G.names[i]}<span>${F.s[i]}</span></div>`).join("");
  $("ply").textContent=`${f} / ${G.frames.length-1}`; $("info").textContent=F.m;}
function go(n){f=Math.max(0,Math.min(GAMES[g].frames.length-1,n));show();}
$("first").onclick=()=>go(0);$("prev").onclick=()=>go(f-1);$("next").onclick=()=>go(f+1);
$("last").onclick=()=>go(1e9);$("game").onchange=e=>{g=+e.target.value;go(0);};
document.onkeydown=e=>{const k={ArrowLeft:f-1,ArrowRight:f+1,Home:0,End:1e9}[e.key];if(k!==undefined){e.preventDefault();go(k);}};
show();
</script></body></html>"""


def build(path: Path, limit: int = 50, out: Path = OUT) -> Path:
    games = load_games(path, limit)
    if not games:
        raise SystemExit(f"no games found at {path}")
    sprites, source = sprite_set()
    html = PAGE.replace("__GAMES__", json.dumps(games, separators=(",", ":"))) \
               .replace("__SPRITES__", json.dumps(sprites, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{len(games)} game(s), sprites from {source} -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a game record .json, or a folder of them")
    ap.add_argument("--limit", type=int, default=50, help="max games when given a folder")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)
    out = build(Path(args.path), args.limit)
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
