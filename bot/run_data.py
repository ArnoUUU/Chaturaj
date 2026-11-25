"""Load one training run's records as pandas tables, and export them as CSV files.

    python -m bot.run_data <label | logs/<label> | models/<label>> [--out DIR] [--root DIR]

A run is read from models/<label>/meta.json and logs/<label>/ (games.jsonl, steps.csv); with
a label it is looked up under the repo root (--root overrides it, and when the label is not
found there, the main checkout of this git repository is tried). Every table always exists,
empty when the run has no data for it, so old runs (no per-move records, no steps.csv) load.

Tables (load(run) -> dict of DataFrames; export writes each one as <name>.csv):

  iterations  meta["history"], one row per iteration. Nested dicts become dotted columns:
              endings.last_king, search.mean_depth, resigns.claims, ...
  checks      meta["checks"], one row per check: check, iteration, vs_best.pairwise,
              vs_best.pairwise_se, vs_mcts.pairwise, vs_mcts.maxn_seconds_per_move, decision, ...
  games       games.jsonl, one row per line. game_id = the line number (1-based).
              Columns: game_id, type, iteration, check, vs, seed (text), opening, seat (letter,
              match games), plies, reason, score_R, score_B, score_Y, score_G, winners ("R",
              or "R+B" for a tie at the top), pairwise, n_resigns, first_elim_ply (the ply of
              the first entry of "eliminations").
  moves       one row per ply, for the games with per-move records (self-play and match games
              have movers / kinds / values; evaluate records have move_info). Columns: game_id,
              type, iteration, check, ply (0-based), seat (letter), move (UCI, e.g. "d8e7";
              "resign" for RESIGN = 65535; evaluate records keep their own strings), kind
              (opening / explore / search / forced / resign), by (opening / bot / opponent /
              random), value_R, value_B, value_Y, value_G (root utilities, absolute seats),
              value_mover, depth, nodes, secs, sims (MCTS simulations), q_R .. q_G (the chosen
              child's mean value). Missing values are NaN.
  steps       logs/<label>/steps.csv: iteration, step (1..n within the iteration),
              global_step, loss, grad_norm, lr.
  resigns     one row per resignation in a game's "resigns" list: game_id, type, iteration,
              check, ply, seat (letter), active (players still in), score_R .. score_G
              (scores before resigning).

Seats are written as letters (R, B, Y, G) throughout; seat index 0..3 in the records is
RBYG order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLORS = "RBYG"
RESIGN = 65535
TABLES = ("iterations", "checks", "games", "moves", "steps", "resigns")

GAME_COLS = ["game_id", "type", "iteration", "check", "vs", "seed", "opening", "seat", "plies", "reason",
             *[f"score_{c}" for c in COLORS], "winners", "pairwise", "n_resigns", "first_elim_ply"]
MOVE_COLS = ["game_id", "type", "iteration", "check", "ply", "seat", "move", "kind", "by",
             *[f"value_{c}" for c in COLORS], "value_mover", "depth", "nodes", "secs", "sims",
             *[f"q_{c}" for c in COLORS]]
RESIGN_COLS = ["game_id", "type", "iteration", "check", "ply", "seat", "active",
               *[f"score_{c}" for c in COLORS]]
STEP_COLS = ["iteration", "step", "global_step", "loss", "grad_norm", "lr"]
NUMERIC = {
    "games": ["game_id", "iteration", "opening", "plies", *[f"score_{c}" for c in COLORS], "pairwise",
              "n_resigns", "first_elim_ply"],
    "moves": ["game_id", "iteration", "ply", *[f"value_{c}" for c in COLORS], "value_mover", "depth",
              "nodes", "secs", "sims", *[f"q_{c}" for c in COLORS]],
    "resigns": ["game_id", "iteration", "ply", "active", *[f"score_{c}" for c in COLORS]],
}


# --- locating a run ---------------------------------------------------------------------

def main_checkout() -> Path | None:
    """The main working tree of this git repository (the first entry of `git worktree list`)."""
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("worktree "):
            return Path(line[len("worktree "):].strip())
    return None


def locate(run, root=None) -> tuple[Path, Path, str]:
    """(models dir, logs dir, label) for a label, or for a path to logs/<label> or models/<label>."""
    p = Path(str(run))
    if p.is_dir() and p.parent.name in ("logs", "models"):
        label = p.name
        if p.parent.name == "logs":
            base = p.parent.parent
            return base / "logs" / "models" / label, p, label
        base = p.parent.parent
        return p, base / "logs" / label, label
    label = str(run).strip("/\\")
    bases = [Path(root) if root is not None else ROOT]
    if root is None:                      # the main checkout too: runs live there, not in worktrees
        main = main_checkout()
        if main is not None and main not in bases:
            bases.append(main)

    def pick(kind: str) -> Path:
        for base in bases:
            if (base / kind / label).exists():
                return base / kind / label
        return bases[0] / kind / label
    return pick("logs/models"), pick("logs"), label


# --- reading ----------------------------------------------------------------------------

def _read_meta(models: Path) -> dict:
    path = models / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_games(logs: Path) -> list[tuple[int, dict]]:
    path = logs / "games.jsonl"
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append((lineno, json.loads(line)))
            except json.JSONDecodeError:
                warnings.warn(f"{path}: line {lineno} is not valid JSON, skipped")
    return rows


def _scores(r: dict) -> list:
    s = r.get("scores")
    if isinstance(s, dict):
        return [s.get(c) for c in COLORS]
    if isinstance(s, list) and len(s) == 4:
        return list(s)
    return [None] * 4


def _winners(sc: list) -> str | None:
    if any(x is None for x in sc):
        return None
    top = max(sc)
    return "+".join(c for c, x in zip(COLORS, sc) if x == top)


def _seat(x) -> str | None:
    if x is None:
        return None
    return x if isinstance(x, str) else COLORS[int(x)]


def _sq(i: int) -> str:
    row, col = divmod(int(i), 8)           # row 0 = rank 8, col 0 = file a
    return f"{chr(ord('a') + col)}{8 - row}"


def uci(code) -> str:
    """UCI text of a move code (from * 64 + to; square = row * 8 + col); RESIGN -> "resign"."""
    code = int(code)
    if code == RESIGN:
        return "resign"
    s, e = divmod(code, 64)
    return _sq(s) + _sq(e)


def _at(lst, i):
    return lst[i] if lst is not None and i < len(lst) else None


def _game_row(gid: int, r: dict) -> dict:
    sc = _scores(r)
    moves = r.get("moves") or []
    if "resigns" in r:
        n_res = len(r["resigns"])
    else:
        n_res = sum(1 for m in moves if m == RESIGN or m == "resign")
    el = r.get("eliminations") or []
    row = {"game_id": gid, "type": r.get("type"), "iteration": r.get("iteration"), "check": r.get("check"),
           "vs": r.get("vs"), "seed": str(r["seed"]) if "seed" in r else None,
           "opening": r.get("opening", r.get("opening_plies")), "seat": r.get("seat"), "plies": r.get("plies"),
           "reason": r.get("reason"), "winners": _winners(sc), "pairwise": r.get("pairwise"),
           "n_resigns": n_res, "first_elim_ply": el[0][0] if el else None}
    for c, x in zip(COLORS, sc):
        row[f"score_{c}"] = x
    return row


def _move_rows(gid: int, r: dict) -> list[dict]:
    if "movers" not in r and "move_info" not in r:
        return []
    moves = r.get("moves") or []
    movers, kinds, values = r.get("movers"), r.get("kinds"), r.get("values")
    depths, nodes, secs = r.get("depths"), r.get("nodes"), r.get("secs")
    info = r.get("move_info")
    rows = []
    for p, mv in enumerate(moves):
        mi = _at(info, p) or {}
        if movers is not None and p < len(movers):
            seat = COLORS[int(movers[p])]
        else:
            seat = mi.get("seat") or COLORS[p % 4]
        vals = _at(values, p) if values is not None else mi.get("value")
        row = {"game_id": gid, "type": r.get("type"), "iteration": r.get("iteration"), "check": r.get("check"),
               "ply": p, "seat": seat, "move": mv if isinstance(mv, str) else uci(mv),
               "kind": _at(kinds, p), "by": mi.get("by"),
               "value_mover": vals[COLORS.index(seat)] if vals else None,
               "depth": _at(depths, p) if depths is not None else mi.get("depth"),
               "nodes": _at(nodes, p) if nodes is not None else mi.get("nodes"),
               "secs": _at(secs, p) if secs is not None else mi.get("seconds"),
               "sims": mi.get("sims")}
        for i, c in enumerate(COLORS):
            row[f"value_{c}"] = vals[i] if vals else None
            row[f"q_{c}"] = mi["q"][i] if mi.get("q") else None
        rows.append(row)
    return rows


def _resign_rows(gid: int, r: dict) -> list[dict]:
    rows = []
    for rs in r.get("resigns") or []:
        row = {"game_id": gid, "type": r.get("type"), "iteration": r.get("iteration"), "check": r.get("check"),
               "ply": rs.get("ply"), "seat": _seat(rs.get("seat")), "active": rs.get("active")}
        for i, c in enumerate(COLORS):
            row[f"score_{c}"] = rs["scores"][i] if rs.get("scores") else None
        rows.append(row)
    return rows


def _frame(rows: list[dict], cols: list[str], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=cols)
    for c in NUMERIC.get(name, []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load(models: Path, logs: Path) -> dict[str, pd.DataFrame]:
    meta = _read_meta(models)
    history = pd.json_normalize(meta.get("history") or [])
    checks = pd.json_normalize(meta.get("checks") or [])
    games, moves, resigns = [], [], []
    for gid, r in _read_games(logs):
        games.append(_game_row(gid, r))
        moves.extend(_move_rows(gid, r))
        resigns.extend(_resign_rows(gid, r))
    steps_path = logs / "steps.csv"
    if steps_path.is_file() and steps_path.stat().st_size > 0:
        steps = pd.read_csv(steps_path)
    else:
        steps = pd.DataFrame(columns=STEP_COLS)
    return {"iterations": history, "checks": checks, "games": _frame(games, GAME_COLS, "games"),
            "moves": _frame(moves, MOVE_COLS, "moves"), "steps": steps,
            "resigns": _frame(resigns, RESIGN_COLS, "resigns")}


def load(run, root=None) -> dict[str, pd.DataFrame]:
    """The tables of a run (see the module docstring). `run` is a label, or a path to
    logs/<label> or models/<label>; `root` is the repo root to look in for a label."""
    models, logs, _ = locate(run, root)
    return _load(models, logs)


# --- writing ----------------------------------------------------------------------------

def default_out(run, root=None) -> Path:
    _, logs, _ = locate(run, root)
    return logs / "tables"


def write(tables: dict[str, pd.DataFrame], out_dir) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths


def export(run, out_dir=None, root=None) -> dict[str, Path]:
    """Write one CSV per table to out_dir (default logs/<label>/tables/). Returns the paths."""
    tables = load(run, root)
    return write(tables, out_dir if out_dir is not None else default_out(run, root))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run label, or a path to logs/<label> or models/<label>")
    ap.add_argument("--out", default=None, help="directory for the CSV files (default logs/<label>/tables)")
    ap.add_argument("--root", default=None, help="repo root to look for the label in (default: this repo)")
    args = ap.parse_args(argv)
    tables = load(args.run, args.root)
    paths = write(tables, args.out if args.out is not None else default_out(args.run, args.root))
    for name, df in tables.items():
        print(f"{name:<11} {len(df):>7} rows x {df.shape[1]:>3} columns  {paths[name]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
