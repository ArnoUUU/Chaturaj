"""Worker processes for NNUE self-play (bot/selfplay_nnue.py).

Kept light on purpose: NumPy and the compiled core only, no PyTorch, so many workers fit in
memory. Each game runs entirely in the compiled core (rules + max^n search), and returns the
positions as raw board codes; the main process turns them into network inputs.

The games against MCTS + MLP (the yardstick) need PyTorch; `yardstick_block` imports it
lazily, and runs in a separate small pool.
"""

from __future__ import annotations

import numpy as np

try:
    import chaturaji_engine as E
except ImportError:      # the trainer checks for this before starting workers
    E = None

_NETS: dict = {}
_SEARCHERS: dict = {}


def load_core_net(path: str):
    """A chaturaji_engine.Nnue from an nnue.npz weight file (cached per path)."""
    if path not in _NETS:
        with np.load(path) as z:
            w = {k: z[k] for k in z.files}
        n = sum(1 for k in w if k[0] == "l" and k[1:-2].isdigit() and k.endswith("_w"))
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
        res = [bool(x) for x in w["residual"]] if "residual" in w else [False] * n
        _NETS.clear()          # one network per worker at a time (weights change every iteration)
        _SEARCHERS.clear()
        _NETS[path] = E.Nnue(f32(w["ft_w"]), f32(w["ft_b"]), [f32(w[f"l{i}_w"]) for i in range(1, n + 1)],
                             [f32(w[f"l{i}_b"]) for i in range(1, n + 1)], res, f32(w["out_w"]), f32(w["out_b"]))
    return _NETS[path]


PREFS = ("discount", "elim_bonus", "elim_horizon")


def prefs(task: dict) -> dict:
    """The search preferences in a task (defaults: off)."""
    return {k: task[k] for k in PREFS if k in task}


def searcher(path: str, depth: int, qdepth: int, check_ext: int, slot: int = 0, width: int = 0,
             full_plies: int = 2, resign: bool = True, **pref):
    key = (path, depth, qdepth, check_ext, width, full_plies, resign, tuple(sorted(pref.items())), slot)
    if key not in _SEARCHERS:
        net = _NETS.get(path) or load_core_net(path)
        _SEARCHERS[key] = E.MaxN(net, depth=depth, qdepth=qdepth, check_ext=check_ext, width=width,
                                 full_plies=full_plies, tt_size=1_000_000, resign=resign, **pref)
    return _SEARCHERS[key]


def _pairwise(scores, seat):
    res = [1.0 if scores[seat] > scores[c] else 0.5 if scores[seat] == scores[c] else 0.0
           for c in range(4) if c != seat]
    return sum(res) / 3, res


def selfplay_game(task: dict) -> dict:
    """One self-play game: every seat is max^n with the same network.

    Exploration: a random opening of `open_lo`..`open_hi` plies, then a random move with
    probability `explore` during the first `explore_plies` plies. Every `stride`-th position
    is returned (board codes, mover, the 16 globals), with the final scores.
    """
    rng = np.random.default_rng(task["seed"])
    s = searcher(task["weights"], task["depth"], task["qdepth"], task["check_ext"], 0,
                 task.get("width", 0), task.get("full_plies", 2), resign=task.get("resign", True),
                 **prefs(task))
    s.clear()
    pos = E.Position.initial()
    n_open = int(rng.integers(task["open_lo"], task["open_hi"] + 1))
    boards, ks, gl, moves, captures, checks, elims = [], [], [], [], [], [], []
    ply = 0
    while pos.game_over is None and pos.turn < task["max_turn"]:
        ms = pos.legal_moves()
        if ply % task["stride"] == 0:
            k = pos.current_player
            boards.append(np.asarray(pos.board))
            ks.append(k)
            gl.append(np.asarray(pos.globals(k)))
        if ply < n_open or (ply < task["explore_plies"] and rng.random() < task["explore"]):
            m = ms[int(rng.integers(len(ms)))]
        else:
            m = s.best_move(pos)[0]
        if m == E.RESIGN:                          # the mover resigns: no move, no checks or captures
            pos.eliminate(pos.current_player)
            moves.append(int(m))
            ply += 1
            continue
        mover = pos.current_player
        if pos.gives_check(m):
            checks.append((ply, mover))
        points, cap, bonus, elim = pos.push(m)
        moves.append(int(m))
        if cap and (cap - 1) // 5 < 4:             # an active piece: (ply, attacker, victim, points)
            captures.append((ply, mover, (cap - 1) // 5, points))
        if elim is not None:                       # (ply, eliminator, eliminated)
            elims.append((ply, mover, int(elim)))
        ply += 1
    return {"seed": task["seed"], "boards": np.stack(boards).astype(np.uint8), "ks": np.array(ks, dtype=np.int8),
            "globals": np.stack(gl).astype(np.float32), "scores": list(pos.scores),
            "reason": pos.game_over or "ply_cap", "plies": ply, "opening": n_open, "moves": moves,
            "captures": captures, "checks": checks, "eliminations": elims}


def match_game(task: dict) -> dict:
    """Candidate network in one seat, opponent network in the other three (each seat its own
    search), after a seeded random opening; no exploration."""
    seat = task["seat"]
    w, fp = task.get("width", 0), task.get("full_plies", 2)
    rs = task.get("resign", True)
    cand = searcher(task["cand"], task["depth"], task["qdepth"], task["check_ext"], 9, w, fp, resign=rs,
                    **prefs(task))
    opps = {c: searcher(task["opp"], task["depth"], task["qdepth"], task["check_ext"], c, w, fp, resign=rs,
                        **prefs(task))
            for c in range(4) if c != seat}
    for x in (cand, *opps.values()):
        x.clear()
    rng = np.random.default_rng(task["opening_seed"])
    pos = E.Position.initial()
    ply = 0
    while pos.game_over is None and pos.turn < task["max_turn"]:
        if ply < task["opening_plies"]:
            ms = pos.legal_moves()
            m = ms[int(rng.integers(len(ms)))]
        else:
            m = (cand if pos.current_player == seat else opps[pos.current_player]).best_move(pos)[0]
        if m == E.RESIGN:
            pos.eliminate(pos.current_player)
        else:
            pos.push(m)
        ply += 1
    pw, res = _pairwise(pos.scores, seat)
    return {"seat": "RBYG"[seat], "scores": list(pos.scores), "pairwise": pw, "results": res,
            "reason": pos.game_over or "ply_cap", "plies": ply}


def yardstick_block(task: dict) -> dict:
    """One block of 4 games (same opening, max^n in each seat in turn) against MCTS + MLP."""
    import torch
    torch.set_num_threads(1)
    from .evaluate import evaluate, load_evaluator
    from .maxn import MaxN
    from .nnue import NumpyNNUE
    player = MaxN(NumpyNNUE.load(task["weights"]), time_ms=None, depth=task["depth"], qdepth=task["qdepth"],
                  check_ext=task["check_ext"], width=task.get("width", 0),
                  full_plies=task.get("full_plies", 2), resign=task.get("resign", True), **prefs(task))
    times = []

    def play(state):
        import time
        t = time.perf_counter()
        m = player.best_move(state)
        times.append(time.perf_counter() - t)
        return m

    res = evaluate(None, games=4, sims=task["sims"], opponent=load_evaluator(task["mlp"]), opponent_name="mlp",
                   opening_plies=task["opening_plies"], seed=task["seed"], max_plies=task["max_plies"],
                   record=True, bot_player=play)
    return {"records": res["records"], "maxn_seconds": float(np.mean(times)) if times else 0.0}
