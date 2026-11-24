"""max^n search over the NNUE (Python prototype, phase 0 of logs/nnue_alphabeta_plan.md).

Every player picks the move that maximizes their own utility: their expected chess.com FFA
score (mean of P(finish above each opponent), ties half), as the NNUE estimates it at the
leaves. Nobody is assumed to be on a team, so ganging up on a "prey", breaking off an
attack or turning on a partner happen exactly when the utilities say they pay.

  - iterative deepening (depth 1, 2, ...) inside a time budget, or to a fixed depth
  - quiescence on point-scoring captures at the leaves (the mover may also stand pat)
  - sibling leaves are evaluated together in one batched network call (the compiled core
    reads each weight once per batch, which large networks need)
  - a transposition table (position, side to move, scores, eliminated seats)
  - move ordering: the table's best move, then captures by points gained
  - immediate pruning only: a utility of 1 (finishing above all three) can't be beaten.
    max^n allows little else; the compiled core is what will make depth affordable.

Utilities are in absolute seat order (R, B, Y, G) and always sum to 2.

    python -m bot.maxn models/<nnue> --opponent models/<mlp> --games 40 --time-ms 250
    python -m bot.maxn calibrate models/<mlp>        # ms per MCTS simulation on this machine
    python -m bot.maxn merge logs/maxn/*.jsonl       # combine match shards
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chaturaji.constants import DEAD, PIECE_POINTS, TURN_ORDER
from chaturaji.game import apply_move, is_game_over
from chaturaji.moves import Move, legal_moves
from chaturaji.state import ChaturajiState, board_key

from .nnue import Accumulator, NumpyNNUE

_SEAT = {c: i for i, c in enumerate(TURN_ORDER)}


class _Timeout(Exception):
    pass


def exact_utilities(state: ChaturajiState) -> np.ndarray:
    """Final FFA utilities from the scores: per seat, mean over opponents of 1 / 0.5 / 0."""
    s = np.array([state.scores[c] for c in TURN_ORDER], dtype=np.float64)
    res = (s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])
    return (res.sum(1) - 0.5) / 3.0                    # drop the self-comparison (0.5)


def capture_points(board, move: Move) -> int:
    victim = board[move.end[0]][move.end[1]]
    if victim is None:
        return -1                                       # not a capture
    if victim[0] == DEAD:
        return 3 if victim[1] == "K" else 0
    return PIECE_POINTS[victim[1]]


class MaxN:
    def __init__(self, net: NumpyNNUE, time_ms: float | None = 250, depth: int | None = None,
                 max_depth: int = 12, qdepth: int = 2, tt_size: int = 200_000):
        if time_ms is None and depth is None:
            raise ValueError("give a time budget or a fixed depth")
        self.net, self.time_ms, self.depth = net, time_ms, depth
        self.max_depth, self.qdepth, self.tt_size = max_depth, qdepth, tt_size
        self.tt: dict = {}
        self.stats: dict = {}

    # --- public ------------------------------------------------------------------------

    def best_move(self, state: ChaturajiState) -> Move:
        moves = legal_moves(state)
        if len(moves) == 1:
            self.stats = {"depth": 0, "nodes": 0, "forced": True}
            return moves[0]
        self.deadline = (time.perf_counter() + self.time_ms / 1000.0) if self.time_ms else None
        self.nodes = 0
        if len(self.tt) > self.tt_size:
            self.tt.clear()
        acc = Accumulator(self.net, state.board)
        best, value, reached = moves[0], None, 0
        for d in range(1, (self.depth or self.max_depth) + 1):
            try:
                v, m = self._search(state, d, acc, root=True)
            except _Timeout:
                break
            best, value, reached = m, v, d
            if v[_SEAT[state.current_player()]] >= 1.0 - 1e-9:
                break                                   # already finishes above everyone
        self.stats = {"depth": reached, "nodes": self.nodes,
                      "value": None if value is None else np.round(value, 3).tolist()}
        return best

    def __call__(self, state: ChaturajiState) -> Move:
        return self.best_move(state)

    # --- search ------------------------------------------------------------------------

    def _tick(self):
        self.nodes += 1
        if self.deadline is not None and (self.nodes & 63) == 0 and time.perf_counter() > self.deadline:
            raise _Timeout

    @staticmethod
    def _key(state: ChaturajiState):
        return (board_key(state.board), state.turn % 4, tuple(state.scores[c] for c in TURN_ORDER),
                tuple(sorted(state.eliminated)))

    def _ordered(self, state: ChaturajiState, tt_move: Move | None) -> list[Move]:
        moves = legal_moves(state)
        board = state.board
        keyed = sorted(moves, key=lambda m: -capture_points(board, m))
        if tt_move is not None and tt_move in keyed:
            keyed.remove(tt_move)
            keyed.insert(0, tt_move)
        return keyed

    def _search(self, state: ChaturajiState, depth: int, acc: Accumulator, root: bool = False):
        self._tick()
        if is_game_over(state):
            return exact_utilities(state), None
        if depth == 0:
            return self._quiesce(state, self.qdepth, acc), None
        key = self._key(state)
        hit = self.tt.get(key)
        if hit is not None and hit[0] >= depth and not root:
            return hit[1], hit[2]
        me = _SEAT[state.current_player()]
        best_v, best_m = None, None
        moves = self._ordered(state, hit[2] if hit else None)
        children = [apply_move(state, m)[0] for m in moves]
        stands = {}
        if depth == 1:                    # every child is a leaf: one batched evaluation
            live = [i for i, c in enumerate(children) if not is_game_over(c)]
            self.nodes += len(live)
            for i, v in zip(live, acc.evaluate_children([children[i] for i in live])):
                stands[i] = v
        for i, (m, child) in enumerate(zip(moves, children)):
            if depth == 1:
                self._tick()
                if is_game_over(child):
                    v = exact_utilities(child)
                else:
                    acc.push(child.board)
                    try:
                        v = self._quiesce(child, self.qdepth, acc, stands[i])
                    finally:
                        acc.pop()
            else:
                acc.push(child.board)
                try:
                    v, _ = self._search(child, depth - 1, acc)
                finally:
                    acc.pop()
            if best_v is None or v[me] > best_v[me]:
                best_v, best_m = v, m
                if v[me] >= 1.0 - 1e-9:
                    break                               # immediate pruning
        self.tt[key] = (depth, best_v, best_m)
        return best_v, best_m

    def _quiesce(self, state: ChaturajiState, q: int, acc: Accumulator,
                 stand: np.ndarray | None = None) -> np.ndarray:
        """Stand pat or play a point-scoring capture; `stand` is this position's evaluation
        when the caller already has it (from a batched sibling evaluation)."""
        if stand is None:
            stand = acc.evaluate(state)
        if q == 0:
            return stand
        me = _SEAT[state.current_player()]
        board = state.board
        best = stand
        caps = [m for m in legal_moves(state) if capture_points(board, m) > 0]
        if not caps:
            return best
        caps.sort(key=lambda m: -capture_points(board, m))
        children = [apply_move(state, m)[0] for m in caps]
        live = [i for i, c in enumerate(children) if not is_game_over(c)]
        evals = dict(zip(live, acc.evaluate_children([children[i] for i in live])))
        self.nodes += len(live)
        for i, child in enumerate(children):
            self._tick()
            if is_game_over(child):
                v = exact_utilities(child)
            elif q == 1:
                v = evals[i]
            else:
                acc.push(child.board)
                try:
                    v = self._quiesce(child, q - 1, acc, evals[i])
                finally:
                    acc.pop()
            if v[me] > best[me]:
                best = v
        return best


def load_maxn(path: str | Path, **kw) -> MaxN:
    """A MaxN player from models/<label> (its nnue.npz, or model.pt converted)."""
    p = Path(path)
    if p.is_dir() and (p / "nnue.npz").is_file():
        return MaxN(NumpyNNUE.load(p / "nnue.npz"), **kw)
    from . import registry
    model, kind, _, _ = registry.load_model(p)
    if kind != "nnue":
        raise ValueError(f"{p} is a {kind} model, not nnue")
    return MaxN(NumpyNNUE.from_model(model), **kw)


def calibrate(opponent: str, sims: int = 64, positions: int = 12) -> float:
    """Milliseconds per MCTS simulation for this model on this machine."""
    from .evaluate import load_evaluator
    from .mcts import MCTS, Node
    ev = load_evaluator(opponent)
    mcts = MCTS(ev, num_simulations=sims, max_turn=None)
    rng = np.random.default_rng(0)
    s = ChaturajiState.initial()
    states = []
    while len(states) < positions and not is_game_over(s):
        ms = legal_moves(s)
        s, _ = apply_move(s, ms[rng.integers(len(ms))])
        if s.turn % 3 == 0:
            states.append(s)
    mcts.search(Node(states[0]))                       # warm up
    t = time.perf_counter()
    for st in states:
        mcts.search(Node(st))
    return 1000 * (time.perf_counter() - t) / (len(states) * sims)


def merge(paths: list[str]) -> dict:
    """Combine match shards (the --out .jsonl files) into one result."""
    recs = [json.loads(l) for p in paths for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    pair = np.array([r["pairwise"] for r in recs])
    seats = {c: round(float(np.mean([r["pairwise"] for r in recs if r["seat"] == c])), 3)
             for c in TURN_ORDER if any(r["seat"] == c for r in recs)}
    res = [v for r in recs for v in r["results"].values()]
    return {"games": len(recs), "pairwise": round(float(pair.mean()), 3),
            "pairwise_se": round(float(pair.std(ddof=1) / np.sqrt(len(pair))), 3) if len(pair) > 1 else None,
            "per_seat_pairwise": seats,
            "wins_draws_losses": [res.count(1.0), res.count(0.5), res.count(0.0)],
            "endings": {k: sum(r["reason"] == k for r in recs) for k in sorted({r["reason"] for r in recs})},
            "mean_plies": round(float(np.mean([r["plies"] for r in recs])), 1)}


def main(argv=None) -> None:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["calibrate"]:
        print(f"{calibrate(argv[1]):.3f}")
        return
    if argv[:1] == ["merge"]:
        print(json.dumps(merge(argv[1:])))
        return
    from .evaluate import evaluate, load_evaluator, model_name
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="models/<nnue label>")
    ap.add_argument("--opponent", default="random", help='"random" or an MCTS model (e.g. an MLP)')
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--time-ms", type=float, default=250)
    ap.add_argument("--depth", type=int, default=None, help="fixed depth instead of a time budget")
    ap.add_argument("--qdepth", type=int, default=2)
    ap.add_argument("--sims", type=int, default=32, help="opponents' MCTS simulations per move")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the game records here (.jsonl)")
    args = ap.parse_args(argv)
    player = load_maxn(args.model, time_ms=None if args.depth else args.time_ms,
                       depth=args.depth, qdepth=args.qdepth)
    depths = []

    def play(state):
        m = player.best_move(state)
        depths.append(player.stats.get("depth", 0))
        return m

    opp = None if args.opponent == "random" else load_evaluator(args.opponent)
    t0 = time.time()
    res = evaluate(None, games=args.games, sims=args.sims, opponent=opp,
                   opponent_name="random" if opp is None else model_name(args.opponent),
                   max_plies=args.max_plies, seed=args.seed, opening_plies=args.opening_plies,
                   record=bool(args.out), bot_player=play)
    records = res.pop("records", [])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"type": "maxn_match", **r}) + "\n")
    d = np.array(depths)
    res.update({"model": model_name(args.model), "maxn_time_ms": None if args.depth else args.time_ms,
                "maxn_depth": args.depth, "qdepth": args.qdepth,
                "depth_reached": {"mean": round(float(d[d > 0].mean()), 2) if (d > 0).any() else 0,
                                  "counts": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}},
                "minutes": round((time.time() - t0) / 60, 1)})
    for k in ("game_pairwise", "game_shares"):
        res.pop(k, None)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
