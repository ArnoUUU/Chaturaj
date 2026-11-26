"""Greedy bots: a lightweight hand-written eval under paranoid alpha-beta search.

No network. Each bot scores a line from its own seat: points gained, minus (weighted) material
lost, plus a few nudges, and every opponent is assumed to play against it (paranoid search).
Depth is counted in plies (one player's move); 4 plies = 1 round. Rounds = turn // 4.

Eval (the bot is `me`; every term is added along the search line, so a node's value is
"value to go" and the transposition table stays valid whatever path reached it):
  + points I score (captures, check bonuses, end bonuses)
  + king_bonus (4) for capturing an enemy king in rounds 0-9 (on top of the 3 points)
  - loss_weight (1.2) x value of each of my pieces captured
  - pawn_loss (2) extra for losing a pawn in rounds 0-14
  - endgame_loss (1) extra per piece lost once the endgame flag is set (any pawn within 2
    squares of promotion, or a king captured after round 10); the flag is sticky
  - losing my king: (30 - round) x active kings on the board, until round 30 (king_floor extra)
  + trade_bonus per trade (optional): I capture, then within one round I lose a piece worth
    no more than what I took
  - threat penalty at the leaves (optional): my attacked pieces, weighted by how soon each
    attacker moves before my next turn (0.9 / 0.6 / 0.4 for the 1st / 2nd / 3rd player), net of
    my defenders; if I am to move I can save the biggest one
  game over: placement by points, +150 / -20 / -40 / -100 (ties share)
  two players left: the mover may claim the win (resign), tried at every such node, so
  opponents are assumed to claim immediately.

Three bots:
  greedy        the full eval with the threat penalty
  greedy-trade  the same plus +1 per trade
  naive         shallow, no threat penalty, big early trade bonus, pushes pawns after round 15

    python -m bot.greedy time --depths 1 2 3 4 5 6       # seconds per move by depth
    python -m bot.greedy match greedy naive uniform greedy --games 4 --depth 3
    python -m bot.greedy match greedy mlp:models/<label> uniform uniform --games 2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

from chaturaji.constants import (BISHOP_DIRS, DEAD, KING_OFFSETS, KNIGHT_OFFSETS, PAWN_DIRECTION,
                                 PIECE_POINTS, ROOK_DIRS, TURN_ORDER)
from chaturaji.game import apply_action, apply_move, eliminate, is_game_over, require_moves
from chaturaji.moves import RESIGN, Move, Resign, _pawn_capture_offsets, legal_moves
from chaturaji.state import ChaturajiState

PLACEMENT = (150.0, -20.0, -40.0, -100.0)       # 1st .. 4th
SEAT = {c: i for i, c in enumerate(TURN_ORDER)}
THREAT_P = (0.9, 0.6, 0.4)                      # the 1st, 2nd, 3rd mover before my next turn
INF = 1e9


@dataclass(frozen=True)
class Params:
    loss_weight: float = 1.2
    king_bonus: float = 4.0
    king_bonus_rounds: int = 10
    pawn_loss: float = 2.0
    pawn_loss_rounds: int = 15
    endgame_loss: float = 1.0
    endgame_pawn_dist: int = 2
    king_loss_rounds: int = 30
    king_floor: float = 0.0
    trade_bonus: float = 0.0
    trade_rounds: int = 1000                    # trade bonus applies only before this round
    threat: bool = True
    push_bonus: float = 0.0                     # per pawn step after push_from round
    push_from: int = 15
    depth: int = 4
    qdepth: int = 4


PRESETS = {
    "greedy": Params(),
    "greedy-trade": Params(trade_bonus=1.0),
    "naive": Params(loss_weight=1.0, king_bonus=0.0, pawn_loss=0.0, endgame_loss=0.0,
                    trade_bonus=3.0, trade_rounds=15, threat=False, push_bonus=0.6, depth=2, qdepth=0),
}


class _Timeout(Exception):
    pass


def placement_value(scores: dict, me: str) -> float:
    mine = scores[me]
    better = sum(1 for c in TURN_ORDER if c != me and scores[c] > mine)
    tied = sum(1 for c in TURN_ORDER if c != me and scores[c] == mine)
    return sum(PLACEMENT[better:better + tied + 1]) / (tied + 1)


def _pawn_to_go(color: str, r: int, c: int) -> int:
    dr, dc = PAWN_DIRECTION[color]
    return (7 - r if dr > 0 else r) if dr else (7 - c if dc > 0 else c)


def endgame_now(board) -> bool:
    """Any active pawn within 2 squares of promotion (initial flag at the search root)."""
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p and p[0] != DEAD and p[1] == "P" and _pawn_to_go(p[0], r, c) <= 2:
                return True
    return False


# --- attacks (lowest-value attacker of a square, per colour) -----------------------------

_AV = {"P": 1, "N": 3, "K": 3, "B": 5, "R": 5}


def attackers(board, r: int, c: int, color: str) -> int:
    """Value of the cheapest piece of `color` attacking (r, c); 0 if none."""
    best = 99
    for odr, odc in _pawn_capture_offsets(color):
        tr, tc = r - odr, c - odc
        if 0 <= tr < 8 and 0 <= tc < 8 and board[tr][tc] == color + "P":
            return 1
    for dr, dc in KNIGHT_OFFSETS:
        tr, tc = r + dr, c + dc
        if 0 <= tr < 8 and 0 <= tc < 8 and board[tr][tc] == color + "N":
            best = 3
    for dr, dc in KING_OFFSETS:
        tr, tc = r + dr, c + dc
        if 0 <= tr < 8 and 0 <= tc < 8 and board[tr][tc] == color + "K":
            best = 3
    for dirs, pt in ((ROOK_DIRS, "R"), (BISHOP_DIRS, "B")):
        for dr, dc in dirs:
            tr, tc = r + dr, c + dc
            while 0 <= tr < 8 and 0 <= tc < 8:
                t = board[tr][tc]
                if t is not None:
                    if t == color + pt and best > 5:
                        best = 5
                    break
                tr += dr
                tc += dc
    return 0 if best == 99 else best


class Bot:
    """Paranoid alpha-beta over the greedy eval. `best_move(state)` plays the side to move."""

    def __init__(self, params: Params | str = "greedy", depth: int | None = None,
                 time_ms: float | None = None, max_depth: int = 12, tt_size: int = 400_000):
        self.p = PRESETS[params] if isinstance(params, str) else params
        self.depth, self.time_ms, self.max_depth, self.tt_size = depth, time_ms, max_depth, tt_size
        self.tt: dict = {}
        self.stats: dict = {}
        self.name = params if isinstance(params, str) else "custom"

    def __call__(self, state):
        return self.best_move(state)

    # --- eval pieces ---------------------------------------------------------------------

    def _delta(self, me: str, state: ChaturajiState, move: Move, info, flags):
        """(value change for me from this move, new flags). flags = (endgame, trade) where
        trade is None or (value I captured, plies since)."""
        p = self.p
        r = state.turn // 4
        endgame, trade = flags
        d = 0.0
        mover = info.color
        if trade is not None:
            trade = (trade[0], trade[1] + 1) if trade[1] < 3 else None
        cap = info.captured
        if mover == me:
            d += info.points
            if cap is not None and cap[0] != DEAD:
                if cap[1] == "K" and r < p.king_bonus_rounds:
                    d += p.king_bonus
                if cap[0] != me:
                    trade = (PIECE_POINTS[cap[1]], 0)
            if p.push_bonus and r >= p.push_from and info.piece[1] == "P":
                d += p.push_bonus
        elif cap is not None and cap[0] == me:
            v = PIECE_POINTS[cap[1]]
            if cap[1] == "K":
                kings = sum(1 for row in state.board for x in row if x and x[0] != DEAD and x[1] == "K")
                if r < p.king_loss_rounds:
                    d -= (p.king_loss_rounds - r) * kings
                d -= p.king_floor
                d -= p.loss_weight * v
            else:
                d -= p.loss_weight * v
                if cap[1] == "P" and r < p.pawn_loss_rounds:
                    d -= p.pawn_loss
                if endgame:
                    d -= p.endgame_loss
                if trade is not None and v <= trade[0] and r < p.trade_rounds:
                    d += p.trade_bonus
                    trade = None
        if not endgame:
            if info.eliminated is not None and r >= 10:
                endgame = True
            elif info.piece[1] == "P":
                er, ec = move.end
                if _pawn_to_go(mover, er, ec) <= p.endgame_pawn_dist:
                    endgame = True
        return d, (endgame, trade)

    def _threat(self, me: str, state: ChaturajiState) -> float:
        """Expected loss at a leaf from my pieces that opponents attack right now."""
        board = state.board
        r = state.turn // 4
        order = []                                   # opponents in the order they move before me
        s = SEAT[state.current_player()]
        for k in range(4):
            c = TURN_ORDER[(s + k) % 4]
            if c == me:
                break
            if c not in state.eliminated:
                order.append(c)
        if not order:                                # I am to move: opponents come after me
            order = [TURN_ORDER[(SEAT[me] + k) % 4] for k in range(1, 4)]
            order = [c for c in order if c not in state.eliminated]
            mine_to_move = True
        else:
            mine_to_move = False
        kings = None
        losses = []
        for rr in range(8):
            for cc in range(8):
                pc = board[rr][cc]
                if not pc or pc[0] != me:
                    continue
                v = PIECE_POINTS[pc[1]]
                worst = 0.0
                for k, c in enumerate(order):
                    a = attackers(board, rr, cc, c)
                    if not a:
                        continue
                    if pc[1] == "K":
                        if kings is None:
                            kings = sum(1 for row in board for x in row if x and x[0] != DEAD and x[1] == "K")
                        loss = max(0, self.p.king_loss_rounds - r) * kings + self.p.loss_weight * v
                    else:
                        defended = attackers(board, rr, cc, me) > 0
                        net = max(0.0, v - a) if defended else v
                        loss = self.p.loss_weight * net
                    worst = max(worst, loss * THREAT_P[min(k + (1 if mine_to_move else 0), 2)])
                if worst:
                    losses.append(worst)
        if not losses:
            return 0.0
        if mine_to_move:
            losses.remove(max(losses))
        return sum(losses)

    def _leaf(self, me: str, state: ChaturajiState) -> float:
        v = -self._threat(me, state) if self.p.threat else 0.0
        return v

    def _claim(self, me: str, state: ChaturajiState) -> float | None:
        """Value (for me) if the mover claims the win now; only with two players left."""
        if len(state.eliminated) != 2:
            return None
        return placement_value(eliminate(state, state.current_player()).scores, me)

    @staticmethod
    def _key(me: str, state: ChaturajiState, flags):
        # `me` is part of the key: values are from the searching seat's view, and one Bot may
        # play several seats (the GUI and match scripts share one per preset)
        return (me, tuple(tuple(row) for row in state.board), state.turn % 4, min(state.turn // 4, 30),
                tuple(state.scores[c] for c in TURN_ORDER), tuple(sorted(state.eliminated)), flags)

    def _tick(self):
        self.nodes += 1
        if self.deadline is not None and (self.nodes & 127) == 0 and time.perf_counter() > self.deadline:
            raise _Timeout

    def _ordered(self, state: ChaturajiState, moves, tt_move, only_captures=False):
        board = state.board
        scored = []
        for m in moves:
            victim = board[m.end[0]][m.end[1]]
            gain = 0
            if victim is not None:
                gain = 3 if victim == DEAD + "K" else (0 if victim[0] == DEAD else PIECE_POINTS[victim[1]])
                if victim[0] == DEAD and victim[1] != "K":
                    victim = None
            if only_captures and (victim is None or gain == 0):
                continue
            att = PIECE_POINTS[board[m.start[0]][m.start[1]][1]]
            key = 1000 if tt_move == m else (gain * 10 - att + 100 if victim is not None else 0)
            scored.append((key, m))
        scored.sort(key=lambda t: -t[0])
        return [m for _, m in scored]

    # --- search --------------------------------------------------------------------------

    def _child(self, me, state, m, flags):
        child, info = apply_move(state, m)
        d, nf = self._delta(me, state, m, info, flags)
        if me in child.eliminated:
            return child, d, None, True               # I was eliminated: line over
        if is_game_over(child):
            return child, d + placement_value(child.scores, me), None, True
        return child, d, nf, False

    def _quiesce(self, me, state, alpha, beta, flags, q):
        self._tick()
        maxing = state.current_player() == me
        stand = self._leaf(me, state)
        claim = self._claim(me, state)
        if claim is not None:
            stand = max(stand, claim) if maxing else min(stand, claim)
        if q <= 0:
            return stand
        if maxing:
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
        else:
            if stand <= alpha:
                return stand
            beta = min(beta, stand)
        best = stand
        for m in self._ordered(state, legal_moves(state), None, only_captures=True):
            child, d, nf, term = self._child(me, state, m, flags)
            v = d if nf is None and term else d + self._quiesce(me, child, alpha - d, beta - d, nf, q - 1)
            if maxing:
                if v > best:
                    best = v
                    alpha = max(alpha, v)
            elif v < best:
                best = v
                beta = min(beta, v)
            if alpha >= beta:
                break
        return best

    def _search(self, me, state, depth, alpha, beta, flags):
        if depth <= 0:
            return self._quiesce(me, state, alpha, beta, flags, self.p.qdepth)
        self._tick()
        key = self._key(me, state, flags)
        ent = self.tt.get(key)
        tt_move = None
        a0, b0 = alpha, beta
        if ent is not None:
            ed, eflag, ev, tt_move = ent
            if ed >= depth:
                if eflag == 0:
                    return ev
                if eflag == 1:
                    alpha = max(alpha, ev)
                else:
                    beta = min(beta, ev)
                if alpha >= beta:
                    return ev
        maxing = state.current_player() == me
        best = -INF if maxing else INF
        best_move = None
        claim = self._claim(me, state)
        if claim is not None:
            best = claim
            if maxing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
        if alpha < beta:
            for m in self._ordered(state, legal_moves(state), tt_move):
                child, d, nf, term = self._child(me, state, m, flags)
                if term:
                    v = d
                else:
                    v = d + self._search(me, child, depth - 1, alpha - d, beta - d, nf)
                if maxing:
                    if v > best:
                        best, best_move = v, m
                        alpha = max(alpha, v)
                elif v < best:
                    best, best_move = v, m
                    beta = min(beta, v)
                if alpha >= beta:
                    break
        flag = 0 if a0 < best < b0 else (1 if best >= b0 else 2)
        if len(self.tt) < self.tt_size:
            self.tt[key] = (depth, flag, best, best_move)
        return best

    def best_move(self, state: ChaturajiState) -> Move | Resign:
        me = state.current_player()
        moves = require_moves(state)
        flags0 = (endgame_now(state.board), None)
        self.nodes = 0
        self.deadline = (time.perf_counter() + self.time_ms / 1000.0) if self.time_ms else None
        t0 = time.perf_counter()
        if len(self.tt) > self.tt_size // 2:
            self.tt.clear()
        claim = self._claim(me, state)
        best, bv, reached = moves[0], -INF, 0
        limit = self.depth or self.p.depth
        if self.time_ms and not self.depth:
            limit = self.max_depth
        order = self._ordered(state, moves, None)
        for d in range(1, limit + 1):
            try:
                alpha = -INF
                cur_best, cur_v, scores = None, -INF, []
                for m in order:
                    child, dl, nf, term = self._child(me, state, m, flags0)
                    v = dl if term else dl + self._search(me, child, d - 1, alpha - dl, INF, nf)
                    scores.append((v, m))
                    if v > cur_v:
                        cur_v, cur_best = v, m
                        alpha = max(alpha, v)
                best, bv, reached = cur_best, cur_v, d
                scores.sort(key=lambda t: -t[0])
                order = [m for _, m in scores]
            except _Timeout:
                break
        secs = time.perf_counter() - t0
        if claim is not None and claim > bv:
            best, bv = RESIGN, claim
        self.stats = {"depth": reached, "nodes": self.nodes, "seconds": round(secs, 4), "value": round(bv, 2)}
        return best


# --- timing ------------------------------------------------------------------------------

def sample_positions(seed: int = 7) -> dict[str, ChaturajiState]:
    """Opening (start), middlegame (round 6) and endgame (a king is gone and a pawn is close to
    promotion, else round 25). Reached by depth-2 greedy play with 15% random moves, so it is
    reproducible and not wiped out the way random play is."""
    rng = random.Random(seed)
    bots = [Bot("greedy", depth=2), Bot("naive"), Bot("greedy-trade", depth=2), Bot("naive")]
    out = {"opening": ChaturajiState.initial()}
    st = ChaturajiState.initial()
    while not is_game_over(st) and st.turn < 200:
        if rng.random() < 0.15:
            act = rng.choice(require_moves(st))
        else:
            act = bots[st.turn % 4](st)
        st, _ = apply_action(st, act)
        if "middle" not in out and st.turn >= 24:
            out["middle"] = st
        if "end" not in out and st.turn >= 60 and (st.eliminated or endgame_now(st.board) or st.turn >= 100):
            out["end"] = st
            break
    out.setdefault("middle", st)
    out.setdefault("end", st)
    return out


def time_table(depths, preset: str = "greedy", positions: dict | None = None) -> list[dict]:
    rows = []
    for name, st in (positions or sample_positions()).items():
        for d in depths:
            bot = Bot(preset, depth=d)
            t = time.perf_counter()
            bot.best_move(st)
            dt = time.perf_counter() - t
            rows.append({"bot": preset, "position": name, "depth": d, "seconds": round(dt, 3),
                         "nodes": bot.stats["nodes"], "knps": round(bot.stats["nodes"] / dt / 1000, 1)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


# --- match -------------------------------------------------------------------------------

def _player(spec: str, depth: int | None, time_ms: float | None):
    if spec == "uniform":
        rng = random.Random()
        return lambda st: rng.choice(require_moves(st))
    if spec in PRESETS:
        return Bot(spec, depth=depth, time_ms=time_ms)
    if spec.startswith("mlp:"):
        from .evaluate import load_evaluator
        from .mcts import MCTS, Node
        from .registry import find_model
        mcts = MCTS(load_evaluator(find_model(spec[4:])), num_simulations=64, max_turn=None)

        def play(st):
            root = Node(st)
            mcts.search(root)
            return root.moves[mcts.choose(root)]
        return play
    raise SystemExit(f"unknown player {spec!r}: uniform, mlp:<model>, or one of {sorted(PRESETS)}")


def match(specs, games: int, depth: int | None, time_ms: float | None, max_turns: int = 400, out=None) -> dict:
    players = [_player(s, depth, time_ms) for s in specs]
    totals = {i: 0.0 for i in range(4)}
    wins = {i: 0 for i in range(4)}
    for g in range(games):
        shift = g % 4                                  # rotate seats so colour does not matter
        seat_of = {TURN_ORDER[(i + shift) % 4]: i for i in range(4)}
        st = ChaturajiState.initial()
        while not is_game_over(st) and st.turn < max_turns:
            who = st.current_player()
            act = players[seat_of[who]](st)
            st, _ = apply_action(st, act)
        best = max(st.scores.values())
        for c, i in seat_of.items():
            totals[i] += st.scores[c]
            wins[i] += st.scores[c] == best
        rec = {"game": g, "scores": {specs[i] + f"#{i}": st.scores[c] for c, i in seat_of.items()}, "turns": st.turn}
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return {f"{specs[i]}#{i}": {"mean_score": round(totals[i] / games, 2), "wins": wins[i]} for i in range(4)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time", help="seconds per move by depth (plies)")
    t.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    t.add_argument("--bots", nargs="+", default=["greedy", "naive"])
    m = sub.add_parser("match", help="four seats; each is uniform, mlp:<model> or a greedy preset")
    m.add_argument("seats", nargs=4)
    m.add_argument("--games", type=int, default=4)
    m.add_argument("--depth", type=int, default=None)
    m.add_argument("--time-ms", type=float, default=None)
    m.add_argument("--max-turns", type=int, default=400)
    m.add_argument("--out", default=None, help="append one JSON line per game here")
    a = ap.parse_args(argv)
    if a.cmd == "time":
        pos = sample_positions()
        for b in a.bots:
            time_table(a.depths, b, pos)
    else:
        print(json.dumps(match(a.seats, a.games, a.depth, a.time_ms, a.max_turns, a.out), indent=1))


if __name__ == "__main__":
    main()
