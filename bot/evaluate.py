"""Evaluation: the bot (MCTS + model) in one seat against three opponents.

Opponents are random movers, or another model (e.g. the supervised warm start or an older
snapshot) searching with the same MCTS settings. The opponent model fills all three other
seats; each seat searches for itself, so the three don't cooperate.

The bot takes each seat in turn (game g -> seat g % 4) so no seat advantage leaks in.
Reported numbers:
  pairwise    : chess.com's FFA rule. Each game is three 1v1 results against the other
                seats by final points: win 1, draw 0.5, loss 0. pairwise = the mean over
                all those results (fair = 0.5). This is the headline number.
  score_share : bot's points / all points (fair = 0.25), for reference only
  win_rate    : fraction of games where the bot had the top score (ties count)
  avg_place   : 1 (best) to 4

Compare past models from the command line:
    python -m bot.evaluate models/<new> --opponent models/<supervised> --games 40
    python -m bot.evaluate models/<new> --opponent models/<new>/snapshots/snap_003_*.pt random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chaturaji.constants import TURN_ORDER
from chaturaji.game import apply_action, apply_move, game_over_reason, is_game_over, require_moves
from chaturaji.state import ChaturajiState

from .mcts import MCTS, Node


def _r4(x):
    """Floats rounded to 4 decimals (numpy arrays and nested lists included); None stays None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


def _mcts_info(root: Node, idx: int, seat: str, by: str) -> dict:
    """The move-info entry of an MCTS move: the root's visits and values, and the chosen
    child's mean value (its visits include the resign child's pre-visit, as in choose())."""
    n = max(float(root.child_n[idx]), 1.0)
    return {"seat": seat, "by": by, "sims": MCTS.real_visits(root), "value": _r4(root.value),
            "q": _r4(root.child_w[idx] / n)}


def evaluate(evaluator, games: int = 4, sims: int = 16, selection: str = "puct",
             opponent=None, opponent_name: str = "random", max_plies: int = 300,
             seed: int = 0, opening_plies: int = 0, pool: list | None = None,
             record: bool = False, bot_player=None, bot_stats=None) -> dict:
    """`opponent` is an MCTS evaluator for the three other seats, or None for random moves.

    `pool` (a list of (name, evaluator)) replaces `opponent` with a mixed field: each block
    of 4 games draws three pool members (repeats allowed) for the other seats, and the bot
    plays that lineup from every seat. `bot_player` (a function state -> Move, e.g. a max^n
    search) plays the bot's seat instead of MCTS over `evaluator`; `bot_stats` (a zero-argument
    callable) is then called after each of its moves, and its dict goes into that move's entry.
    With `record`, every game (seat, opponents, scores, result, moves, and "move_info": one
    entry per ply with the seat, who chose it, and for searched moves the visit counts, root
    values and the chosen child's mean value) is returned under "records".

    Model-vs-model games are deterministic (argmax search, no noise), so with
    `opening_plies` > 0 each block of 4 games shares one random opening of that many
    plies (seeded by the block number) and the bot plays it from each seat in turn.
    """
    rng = np.random.default_rng(seed)
    mcts = MCTS(evaluator, num_simulations=sims, selection=selection, max_turn=None, rng=rng)
    opp = (MCTS(opponent, num_simulations=sims, selection=selection, max_turn=None, rng=rng)
           if opponent is not None else None)
    pool_mcts = [(name, MCTS(ev, num_simulations=sims, selection=selection, max_turn=None, rng=rng))
                 for name, ev in (pool or [])]
    shares, wins, places, seats, pair, records = [], [], [], [], [], []
    wdl = [0, 0, 0]
    for g in range(games):
        bot = TURN_ORDER[g % 4]
        others = [c for c in TURN_ORDER if c != bot]
        if pool_mcts:   # one lineup per block of 4, same draw for every seat rotation
            picks = np.random.default_rng([seed, g // 4, 7]).integers(len(pool_mcts), size=3)
            seat_opp = {c: pool_mcts[i] for c, i in zip(others, picks)}
        else:
            seat_opp = {c: (opponent_name, opp) for c in others}
        moves_played, move_info = [], []
        state = ChaturajiState.initial()
        plies = 0
        opening_rng = np.random.default_rng([seed, g // 4])
        while not is_game_over(state) and plies < opening_plies:
            moves = require_moves(state)
            move = moves[opening_rng.integers(len(moves))]
            moves_played.append(move.uci)
            if record:
                move_info.append({"seat": state.current_player(), "by": "opening"})
            state, _ = apply_move(state, move)
            plies += 1
        while not is_game_over(state) and plies < max_plies:
            who = state.current_player()
            if who == bot and bot_player is not None:
                move = bot_player(state)
                if record:
                    info = {"seat": who, "by": "bot"}
                    if bot_stats is not None:
                        info.update(_r4(bot_stats()))
                    move_info.append(info)
            elif who == bot:
                root = mcts.search(Node(state))
                idx = mcts.choose(root)
                move = root.moves[idx]
                if record:
                    move_info.append(_mcts_info(root, idx, who, "bot"))
            elif seat_opp[who][1] is not None:
                m = seat_opp[who][1]
                root = m.search(Node(state))
                idx = m.choose(root)
                move = root.moves[idx]
                if record:
                    move_info.append(_mcts_info(root, idx, who, "opponent"))
            else:
                moves = require_moves(state)
                move = moves[rng.integers(len(moves))]
                if record:
                    move_info.append({"seat": who, "by": "random"})
            moves_played.append(move.uci)   # "resign" for RESIGN
            state, _ = apply_action(state, move)
            plies += 1
        scores = state.scores
        total = sum(scores.values())
        shares.append(scores[bot] / total if total else 0.25)
        wins.append(scores[bot] == max(scores.values()))
        places.append(1 + sum(1 for c in TURN_ORDER if scores[c] > scores[bot]))
        res = [1.0 if scores[bot] > scores[c] else 0.5 if scores[bot] == scores[c] else 0.0
               for c in TURN_ORDER if c != bot]
        pair.append(sum(res) / 3)
        for r in res:
            wdl[0 if r == 1 else 1 if r == 0.5 else 2] += 1
        seats.append(bot)
        if record:
            records.append({"game": g, "seat": bot, "opponents": {c: n for c, (n, _) in seat_opp.items()},
                            "scores": dict(scores), "pairwise": round(sum(res) / 3, 3),
                            "results": {c: r for c, r in zip(others, res)},
                            "reason": game_over_reason(state) or "ply_cap", "plies": plies,
                            "opening_plies": min(opening_plies, plies), "moves": moves_played,
                            "move_info": move_info})
    per_seat = {c: round(float(np.mean([s for s, b in zip(shares, seats) if b == c])), 3)
                for c in TURN_ORDER if c in seats}
    per_seat_pair = {c: round(float(np.mean([s for s, b in zip(pair, seats) if b == c])), 3)
                     for c in TURN_ORDER if c in seats}
    return {"opponent": opponent_name, "games": games, "sims": sims, "selection": selection,
            "opening_plies": opening_plies,
            "pairwise": round(float(np.mean(pair)), 3),
            "pairwise_se": round(float(np.std(pair, ddof=1) / np.sqrt(len(pair))), 3)
            if len(pair) > 1 else None,
            "per_seat_pairwise": per_seat_pair,
            "wins_draws_losses": wdl,
            "game_pairwise": [round(float(x), 3) for x in pair],
            "score_share": round(float(np.mean(shares)), 3),
            "score_share_se": round(float(np.std(shares, ddof=1) / np.sqrt(len(shares))), 3)
            if len(shares) > 1 else None,
            "per_seat_share": per_seat,
            "game_shares": [round(float(x), 3) for x in shares],
            "win_rate": round(float(np.mean(wins)), 3),
            "avg_place": round(float(np.mean(places)), 2),
            **({"records": records} if record else {})}


def evaluate_vs_random(evaluator, games: int = 4, sims: int = 16, selection: str = "puct",
                       max_plies: int = 300, seed: int = 0) -> dict:
    return evaluate(evaluator, games, sims, selection, None, "random", max_plies, seed)


def model_name(path: str | Path) -> str:
    """Short name for a model path: the run label, plus the snapshot name if it is one."""
    p = Path(path)
    if p.name == "model.pt":
        p = p.parent
    if p.parent.name == "snapshots":
        return f"{p.parent.parent.name}/{p.stem}"
    return p.stem if p.suffix == ".pt" else p.name


def load_evaluator(path: str | Path, device: str = "cpu"):
    from . import registry
    model, kind, _, _ = registry.load_model(path, device)
    model.eval()
    return registry.evaluator(kind, model, device)


def evaluate_vs(evaluator, opponent: str | Path, device: str = "cpu", **kw) -> dict:
    """Evaluate against "random", "benchmark" (the current benchmark model) or a saved model
    (any model reference: see registry.find_model)."""
    if str(opponent) == "benchmark":
        from . import benchmark
        bench = benchmark.current()
        if bench is None:
            opponent = "random"
        else:
            opponent = benchmark.model_file(bench)
            kw.setdefault("opening_plies", 4)
    if str(opponent) == "random":
        return evaluate(evaluator, opponent=None, opponent_name="random", **kw)
    return evaluate(evaluator, opponent=load_evaluator(opponent, device),
                    opponent_name=model_name(opponent), **kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="models/<label>, a model.pt, or a snapshot .pt")
    ap.add_argument("--opponent", nargs="+", default=["random"],
                    help='one or more opponents: "random" or model paths (each is evaluated separately)')
    ap.add_argument("--games", type=int, default=40, help="games per opponent (multiples of 4 balance seats)")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--selection", default="puct")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    ev = load_evaluator(args.model, args.device)
    for opp in args.opponent:
        result = evaluate_vs(ev, opp, args.device, games=args.games, sims=args.sims,
                             selection=args.selection, max_plies=args.max_plies, seed=args.seed)
        print(json.dumps({"model": model_name(args.model), **result}), flush=True)


if __name__ == "__main__":
    main()
