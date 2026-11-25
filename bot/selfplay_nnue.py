"""Self-play training of the NNUE with the compiled max^n search.

Loop (one iteration):
  1. Play `--games` self-play games in parallel worker processes (bot/selfplay_worker.py).
     Every seat is max^n with the current network at a fixed depth (`--depth`, `--qdepth`,
     `--check-ext`); variety comes from a random opening of 4-8 plies and an occasional
     random move early on (`--explore`).
  2. Every position of every game (all four seats) becomes a training sample, labelled
     with the game's real final result: for each pair of seats, 1 if seat i finished above
     seat j, 0.5 on equal points, 0 below.
  3. Train `--steps` steps. Each batch mixes self-play positions (a replay buffer of the
     latest `--buffer`) with the earlier data (scraped games + the MLP league run's games,
     `--base-frac` of each batch), so the network can't drift away from real play.
  4. Save the weights (model.pt + nnue.npz) for the next iteration's games.

Checks (at the start, every `--check-min` minutes and at the end):
  vs_best       max^n with the current weights in one seat vs the best weights so far in
                the other three, every seat, paired random openings (`--check-games`).
                New best if pairwise >= 0.5; back to the best weights if clearly worse
                (pairwise + 2 SE < 0.5).
  vs_mcts       max^n (same depth) vs MCTS + MLP (the league run's final model, 32
                simulations) in the other three seats (`--yard-games`), the fixed yardstick.

Logged (everything is verbose on purpose; `python -m bot.run_data <label>` loads it into tables):
  models/<label>/meta.json   run arguments, environment, config, "history" (one entry per
                             iteration: losses and their spread, gradient norms, endings,
                             game lengths (plies min/median/max), resignations, random
                             moves, search depth/nodes/seconds, timings), "checks" (vs the
                             best weights and vs MCTS + MLP), status.
  logs/<label>/games.jsonl   every self-play game and every check game: seed, opening length,
                             scores, ending, plies, moves (codes; RESIGN = 65535), and for
                             self-play per ply: movers, kinds (opening / explore / search /
                             forced / resign), root values, depths, nodes, seconds; the
                             resignations (ply, seat, players still in, scores); captures of
                             an active piece (ply, attacker, victim, points); checks; eliminations.
                             Check games vs MCTS carry the same per-move info ("move_info").
  logs/<label>/steps.csv     one row per optimiser step: iteration, step, global_step, loss,
                             grad_norm, lr.
  logs/<label>/log_*.jsonl   the iteration / check / done events as they were logged.

    python -m bot.selfplay_nnue --hours 5
    python -m bot.selfplay_nnue --init models/<nnue> --hours 1 --workers 8 --tag try
    python -m bot.selfplay_nnue --init models/<earlier self-play run> --until 06:00   # continue
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import registry
from .nnue import features_from_codes, pairwise_loss, pairwise_targets, save_weights
from .train_nnue import RECORDS, PositionSet, build

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INIT = "models/2026-11-23_1607_nnue-v1_nnue-a256-h32x32_ep8_cpu"
DEFAULT_MLP = "models/opponent_pool/league_model.pt"
DEFAULT_LEAGUE_GAMES = "logs/2026-11-23_0144_league-5h_mlp-e512-w1024x512x256_puct-s32_timed_cpu/games.jsonl"


class Replay:
    """The latest `capacity` self-play positions (ring buffer), same batch format as
    train_nnue.PositionSet."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.idx: list = [None] * capacity
        self.n = np.zeros(capacity, dtype=np.int64)
        self.g = np.zeros((capacity, 16), dtype=np.float32)
        self.y = np.zeros((capacity, 6), dtype=np.float32)
        self.size = self.head = 0

    def add(self, bags: list[np.ndarray], g: np.ndarray, y: np.ndarray) -> None:
        i = self.head
        self.idx[i] = np.concatenate(bags).astype(np.int16)
        self.n[i] = len(bags[0])
        self.g[i], self.y[i] = g, y
        self.head = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def __len__(self):
        return self.size

    def parts(self, rows: np.ndarray):
        return [self.idx[r] for r in rows], np.repeat(self.n[rows], 4), self.g[rows], self.y[rows]


def base_parts(ps: PositionSet, rows: np.ndarray):
    return ([ps.idx[ps.start[r]:ps.start[r] + 4 * ps.n[r]] for r in rows], np.repeat(ps.n[rows], 4),
            ps.g[rows], ps.y[rows])


def make_batch(parts_list):
    idx = [a for p in parts_list for a in p[0]]
    lengths = np.concatenate([p[1] for p in parts_list])
    indices = torch.from_numpy(np.concatenate(idx).astype(np.int64))
    offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
    g = torch.from_numpy(np.concatenate([p[2] for p in parts_list]))
    y = torch.from_numpy(np.concatenate([p[3] for p in parts_list]))
    return (indices, offsets, g), y


def game_stats(rec: dict) -> dict:
    """The ply of the game's first elimination (None if no one was eliminated)."""
    el = rec.get("eliminations") or []
    first = min(p for p, _, _ in el) if el else None
    return {"first_elim": first}


def pairwise_summary(pw: list[float], seats: list[str]) -> dict:
    a = np.array(pw)
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    per_seat = {s: round(float(np.mean([x for x, t in zip(pw, seats) if t == s])), 3) for s in "RBYG" if s in seats}
    return {"games": len(a), "pairwise": round(float(a.mean()), 3), "pairwise_se": round(se, 3),
            "per_seat_pairwise": per_seat}


@torch.no_grad()
def eval_loss(model, parts_fn, n: int, chunk: int = 4096) -> float | None:
    """Mean pairwise loss over rows 0..n-1 of a data source (no training)."""
    if n == 0:
        return None
    was = model.training
    model.eval()
    tot = 0.0
    for a in range(0, n, chunk):
        rows = np.arange(a, min(n, a + chunk))
        inputs, y = make_batch([parts_fn(rows)])
        tot += pairwise_loss(model(*inputs), y).item() * len(rows)
    model.train(was)
    return tot / n


def run(args) -> str:
    from . import selfplay_worker as W
    if W.E is None:
        raise SystemExit("the compiled core is needed (engine/README.md)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    if args.until:                     # a wall-clock end time ("06:00"): the next one from now
        from datetime import datetime, timedelta
        hh, mm = (int(x) for x in args.until.split(":"))
        end = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if end <= datetime.now():
            end += timedelta(days=1)
        args.hours = (end - datetime.now()).total_seconds() / 3600
        print(f"training until {end:%Y-%m-%d %H:%M} ({args.hours:.2f} h)", flush=True)

    def hours() -> float:
        return (time.time() - t0) / 3600

    model, kind, config, init_ckpt = registry.load_model(args.init)
    if kind != "nnue":
        raise SystemExit(f"--init must be an nnue model, got {kind}")
    model.train()
    label = registry.make_label(kind, config, "maxn", args.depth, 0, "cpu", args.tag)
    mdir = registry.model_dir(label)
    ldir = ROOT / "logs" / label
    wdir = ldir / "weights"
    for d in (mdir, ldir, wdir, mdir / "snapshots"):
        d.mkdir(parents=True, exist_ok=True)
    log = registry.ChunkedLogger(label)
    games_log = open(ldir / "games.jsonl", "a", encoding="utf-8")
    steps_path = ldir / "steps.csv"
    steps_new = not steps_path.exists() or steps_path.stat().st_size == 0
    steps_log = open(steps_path, "a", encoding="utf-8")
    if steps_new:
        steps_log.write("iteration,step,global_step,loss,grad_norm,lr\n")
        steps_log.flush()
    meta = {"label": label, "kind": kind, "config": config, "params": registry.param_count(model),
            "args": vars(args), "environment": registry.environment_info("cpu"), "init": str(args.init),
            "history": [], "checks": [], "status": "running"}

    def plot_curves() -> None:
        try:                           # the standard training-curves figure (bot/plot_run.py)
            from .plot_run import plot_meta
            plot_meta(meta, ldir / "curves.png")
            shutil.copy(ldir / "curves.png", mdir / "curves.png")
        except Exception as e:         # never stop training over a figure
            print(f"(curves not drawn: {e})", flush=True)

    def save_current(name: str) -> str:
        path = wdir / f"{name}.npz"
        save_weights(model, path)
        return str(path)

    # --- data: earlier positions (base) + self-play replay buffer
    tb = time.time()
    if args.cache and Path(args.cache).is_file():
        with np.load(args.cache) as z:
            base = PositionSet.from_arrays(z, "train")
    else:
        sp = [Path(p) for p in args.base_selfplay if Path(p).is_file()]
        base, _, counts, n_val = build(Path(args.records), sp, 2, None, 0.05, args.seed)
        if args.cache:
            np.savez(args.cache, **base.save(args.cache, "train"), counts=json.dumps(counts), n_val=n_val)
    replay = Replay(args.buffer)
    val_replay = Replay(args.val_buffer)    # held-out self-play games, never trained on
    # base validation: every `val_every`-th earlier game, held out of the training batches
    held = (base.game % args.val_every) == 0 if args.val_every else np.zeros(len(base), dtype=bool)
    base_train = np.flatnonzero(~held)
    base_val = np.flatnonzero(held)
    rng_v = np.random.default_rng(args.seed + 1)
    if len(base_val) > args.val_base:
        base_val = np.sort(rng_v.choice(base_val, args.val_base, replace=False))
    base_val_fn = lambda rows: base_parts(base, base_val[rows])  # noqa: E731
    meta["validation"] = {"base_positions": int(len(base_val)), "base_every": args.val_every,
                          "selfplay_every": args.val_selfplay_every, "selfplay_buffer": args.val_buffer}
    print(f"base positions: {len(base):,} ({len(base_train):,} for training, {len(base_val):,} held out) "
          f"({time.time() - tb:.0f} s); model {config}, {meta['params']:,} weights", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if init_ckpt.get("optimizer") and init_ckpt.get("kind") == "nnue":
        try:                           # continuing a self-play run: keep its optimizer state
            opt.load_state_dict(init_ckpt["optimizer"])
            for g in opt.param_groups:
                g["lr"], g["weight_decay"] = args.lr, args.weight_decay
            meta["optimizer_resumed"] = True
        except (ValueError, KeyError):
            pass

    current = save_current("it00000")
    best = str(wdir / "best.npz")
    shutil.copy(current, best)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    meta["best"] = "start"

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers)
    common = {"depth": args.depth, "qdepth": args.qdepth, "check_ext": args.check_ext, "width": args.width,
              "full_plies": args.full_plies, "discount": args.discount, "elim_bonus": args.elim_bonus,
              "elim_horizon": args.elim_horizon, "max_turn": 400, "resign": not args.no_resign}
    total_steps = 0

    def check(tag: str, it: int, final: bool = False) -> None:
        nonlocal best, best_state
        tc = time.time()
        entry = {"check": tag, "iteration": it, "hours": round(hours(), 3)}
        cur = save_current(f"check_{tag}")
        if tag != "start":
            tasks = [{**common, "cand": cur, "opp": best, "seat": g % 4, "opening_seed": [args.seed, 77, it, g // 4],
                      "opening_plies": 4} for g in range(args.check_games)]
            res = pool.map(W.match_game, tasks)
            vs = pairwise_summary([r["pairwise"] for r in res], [r["seat"] for r in res])
            entry["vs_best"] = {**vs, "against": meta["best"]}
            for r in res:
                games_log.write(json.dumps({"type": "check", "check": tag, "vs": "best", **r}) + "\n")
            hi = vs["pairwise"] + 2 * vs["pairwise_se"]
            if vs["pairwise"] >= 0.5:
                best = str(wdir / "best.npz")
                shutil.copy(cur, best)
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                meta["best"] = tag
                entry["decision"] = "new best"
                registry.save_snapshot(model, kind, config, label, meta, f"best_{tag}")
            elif hi < 0.5:
                model.load_state_dict(best_state)
                entry["decision"] = f"clearly worse: back to {meta['best']}"
            else:
                entry["decision"] = f"not better than {meta['best']}; kept training"
        if args.yard_games:
            yp = ctx.Pool(args.yard_workers)
            ytasks = [{**common, "weights": cur, "mlp": args.mlp, "sims": args.yard_sims, "opening_plies": 4,
                       "seed": args.seed * 1000 + b, "max_plies": 300} for b in range(args.yard_games // 4)]
            blocks = yp.map(W.yardstick_block, ytasks)
            yp.close()
            yp.join()
            recs = [r for b in blocks for r in b["records"]]
            vs = pairwise_summary([r["pairwise"] for r in recs], [r["seat"] for r in recs])
            entry["vs_mcts"] = {**vs, "mlp": args.mlp, "sims": args.yard_sims,
                                "maxn_seconds_per_move": round(float(np.mean([b["maxn_seconds"] for b in blocks])), 4)}
            for r in recs:
                games_log.write(json.dumps({"type": "check", "check": tag, "vs": "mcts", **r}) + "\n")
        entry["seconds"] = round(time.time() - tc, 1)
        meta["checks"].append(entry)
        meta["check_sec"] = entry["seconds"]
        log.log("check", **entry)
        games_log.flush()
        line = f"check {tag} @ {entry['hours']:.2f}h"
        if "vs_best" in entry:
            line += f" | vs best {entry['vs_best']['pairwise']:.3f} ± {2 * entry['vs_best']['pairwise_se']:.3f} -> {entry['decision']}"
        if "vs_mcts" in entry:
            v = entry["vs_mcts"]
            line += f" | vs MCTS+MLP {v['pairwise']:.3f} ± {2 * v['pairwise_se']:.3f} (max^n {v['maxn_seconds_per_move']:.3f} s/move)"
        print(line + f" [{entry['seconds']:.0f} s]", flush=True)
        plot_curves()

    v0 = eval_loss(model, base_val_fn, len(base_val))
    meta["history"].append({"iteration": 0, "hours": 0.0, "steps": 0, "lr": args.lr,
                            "val_base_loss": round(v0, 5) if v0 is not None else None})
    print(f"start | val loss (held-out earlier games) {v0}", flush=True)
    check("start", 0)
    reserve = lambda: 1.15 * meta.get("check_sec", 300)  # noqa: E731
    it, last_check, last_iter = 0, time.time(), 0.0
    budget = args.hours * 3600
    while time.time() - t0 + last_iter + reserve() < budget:
        ti = time.time()
        it += 1
        tasks = [{**common, "weights": current, "seed": [args.seed, it, g], "open_lo": 4, "open_hi": 8,
                  "explore": args.explore, "explore_plies": 30, "stride": 1} for g in range(args.games)]
        results = pool.map(W.selfplay_game, tasks)
        tg = time.time() - ti
        endings, plies, first_el, n_pos = Counter(), [], [], 0
        n_claims = n_early = n_random = 0
        s_depth, s_nodes, s_sec = [], [], []
        for gi, r in enumerate(results):
            scores = np.array(r["scores"], dtype=np.float32)
            held_out = bool(args.val_selfplay_every) and gi % args.val_selfplay_every == 0
            for b, k, g in zip(r["boards"], r["ks"], r["globals"]):
                k = int(k)
                (val_replay if held_out else replay).add(features_from_codes(b, k), g,
                                                         pairwise_targets(np.roll(scores, -k)))
                n_pos += 1
            st = game_stats(r)
            endings[r["reason"]] += 1
            plies.append(r["plies"])
            if st["first_elim"] is not None:
                first_el.append(st["first_elim"])
            for rs in r["resigns"]:
                n_claims += rs["active"] == 2
                n_early += rs["active"] >= 3
            for kind, d, nd, sec in zip(r["kinds"], r["depths"], r["nodes"], r["secs"]):
                n_random += kind in ("opening", "explore")
                if d is not None:
                    s_depth.append(d)
                    s_nodes.append(nd)
                    s_sec.append(sec)
            games_log.write(json.dumps({"type": "selfplay", "iteration": it, "seed": r["seed"], "opening": r["opening"],
                                        "scores": r["scores"], "reason": r["reason"], "plies": r["plies"],
                                        "moves": r["moves"], "captures": r["captures"], "checks": r["checks"],
                                        "eliminations": r.get("eliminations", []),
                                        **{k: r[k] for k in W.MOVE_FIELDS}, "resigns": r["resigns"]}) + "\n")
        games_log.flush()
        losses, gnorms = [], []
        if len(replay) >= args.min_buffer:
            nb = int(args.batch * args.base_frac)
            for _ in range(args.steps):
                parts = [replay.parts(np.random.randint(0, len(replay), args.batch - nb))]
                if nb:
                    parts.append(base_parts(base, base_train[np.random.randint(0, len(base_train), nb)]))
                inputs, y = make_batch(parts)
                loss = pairwise_loss(model(*inputs), y)
                opt.zero_grad()
                loss.backward()
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                opt.step()
                losses.append(loss.item())
                total_steps += 1
        if losses:                     # one row per optimiser step, written in one go
            lr_now = opt.param_groups[0]["lr"]
            first = total_steps - len(losses)
            steps_log.write("".join(f"{it},{i + 1},{first + i + 1},{losses[i]:.6f},{gnorms[i]:.6f},{lr_now:.8g}\n"
                                    for i in range(len(losses))))
            steps_log.flush()
        prev = current
        current = save_current(f"it{it:05d}")
        if Path(prev).name.startswith("it") and prev != current:
            Path(prev).unlink(missing_ok=True)
        te = time.time()
        vb = eval_loss(model, base_val_fn, len(base_val))
        vsp = eval_loss(model, val_replay.parts, len(val_replay))
        eval_sec = time.time() - te
        last_iter = time.time() - ti
        entry = {"iteration": it, "hours": round(hours(), 3), "games": len(results), "positions": n_pos,
                 "buffer": len(replay), "selfplay_sec": round(tg, 1), "train_sec": round(last_iter - tg, 1),
                 "loss": round(float(np.mean(losses)), 4) if losses else None,
                 "val_base_loss": round(vb, 5) if vb is not None else None,
                 "val_selfplay_loss": round(vsp, 5) if vsp is not None else None,
                 "val_selfplay_positions": len(val_replay),
                 "lr": opt.param_groups[0]["lr"], "steps": total_steps,
                 "grad_norm": round(float(np.mean(gnorms)), 4) if gnorms else None, "endings": dict(endings),
                 "mean_plies": round(float(np.mean(plies)), 1),
                 "plies_min": int(min(plies)), "plies_median": round(float(np.median(plies)), 1),
                 "plies_max": int(max(plies)),
                 "first_elim_ply": round(float(np.mean(first_el)), 1) if first_el else None,
                 "loss_std": round(float(np.std(losses)), 5) if losses else None,
                 "grad_norm_max": round(float(max(gnorms)), 4) if gnorms else None,
                 "resigns": {"claims": n_claims, "early": n_early},
                 "random_moves": n_random,
                 "search": {"mean_depth": round(float(np.mean(s_depth)), 2) if s_depth else None,
                            "mean_nodes": round(float(np.mean(s_nodes)), 1) if s_nodes else None,
                            "mean_sec": round(float(np.mean(s_sec)), 4) if s_sec else None},
                 "eval_sec": round(eval_sec, 1)}
        meta["history"].append(entry)
        meta["iteration"] = it
        log.log("iteration", **entry)
        registry.save_model(model, kind, config, label, meta, opt)
        print(f"it {it} | {entry['hours']:.2f}h | games {len(results)} ({tg:.0f} s) | buffer {len(replay):,} | "
              f"loss {entry['loss']} | val {entry['val_base_loss']} / {entry['val_selfplay_loss']} | plies {entry['mean_plies']} | "
              f"first elim {entry['first_elim_ply']}", flush=True)
        if time.time() - last_check >= args.check_min * 60 and time.time() - t0 + reserve() * 2 < budget:
            check(f"h{hours():.1f}", it)
            last_check = time.time()
    check("final", it, final=True)
    pool.close()
    pool.join()
    final_cmp = meta["checks"][-1].get("vs_best", {})
    if final_cmp and final_cmp["pairwise"] + 2 * final_cmp["pairwise_se"] < 0.5:
        model.load_state_dict(best_state)          # the final weights are clearly worse: keep the best
        meta["final_is_best_snapshot"] = True
    meta["status"] = "done"
    meta["run_hours"] = round(hours(), 3)
    registry.save_model(model, kind, config, label, meta, opt)
    save_weights(model, mdir / "nnue.npz")
    registry.register(label, meta)
    plot_curves()
    log.log("done", label=label)
    games_log.close()
    steps_log.close()
    print(f"done {label}", flush=True)
    return label


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=DEFAULT_INIT, help="starting NNUE (models/<label>)")
    ap.add_argument("--tag", default="nnue-selfplay")
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--until", default=None, help='wall-clock end time "HH:MM" (overrides --hours)')
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 3)))
    ap.add_argument("--games", type=int, default=48, help="self-play games per iteration")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--qdepth", type=int, default=2)
    ap.add_argument("--check-ext", type=int, default=0,
                    help="check extensions per line (off by default: no gain in a 48-game arena test)")
    ap.add_argument("--width", type=int, default=0,
                    help="selective depth: children searched per node from ply --full-plies on (0 = all)")
    ap.add_argument("--full-plies", type=int, default=2, help="plies from the root searched at full width")
    ap.add_argument("--discount", type=float, default=0.9985,
                    help="search preference for shorter games: values pulled towards 0.5 by this per move (1 = off)")
    ap.add_argument("--elim-bonus", type=float, default=0.008,
                    help="search preference for early eliminations: utility added to the eliminator (0 = off); "
                         "keep below a pawn (about 0.01 for nnue-v1)")
    ap.add_argument("--elim-horizon", type=int, default=200, help="turn at which the elimination bonus reaches 0")
    ap.add_argument("--explore", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--base-frac", type=float, default=0.5)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--min-buffer", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--check-min", type=float, default=60)
    ap.add_argument("--val-every", type=int, default=20, help="hold out every n-th earlier game for validation")
    ap.add_argument("--val-base", type=int, default=20_000, help="held-out earlier positions to validate on")
    ap.add_argument("--val-selfplay-every", type=int, default=10, help="hold out every n-th self-play game")
    ap.add_argument("--val-buffer", type=int, default=40_000, help="latest held-out self-play positions kept")
    ap.add_argument("--check-games", type=int, default=48)
    ap.add_argument("--yard-games", type=int, default=40)
    ap.add_argument("--yard-workers", type=int, default=4)
    ap.add_argument("--yard-sims", type=int, default=32)
    ap.add_argument("--mlp", default=DEFAULT_MLP)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--base-selfplay", nargs="*", default=[DEFAULT_LEAGUE_GAMES])
    ap.add_argument("--cache", default="logs/nnue_cache.npz")
    ap.add_argument("--no-resign", action="store_true",
                    help="the mover may not resign (the default: resigning is an action, see the rules)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
