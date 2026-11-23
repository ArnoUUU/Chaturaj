"""Self-play training: self-play -> replay buffer -> SGD -> labelled model + chunked logs.

Each run gets a label, e.g. 2026-11-21_1530_smoke-cpu_mlp-e64-w128x64x32_ucb1-s16_it2_cpu:
  models/<label>/model.pt, meta.json   (updated every iteration, so runs can resume)
  models/index.jsonl                   (one line per finished run, with its evaluation)
  logs/<label>/log_0000.jsonl, ...     (chunked JSON-lines logs)
  logs/<label>/games/*.json            (sample self-play games, open in chaturaji.viewer)

Small CPU test run:
    python -m bot.train --tag smoke-cpu --model mlp --embed 64 --widths 128 64 32 \
        --selection ucb1 --iterations 2 --games 2 --sims 16 --steps 20 --batch 64

GPU run on the cluster (see cluster/README.md):
    python -m bot.train --tag tillicum --device cuda --model mlp --iterations 50 \
        --games 16 --sims 200 --steps 500 --batch 1024

Time-budgeted run (cluster/selftrain.slurm): keep iterating for --hours, and every
--snapshot-every minutes keep a copy in models/<label>/snapshots/snap_NNN_itXXXXX_<time>.pt:
    python -m bot.train --tag selftrain --iterations 0 --hours 23.5 --snapshot-every 60
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from . import registry
from .evaluate import evaluate_vs_random
from .features import NUM_ACTIONS
from .mcts import MCTS, SELECTIONS
from .selfplay import Sample, play_game


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Sample] = deque(maxlen=capacity)

    def extend(self, samples: list[Sample]) -> None:
        self.data.extend(samples)

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, size: int, device: str, kind: str = "resnet"):
        picks = random.sample(range(len(self.data)), min(size, len(self.data)))
        items = [self.data[i] for i in picks]
        inputs = registry.collate(kind, [s.x for s in items], device)
        pol = torch.zeros(len(items), NUM_ACTIONS)
        mask = torch.zeros(len(items), NUM_ACTIONS, dtype=torch.bool)
        for j, s in enumerate(items):
            a = torch.from_numpy(s.actions)
            pol[j, a] = torch.from_numpy(s.policy)
            mask[j, a] = True
        v = torch.from_numpy(np.stack([s.value for s in items]))
        sc = torch.from_numpy(np.stack([s.score for s in items]))
        return inputs, pol.to(device), mask.to(device), v.to(device), sc.to(device)


def compute_loss(model, batch, score_weight: float = 0.5):
    inputs, pol_t, mask, v_t, sc_t = batch
    logits, v_logits, sc_pred = model(*inputs)
    logits = logits.masked_fill(~mask, -1e9)  # only legal moves compete
    policy_loss = -(pol_t * F.log_softmax(logits, 1)).sum(1).mean()
    value_loss = -(v_t * F.log_softmax(v_logits, 1)).sum(1).mean()
    score_loss = F.mse_loss(sc_pred, sc_t)
    total = policy_loss + value_loss + score_weight * score_loss
    return total, {"policy": policy_loss.item(), "value": value_loss.item(),
                   "score": score_loss.item(), "total": total.item()}


def model_config(args) -> dict:
    cfg = registry.default_config(args.model)
    if args.model == "mlp":
        cfg.update({k: v for k, v in (("embed", args.embed), ("widths", args.widths)) if v})
    else:
        cfg.update({k: v for k, v in (("channels", args.channels), ("blocks", args.blocks)) if v})
    return cfg


def train(args) -> str:
    device = args.device
    if not args.iterations and not args.hours:
        raise SystemExit("--iterations 0 (run until time is up) needs --hours")
    run_start = time.time()
    deadline = run_start + args.hours * 3600 if args.hours else None
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.resume:
        model, kind, config, ckpt = registry.load_model(args.resume, device)
        label = ckpt.get("label") or args.resume
        start_iter = (ckpt.get("iteration") or -1) + 1
        step = ckpt.get("step", 0)
    else:
        if args.init:  # new run starting from an existing model's weights (e.g. supervised)
            model, kind, config, _ = registry.load_model(args.init, device)
        else:
            kind, config = args.model, model_config(args)
            model = registry.build_model(kind, config).to(device)
        label = registry.make_label(kind, config, args.selection, args.sims, args.iterations,
                                    device, args.tag)
        ckpt, start_iter, step = None, 0, 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if ckpt and ckpt.get("optimizer"):
        opt.load_state_dict(ckpt["optimizer"])

    # full records go to the log files; stdout (Slurm .out) only gets short lines
    log = registry.ChunkedLogger(label, args.log_chunk, echo=not args.quiet)
    games_dir = log.dir / "games"
    games_dir.mkdir(exist_ok=True)
    meta = {"label": label, "kind": kind, "config": config, "params": registry.param_count(model),
            "args": vars(args), "environment": registry.environment_info(device),
            "status": "running", "history": [], "evaluation": None}
    log.log("start", label=label, kind=kind, params=meta["params"], device=device,
            environment=meta["environment"])
    if args.quiet:
        print(f"start {label} | {kind} {meta['params']:,} params | {device}", flush=True)

    encode = registry.encoder(kind)
    buffer = ReplayBuffer(args.buffer)
    snap_sec = args.snapshot_every * 60
    next_snap = run_start + snap_sec if snap_sec > 0 else None
    snap_dir = registry.model_dir(label) / "snapshots"
    meta["snapshots"] = []

    def maybe_snapshot(it: int) -> None:
        """Every --snapshot-every minutes, keep a copy of the current weights + optimizer."""
        nonlocal next_snap
        now = time.time()
        if next_snap is None or now < next_snap:
            return
        while next_snap <= now:
            next_snap += snap_sec
        n = len(list(snap_dir.glob("snap_*.pt"))) + 1 if snap_dir.exists() else 1
        name = f"snap_{n:03d}_it{it:05d}_{datetime.now():%Y%m%d-%H%M}"
        # weights only (no optimizer state): about a third of the size of model.pt
        path = registry.save_snapshot(model, kind, config, label, {**meta, "step": step}, name)
        if args.keep_snapshots > 0:
            for old in sorted(snap_dir.glob("snap_*.pt"), key=lambda f: f.stat().st_mtime)[:-args.keep_snapshots]:
                old.unlink()
        info = {"name": name, "iteration": it, "step": step,
                "hours": round((now - run_start) / 3600, 2), "path": str(path)}
        meta["snapshots"].append(info)
        log.log("snapshot", **info)

    def out_of_time(extra: float = 0.0) -> bool:
        return deadline is not None and time.time() + extra >= deadline

    iterations = (range(start_iter, start_iter + args.iterations) if args.iterations
                  else itertools.count(start_iter))
    last_iter_sec = 0.0
    for it in iterations:
        if out_of_time(last_iter_sec):  # not enough time left for another full iteration
            log.log("time_up", iteration=it, hours=round((time.time() - run_start) / 3600, 2))
            break
        iter_start = time.time()
        model.eval()
        mcts = MCTS(registry.evaluator(kind, model, device), num_simulations=args.sims,
                    max_turn=args.max_plies, rng=rng, selection=args.selection,
                    c_puct=args.c_puct, c_ucb=args.c_ucb)
        t0 = time.time()
        reasons = []
        for g in range(args.games):
            samples, game = play_game(mcts, args.temp_plies, args.max_plies,
                                      {"iteration": it, "game": g, "label": label}, encoder=encode)
            buffer.extend(samples)
            reasons.append(game.result()["reason"] or "ply_cap")
            if g < args.save_games and it % max(1, args.save_games_every) == 0:
                game.save(games_dir / f"iter{it:04d}_game{g}.json")
            maybe_snapshot(it)
            if out_of_time():
                break
        selfplay_sec = time.time() - t0

        model.train()
        losses = []
        t0 = time.time()
        for _ in range(args.steps):
            loss, parts = compute_loss(model, buffer.batch(args.batch, device, kind), args.score_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(parts)
            step += 1
            maybe_snapshot(it)
            if out_of_time():
                break
        avg = {k: round(float(np.mean([l[k] for l in losses])), 4) for k in losses[0]} if losses else {}
        entry = {"iteration": it, "step": step, "buffer": len(buffer),
                 "selfplay_sec": round(selfplay_sec, 1), "train_sec": round(time.time() - t0, 1),
                 "loss": avg, "game_endings": dict(Counter(reasons))}
        meta["history"].append(entry)
        meta.update(step=step, iteration=it)
        registry.save_model(model, kind, config, label, meta, opt)
        log.log("iteration", **entry)
        if args.quiet and it % max(1, args.print_every) == 0:
            hrs = (time.time() - run_start) / 3600
            print(f"it {it} | {hrs:.2f}h | step {step} | buf {len(buffer)} | loss "
                  + " ".join(f"{k}={v}" for k, v in avg.items()), flush=True)
        last_iter_sec = time.time() - iter_start

    if args.eval_games:
        model.eval()
        meta["evaluation"] = evaluate_vs_random(registry.evaluator(kind, model, device),
                                                args.eval_games, args.sims, args.selection,
                                                seed=args.seed)
        log.log("evaluation", **meta["evaluation"])
        if args.quiet:
            print("evaluation", {k: v for k, v in meta["evaluation"].items() if not isinstance(v, (list, dict))}, flush=True)
    meta["status"] = "complete"
    registry.save_model(model, kind, config, label, meta, opt)
    registry.register(label, meta)
    log.log("done", label=label, model=str(registry.model_dir(label) / "model.pt"))
    if args.quiet:
        print(f"done {label}", flush=True)
    return label


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="short run name used in the label, e.g. smoke-cpu")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="models/<label> folder or model.pt to continue")
    ap.add_argument("--init", default=None,
                    help="start a NEW run (new label) from models/<label> weights, e.g. a supervised model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=registry.KINDS, default="mlp")
    ap.add_argument("--embed", type=int, help="mlp: encoding width (default 512)")
    ap.add_argument("--widths", type=int, nargs=3, help="mlp: 3 FC widths (default 1024 512 256)")
    ap.add_argument("--channels", type=int, help="resnet: channels (default 64)")
    ap.add_argument("--blocks", type=int, help="resnet: residual blocks (default 4)")
    ap.add_argument("--selection", choices=SELECTIONS, default="puct")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--c-ucb", type=float, default=2 ** 0.5)
    ap.add_argument("--iterations", type=int, default=10, help="0 = keep going until --hours is used up")
    ap.add_argument("--hours", type=float, default=None,
                    help="time budget: no new iteration starts if it would run past this")
    ap.add_argument("--snapshot-every", type=float, default=60,
                    help="minutes between snapshot copies in models/<label>/snapshots/ (0 = off)")
    ap.add_argument("--games", type=int, default=8, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--save-games", type=int, default=1, help="self-play games saved per saving iteration")
    ap.add_argument("--save-games-every", type=int, default=1, help="save sample games every N iterations")
    ap.add_argument("--keep-snapshots", type=int, default=0, help="keep only the newest N snapshots (0 = all)")
    ap.add_argument("--quiet", action="store_true",
                    help="stdout gets one short line per --print-every iterations; full detail stays in logs/<label>/")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200, help="training steps per iteration")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=500_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--score-weight", type=float, default=0.5)
    ap.add_argument("--eval-games", type=int, default=4, help="games vs random after training (0 = skip)")
    ap.add_argument("--log-chunk", type=int, default=200, help="log lines per chunk file")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
