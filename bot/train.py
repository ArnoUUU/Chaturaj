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
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque

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

    log = registry.ChunkedLogger(label, args.log_chunk)
    games_dir = log.dir / "games"
    games_dir.mkdir(exist_ok=True)
    meta = {"label": label, "kind": kind, "config": config, "params": registry.param_count(model),
            "args": vars(args), "environment": registry.environment_info(device),
            "status": "running", "history": [], "evaluation": None}
    log.log("start", label=label, kind=kind, params=meta["params"], device=device,
            environment=meta["environment"])

    encode = registry.encoder(kind)
    buffer = ReplayBuffer(args.buffer)
    for it in range(start_iter, start_iter + args.iterations):
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
            if g < args.save_games:
                game.save(games_dir / f"iter{it:04d}_game{g}.json")
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
        avg = {k: round(float(np.mean([l[k] for l in losses])), 4) for k in losses[0]} if losses else {}
        entry = {"iteration": it, "step": step, "buffer": len(buffer),
                 "selfplay_sec": round(selfplay_sec, 1), "train_sec": round(time.time() - t0, 1),
                 "loss": avg, "game_endings": reasons}
        meta["history"].append(entry)
        meta.update(step=step, iteration=it)
        registry.save_model(model, kind, config, label, meta, opt)
        log.log("iteration", **entry)

    if args.eval_games:
        model.eval()
        meta["evaluation"] = evaluate_vs_random(registry.evaluator(kind, model, device),
                                                args.eval_games, args.sims, args.selection,
                                                seed=args.seed)
        log.log("evaluation", **meta["evaluation"])
    meta["status"] = "complete"
    registry.save_model(model, kind, config, label, meta, opt)
    registry.register(label, meta)
    log.log("done", label=label, model=str(registry.model_dir(label) / "model.pt"))
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
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--games", type=int, default=8, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--save-games", type=int, default=1, help="self-play games saved per iteration")
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
