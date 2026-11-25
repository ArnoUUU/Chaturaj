"""Train the NNUE value network (bot/nnue.py) on real game outcomes.

Data, every position labelled with the game's actual final result:
  scraped   chess.com games (scraper/data/records), every position
  selfplay  games.jsonl from self-play runs (training and check games), every `stride`-th ply

Target: for each of the 6 seat pairs (mover-relative), 1 if seat i finished above seat j,
0.5 on equal points, 0 below. Loss: binary cross-entropy on the 6 logits.

Validation (held-out games): the loss, and pairwise accuracy on decided pairs (who finishes
above whom), next to a baseline that just predicts "whoever leads in points now".

    python -m bot.train_nnue --selfplay logs/<run>/games.jsonl --tag nnue-v1 --epochs 6
    python -m bot.train_nnue --preset xl --dropout 0.1 --device cuda --cache data.npz ...

Sizes: --preset small | medium | large | xl (see bot.nnue.PRESETS), or --acc / --hidden.
--cache saves the built positions (or loads them if the file exists), so several sizes
can train on the same data without rebuilding it (about 6 minutes for 10,000 games).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from chaturaji.game import apply_move
from chaturaji.moves import Move
from chaturaji.record import Game
from chaturaji.state import ChaturajiState
from chaturaji.constants import TURN_ORDER

from . import registry
from .features import mover_index, to_relative
from .nnue import PRESETS, nnue_features, pairwise_loss, pairwise_targets, save_weights

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "scraper" / "data" / "records"


class PositionSet:
    """Positions in flat arrays: per position its 4 perspectives' indices (mover first)."""

    def __init__(self):
        self.idx, self.n, self.g, self.y, self.now, self.game = [], [], [], [], [], []

    def add(self, state: ChaturajiState, final_scores: dict, game_id: int) -> None:
        bags, g = nnue_features(state)
        k = mover_index(state)
        final = to_relative(np.array([final_scores[c] for c in TURN_ORDER], dtype=np.float32), k)
        now = to_relative(np.array([state.scores[c] for c in TURN_ORDER], dtype=np.float32), k)
        self.idx.append(np.concatenate(bags).astype(np.int16))
        self.n.append(len(bags[0]))
        self.g.append(g)
        self.y.append(pairwise_targets(final))
        self.now.append(pairwise_targets(now))
        self.game.append(game_id)

    def finish(self) -> "PositionSet":
        self.n = np.array(self.n, dtype=np.int64)
        self.start = np.concatenate([[0], np.cumsum(4 * self.n)[:-1]]).astype(np.int64)
        self.idx = np.concatenate(self.idx) if self.idx else np.zeros(0, np.int16)
        self.g, self.y, self.now = (np.stack(a).astype(np.float32) for a in (self.g, self.y, self.now))
        self.game = np.array(self.game, dtype=np.int64)
        return self

    def __len__(self):
        return len(self.n)

    FIELDS = ("idx", "n", "start", "g", "y", "now", "game")

    def save(self, path, prefix: str) -> dict:
        return {f"{prefix}_{f}": getattr(self, f) for f in self.FIELDS}

    @classmethod
    def from_arrays(cls, z, prefix: str) -> "PositionSet":
        ps = cls()
        for f in cls.FIELDS:
            setattr(ps, f, z[f"{prefix}_{f}"])
        return ps

    def batch(self, rows: np.ndarray, device: str = "cpu"):
        parts = [self.idx[self.start[r]:self.start[r] + 4 * self.n[r]] for r in rows]
        lengths = np.repeat(self.n[rows], 4)
        indices = torch.from_numpy(np.concatenate(parts).astype(np.int64))
        offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
        return ((indices.to(device), offsets.to(device), torch.from_numpy(self.g[rows]).to(device)),
                torch.from_numpy(self.y[rows]).to(device))


def scraped_games(records: Path):
    for p in sorted(records.glob("*.json")):
        try:
            game = Game.load(p)
        except Exception:
            continue
        states = game.states()
        yield [s for h, s in zip(game.history, states) if h.get("uci") is not None], game.state.scores


def selfplay_games(path: Path, stride: int):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = ChaturajiState.initial()
        out = []
        for ply, uci in enumerate(rec["moves"]):
            if ply % stride == 0:
                out.append(s)
            s, _ = apply_move(s, Move.from_uci(uci, s.board))
        if s.scores != rec["scores"]:
            continue                                   # does not replay to its logged result
        yield out, rec["scores"]


def build(records: Path | None, selfplay: list[Path], stride: int, max_games: int | None,
          val_fraction: float, seed: int):
    sources = []
    if records:
        sources.append(("scraped", scraped_games(records)))
    for p in selfplay:
        sources.append((p.parent.name, selfplay_games(p, stride)))
    games = []
    counts = {}
    for name, it in sources:
        n = 0
        for states, scores in it:
            if max_games and n >= max_games:
                break
            games.append((states, scores))
            n += 1
        counts[name] = n
    rng = random.Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_fraction))
    train, val = PositionSet(), PositionSet()
    for gid, (states, scores) in enumerate(games):
        target = val if gid < n_val else train
        for s in states:
            target.add(s, scores, gid)
    return train.finish(), val.finish(), counts, n_val


@torch.no_grad()
def validate(model, data: PositionSet, device: str, batch: int = 4096) -> dict:
    model.eval()
    loss, hits, base, decided, n = 0.0, 0.0, 0.0, 0, 0
    for i in range(0, len(data), batch):
        rows = np.arange(i, min(i + batch, len(data)))
        inputs, y = data.batch(rows, device)
        logits = model(*inputs)
        loss += pairwise_loss(logits, y).item() * len(rows)
        n += len(rows)
        p = torch.sigmoid(logits).cpu().numpy()
        yt = y.cpu().numpy()
        mask = yt != 0.5
        decided += mask.sum()
        hits += ((p > 0.5) == (yt > 0.5))[mask].sum()
        now = data.now[rows]
        base += np.where(now == 0.5, 0.5, (now > 0.5) == (yt > 0.5))[mask].sum()
    model.train()
    return {"loss": round(loss / max(n, 1), 4), "pair_acc": round(float(hits / max(decided, 1)), 4),
            "baseline_score_lead_acc": round(float(base / max(decided, 1)), 4), "positions": n}


def run(args) -> str:
    args.device = registry.resolve_device(args.device)
    print(f"device: {args.device}", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    if args.cache and Path(args.cache).is_file():
        with np.load(args.cache) as z:
            train, val = PositionSet.from_arrays(z, "train"), PositionSet.from_arrays(z, "val")
            counts, n_val = json.loads(str(z["counts"])), int(z["n_val"])
    else:
        train, val, counts, n_val = build(None if args.no_scraped else Path(args.records),
                                          [Path(p) for p in args.selfplay], args.stride,
                                          args.max_games, args.val_fraction, args.seed)
        if args.cache:
            np.savez(args.cache, **train.save(args.cache, "train"), **val.save(args.cache, "val"),
                     counts=json.dumps(counts), n_val=n_val)
    print(f"data: {counts} games -> {len(train):,} train / {len(val):,} val positions "
          f"({n_val} val games) in {time.time() - t0:.0f} s", flush=True)

    kind = "nnue"
    config = dict(PRESETS[args.preset])
    if args.acc:
        config["acc"] = args.acc
    if args.hidden:
        config["hidden"] = args.hidden
    if args.residual is not None:
        config["residual"] = args.residual
    if args.dropout:
        config["dropout"] = args.dropout
    model = registry.build_model(kind, config).to(args.device)
    label = registry.make_label(kind, config, "maxn", 0, 0, args.device, args.tag)
    label = label.replace("_maxn-s0_timed", f"_ep{args.epochs}")
    log = registry.ChunkedLogger(label)
    meta = {"label": label, "kind": kind, "config": config, "params": registry.param_count(model),
            "args": vars(args), "environment": registry.environment_info(args.device),
            "data": {"games": counts, "train_positions": len(train), "val_positions": len(val)},
            "history": [], "status": "running"}
    print(f"model: {config} = {meta['params']:,} weights ({4 * meta['params'] / 1e6:.0f} MB float32)",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
    best = None
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(train))
        losses = []
        for s in range(steps_per_epoch):
            inputs, y = train.batch(perm[s * args.batch:(s + 1) * args.batch], args.device)
            loss = pairwise_loss(model(*inputs), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = validate(model, val, args.device)
        entry = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val": v,
                 "minutes": round((time.time() - t0) / 60, 1)}
        meta["history"].append(entry)
        log.log("epoch", **entry)
        print(f"epoch {epoch + 1}/{args.epochs} | train {entry['train_loss']:.4f} | val {v['loss']:.4f} "
              f"| pair acc {v['pair_acc']:.3f} (score-lead baseline {v['baseline_score_lead_acc']:.3f})",
              flush=True)
        if best is None or v["loss"] < best:
            best = v["loss"]
            meta["best_epoch"] = epoch
            registry.save_model(model, kind, config, label, meta)
    model, _, _, _ = registry.load_model(registry.model_dir(label), args.device)
    save_weights(model, registry.model_dir(label) / "nnue.npz")
    meta["status"] = "done"
    meta["val"] = meta["history"][meta["best_epoch"]]["val"]
    registry.save_model(model, kind, config, label, meta)
    registry.register(label, meta)
    log.log("done", label=label)
    print(f"done {label}", flush=True)
    return label


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--no-scraped", action="store_true")
    ap.add_argument("--selfplay", nargs="*", default=[], help="games.jsonl files")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th ply of self-play games")
    ap.add_argument("--max-games", type=int, default=None, help="per source (for quick tests)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--acc", type=int, default=None, help="accumulator width (overrides the preset)")
    ap.add_argument("--hidden", type=int, nargs="+", default=None, help="hidden widths (overrides)")
    ap.add_argument("--residual", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--cache", default=None, help="positions .npz: written after building, read if present")
    ap.add_argument("--tag", default="nnue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto (CUDA when available), cpu or cuda")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
