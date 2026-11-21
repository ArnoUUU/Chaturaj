"""Supervised training on scraped chess.com games (the "bootstrap" before self-play).

Every move in a scraped game becomes one training position:
  input   the position before the move, from the mover's point of view
  policy  the move that was actually played (with a little label smoothing over legal moves)
  value   who finished with the top score (the real final result)
  score   the real final score margins

Usage (after scraping, e.g. `python -m scraper.game_scraping player arnav2007 --time-control "1 | 5"`):
    python -m bot.supervised --player arnav2007 --time-control "1 | 5" --tag sup-arnav2007-blitz
    python -m bot.supervised --records scraper/data/records --epochs 20     # every scraped game

Continue with self-play from the result:
    python -m bot.train --init models/<label> --tag rl-from-sup ...

Saves models/<label>/ (model, meta with validation metrics and evaluation) and
logs/<label>/ (chunked logs), like bot.train.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chaturaji.moves import Move
from chaturaji.record import Game

from . import registry
from .evaluate import evaluate_vs_random
from .features import canonical_action, legal_canonical, mover_index, outcome_vectors, to_relative
from .selfplay import Sample
from .train import ReplayBuffer

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "scraper" / "data" / "records"
PLAYERS = ROOT / "scraper" / "data" / "players"


def record_paths(records_dir: Path = RECORDS, player: str | None = None,
                 time_controls: list[str] | None = None) -> list[Path]:
    """Engine records to train on: all of them, or one player's games (optionally per control)."""
    if not player:
        return sorted(records_dir.glob("*.json"))
    from scraper.game_scraping import tc_slug
    base = PLAYERS / player.lower()
    slugs = {tc_slug(t) for t in time_controls} if time_controls else None
    ids = []
    for idx in sorted(base.glob("*/index.jsonl")):
        if slugs and idx.parent.name not in slugs:
            continue
        ids += [json.loads(l)["game_id"] for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [records_dir / f"{gid}.json" for gid in dict.fromkeys(ids) if (records_dir / f"{gid}.json").exists()]


def samples_from_game(game: Game, encoder, smoothing: float = 0.1,
                      only_player: str | None = None) -> list[Sample]:
    """One sample per played move. `only_player` keeps just that player's moves."""
    share, margin = outcome_vectors(game.state)
    seat_of = {}
    if only_player:
        seats = game.metadata.get("seats") or {}
        seat_of = {s["name"].lower(): color[0] for color, s in seats.items()}
    keep_color = seat_of.get(only_player.lower()) if only_player else None
    out = []
    for h, state in zip(game.history, game.states()):
        if h.get("uci") is None:            # resignation / timeout entry
            continue
        if keep_color and h["color"] != keep_color:
            continue
        k = mover_index(state)
        _, actions = legal_canonical(state)
        played = canonical_action(Move.from_uci(h["uci"], state.board), k)
        policy = np.full(len(actions), smoothing / len(actions), dtype=np.float32)
        policy[np.flatnonzero(actions == played)] += 1.0 - smoothing
        out.append(Sample(encoder(state), actions, policy,
                          to_relative(share, k).astype(np.float32),
                          to_relative(margin, k).astype(np.float32)))
    return out


@torch.no_grad()
def validate(model, samples: list[Sample], kind: str, device: str, batch: int = 512) -> dict:
    """Policy top-1 accuracy (played move), value/policy loss on held-out games."""
    model.eval()
    if not samples:
        return {}
    buf = ReplayBuffer(len(samples))
    buf.extend(samples)
    hits = n = 0
    pol_loss = val_loss = 0.0
    for start in range(0, len(samples), batch):
        chunk = samples[start:start + batch]
        buf.data.clear()
        buf.extend(chunk)
        inputs, pol_t, mask, v_t, _ = buf.batch(len(chunk), device, kind)
        logits, v_logits, _ = model(*inputs)
        logits = logits.masked_fill(~mask, -1e9)
        hits += (logits.argmax(1) == pol_t.argmax(1)).sum().item()
        pol_loss += -(pol_t * F.log_softmax(logits, 1)).sum(1).sum().item()
        val_loss += -(v_t * F.log_softmax(v_logits, 1)).sum(1).sum().item()
        n += len(chunk)
    return {"top1": round(hits / n, 4), "policy_loss": round(pol_loss / n, 4),
            "value_loss": round(val_loss / n, 4), "positions": n}


def run(args) -> str:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = record_paths(Path(args.records), args.player, args.time_controls)
    if not paths:
        raise SystemExit("no scraped game records found. Run scraper.game_scraping first.")
    games = []
    for p in paths:
        try:
            games.append(Game.load(p))
        except Exception as e:  # a record that no longer replays (e.g. rules changed)
            print(f"skip {p.name}: {e}")
    random.shuffle(games)
    n_val = max(1, int(len(games) * args.val_fraction)) if len(games) > 1 else 0
    val_games, train_games = games[:n_val], games[n_val:]

    if args.init:
        model, kind, config, _ = registry.load_model(args.init, args.device)
    else:
        kind = args.model
        config = registry.default_config(kind)
        if kind == "mlp":
            config.update({k: v for k, v in (("embed", args.embed), ("widths", args.widths)) if v})
        model = registry.build_model(kind, config).to(args.device)
    encoder = registry.encoder(kind)
    only = args.player if args.player_moves_only else None
    train_s = [s for g in train_games for s in samples_from_game(g, encoder, args.smoothing, only)]
    val_s = [s for g in val_games for s in samples_from_game(g, encoder, args.smoothing, only)]

    label = "_".join([datetime.now().strftime("%Y-%m-%d_%H%M"), args.tag,
                      f"{kind}-{registry.size_tag(kind, config)}", f"sup-ep{args.epochs}",
                      "gpu" if args.device.startswith("cuda") else "cpu"])
    log = registry.ChunkedLogger(label, args.log_chunk)
    meta = {"label": label, "kind": kind, "config": config, "params": registry.param_count(model),
            "args": vars(args), "environment": registry.environment_info(args.device),
            "training": "supervised", "status": "running", "history": [], "evaluation": None,
            "data": {"games": len(games), "train_games": len(train_games), "val_games": len(val_games),
                     "train_positions": len(train_s), "val_positions": len(val_s),
                     "game_ids": [g.metadata.get("chesscom_id") for g in games]}}
    log.log("start", **{k: meta[k] for k in ("label", "kind", "params")}, data={
        k: v for k, v in meta["data"].items() if k != "game_ids"})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_s) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    buf = ReplayBuffer(len(train_s) + 1)
    buf.extend(train_s)
    best = None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            inputs, pol_t, mask, v_t, sc_t = buf.batch(args.batch, args.device, kind)
            logits, v_logits, sc_pred = model(*inputs)
            logits = logits.masked_fill(~mask, -1e9)
            policy_loss = -(pol_t * F.log_softmax(logits, 1)).sum(1).mean()
            value_loss = -(v_t * F.log_softmax(v_logits, 1)).sum(1).mean()
            score_loss = F.mse_loss(sc_pred, sc_t)
            loss = policy_loss + args.value_weight * value_loss + args.score_weight * score_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append([policy_loss.item(), value_loss.item(), score_loss.item()])
        train_loss = dict(zip(("policy", "value", "score"), np.round(np.mean(losses, 0), 4).tolist()))
        val = validate(model, val_s, kind, args.device)
        entry = {"epoch": epoch, "train_loss": train_loss, "val": val,
                 "lr": round(sched.get_last_lr()[0], 6)}
        meta["history"].append(entry)
        log.log("epoch", **entry)
        if val and (best is None or val["policy_loss"] < best):
            best = val["policy_loss"]
            meta["best_epoch"] = epoch
            registry.save_model(model, kind, config, label, meta, opt)
    if best is None:
        registry.save_model(model, kind, config, label, meta, opt)

    model, _, _, _ = registry.load_model(registry.model_dir(label), args.device)  # best epoch
    if args.eval_games:
        meta["evaluation"] = evaluate_vs_random(registry.evaluator(kind, model, args.device),
                                                args.eval_games, args.eval_sims, "puct", seed=args.seed)
        log.log("evaluation", **meta["evaluation"])
    meta["status"] = "complete"
    meta["history"] = meta["history"]
    registry.save_model(model, kind, config, label, meta)
    registry.register(label, meta)
    log.log("done", label=label)
    return label


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--player", help="train on this player's scraped games")
    ap.add_argument("--time-control", action="append", dest="time_controls",
                    help='only these controls, e.g. "1 | 5" (repeatable)')
    ap.add_argument("--player-moves-only", action="store_true",
                    help="policy/value targets only from the player's own moves")
    ap.add_argument("--tag", default="sup")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", help="start from an existing models/<label>")
    ap.add_argument("--model", choices=registry.KINDS, default="mlp")
    ap.add_argument("--embed", type=int)
    ap.add_argument("--widths", type=int, nargs=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--score-weight", type=float, default=0.5)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--eval-sims", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-chunk", type=int, default=200)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
