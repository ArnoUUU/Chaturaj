"""Benchmark ladder: new models are judged against the current best, not random movers.

The benchmark is one saved model, referenced from models/benchmark.json (a path under
models/, e.g. opponent_pool/BEST_<name>.pt, or logs/models/... for a retired model). A challenger plays in one seat
with the benchmark in the other three; it plays every seat in turn (so every starting
color), and each block of 4 games shares one seeded random opening (see
bot.evaluate.evaluate). Fair score share is 0.25.

Scoring follows chess.com's FFA rating rule: each game is three 1v1 results against the
other seats by final points (win 1, draw 0.5, loss 0); the "pairwise" score is their mean,
fair = 0.5. The share of points is reported but not used.

A challenger outperforms the benchmark consistently when:
  - it played at least MIN_GAMES games (a multiple of 4, so every seat equally)
  - mean pairwise score - 2 standard errors > 0.5  (clearly above fair, not luck)
  - pairwise score > 0.5 from at least 3 of the 4 seats
With promotion on (the default) it then becomes the benchmark; `--no-promote` only reports.

Every challenge is appended to models/benchmark_history.jsonl; model files are not touched.

    python -m bot.benchmark show
    python -m bot.benchmark set models/opponent_pool/<name>.pt  # start / reset the ladder
    python -m bot.benchmark set models/<label>                 # (a run folder works too)
    python -m bot.benchmark challenge models/<a> models/<b> --games 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import registry
from .evaluate import evaluate, load_evaluator, model_name

MIN_GAMES = 40
FAIR = 0.5   # pairwise score of an even match


def bench_file() -> Path:
    return registry.MODELS_DIR / "benchmark.json"


def history_file() -> Path:
    return registry.MODELS_DIR / "benchmark_history.jsonl"


def _store_ref(path: Path) -> str:
    """How benchmark.json refers to a model: relative to models/ or the repo root if possible."""
    path = path.resolve()
    for base in (registry.MODELS_DIR, registry.ROOT):
        try:
            return path.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def model_file(bench: dict | None = None) -> Path:
    """The benchmark's weights file."""
    bench = bench or current()
    if bench is None:
        raise FileNotFoundError("no benchmark is set")
    return registry.find_model(bench["model"])


def current() -> dict | None:
    f = bench_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def set_benchmark(model: str | Path, reason: str = "set by hand", result: dict | None = None) -> dict:
    path = registry.find_model(model)
    prev = current()
    entry = {"label": registry.label_for(path), "model": _store_ref(path),
             "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason,
             "previous": prev["label"] if prev else None}
    bench_file().write_text(json.dumps(entry, indent=1), encoding="utf-8")
    _append_history({"event": "benchmark", **entry, "result": result})
    return entry


def _append_history(row: dict) -> None:
    with history_file().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def verdict(result: dict, min_games: int = MIN_GAMES) -> tuple[bool, str]:
    """Does this evaluation show the challenger consistently beating the benchmark?"""
    n, share, se = result["games"], result["pairwise"], result.get("pairwise_se") or 0.0
    seats_above = sum(1 for v in result["per_seat_pairwise"].values() if v > FAIR)
    if n < min_games or n % 4:
        return False, f"only {n} games (need a multiple of 4, at least {min_games})"
    if share - 2 * se <= FAIR:
        return False, f"pairwise {share:.3f} ± {2 * se:.3f} is not clearly above {FAIR}"
    if seats_above < 3:
        return False, f"above {FAIR} from only {seats_above} of 4 seats"
    return True, f"pairwise {share:.3f} ± {2 * se:.3f}, above {FAIR} from {seats_above}/4 seats"


def challenge(model: str | Path, games: int = MIN_GAMES, sims: int = 32, selection: str = "puct",
              opening_plies: int = 4, seed: int = 0, device: str = "cpu",
              promote: bool = True) -> dict:
    """Play `model` against the current benchmark; promote it if it wins consistently.
    With no benchmark yet, `model` becomes the benchmark."""
    path = registry.find_model(model)
    label = registry.label_for(path)
    bench = current()
    if bench is None:
        set_benchmark(path, "first benchmark")
        return {"challenger": label, "benchmark": None, "promoted": True,
                "reason": "no benchmark yet; this model is now the benchmark"}
    if bench["label"] == label:
        return {"challenger": label, "benchmark": label, "promoted": False,
                "reason": "already the benchmark"}
    result = evaluate(load_evaluator(path, device), games=games, sims=sims,
                      selection=selection, opponent=load_evaluator(model_file(bench), device),
                      opponent_name=model_name(model_file(bench)),
                      opening_plies=opening_plies, seed=seed)
    ok, why = verdict(result)
    out = {"challenger": label, "benchmark": bench["label"], "promoted": ok and promote,
           "reason": why, "result": result, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _append_history({"event": "challenge", **out})
    if ok and promote:
        set_benchmark(path, f"beat {bench['label']}: {why}", result)
    return out


def summary(out: dict) -> str:
    r = out.get("result") or {}
    line = f"{out['challenger']} vs {out['benchmark']}: "
    if r:
        line += (f"pairwise {r['pairwise']} (se {r.get('pairwise_se')}), W/D/L {r['wins_draws_losses']}, "
                 f"per seat {r['per_seat_pairwise']}, "
                 f"avg place {r['avg_place']} over {r['games']} games -> ")
    return line + ("PROMOTED" if out["promoted"] else "kept benchmark") + f" ({out['reason']})"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    s = sub.add_parser("set")
    s.add_argument("model")
    c = sub.add_parser("challenge", help="challenge in order; a promoted model faces the next one")
    c.add_argument("models", nargs="+")
    c.add_argument("--games", type=int, default=MIN_GAMES)
    c.add_argument("--sims", type=int, default=32)
    c.add_argument("--opening-plies", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="auto", help="auto (CUDA when available), cpu or cuda")
    c.add_argument("--no-promote", action="store_true", help="report only; keep the benchmark")
    args = ap.parse_args(argv)
    if args.cmd == "challenge":
        args.device = registry.resolve_device(args.device)
    if args.cmd == "show":
        print(json.dumps(current(), indent=1))
    elif args.cmd == "set":
        print(json.dumps(set_benchmark(args.model), indent=1))
    else:
        for m in args.models:
            print(summary(challenge(m, args.games, args.sims, opening_plies=args.opening_plies,
                                    seed=args.seed, device=args.device,
                                    promote=not args.no_promote)), flush=True)


if __name__ == "__main__":
    main()
