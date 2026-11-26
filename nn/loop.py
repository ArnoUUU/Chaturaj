"""Learning loop against the greedy baseline (nn/README.md).

Round 0 plays 160 games with an untrained network (every leaf a tie). Each later round trains
a fresh network on every position so far, then plays 160 new games with it: 40 new openings
(4 random plies) x the model in each of the 4 seats, against 3 greedy bots (depth 4).
The round's score against greedy is the mean pairwise score of 3 (1.5 = even).

    python nn/loop.py --rounds 8 --ms 50
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "engine/target/release/nnarena"
RUNS = ROOT / "nn/runs"


def arena(net, rnd, a, games=None, seed=None, out=None):
    out = out or RUNS / f"r{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "arena", "--net", str(net), "--games", str(games or a.games), "--opening-seed", str(rnd if seed is None else seed),
           "--ms", str(a.ms), "--threads", str(a.threads), "--width", str(a.width), "--qdepth", str(a.qdepth),
           "--out-games", str(out / "games.jsonl"), "--out-pos", str(out / "pos.bin")]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def train(rnd, a):
    data = [str(RUNS / f"r{i}" / "pos.bin") for i in range(rnd) if (RUNS / f"r{i}" / "pos.bin").exists()]
    if a.window:
        data = data[-a.window:]
    out = RUNS / f"r{rnd}" / "net.bin"
    cmd = [sys.executable, str(ROOT / "nn/train.py"), "--data", *data, "--out", str(out), "--gamma", str(a.gamma),
           "--aux", str(a.aux), "--h", str(a.h), "--epochs", str(a.epochs), "--patience", str(a.patience), "--seed", str(rnd)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out, json.loads(out.with_suffix(".json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--ms", type=float, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--qdepth", type=int, default=2)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--aux", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--window", type=int, default=0, help="train on the last N rounds only (0 = all)")
    a = ap.parse_args()
    log = RUNS / "loop.jsonl"
    for rnd in range(a.start, a.start + a.rounds):
        t = time.time()
        if rnd == 0:
            net, info = "zero", {}
        else:
            net, info = train(rnd, a)
        res = arena(net, rnd, a)
        row = {"round": rnd, "score_of_3": res["mean_pairwise_of_3"], "games": res["games"],
               "train_positions": info.get("positions"), "best_val": info.get("best_val"),
               "epochs": len(info.get("log", [])), "seconds": round(time.time() - t, 1), "args": vars(a)}
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
