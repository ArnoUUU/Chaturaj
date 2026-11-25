"""Summarize arena results: one row per results/*.jsonl file.

    python claude-playground/summarize.py [names...]

Score = the test player's mean pairwise score against the three base players (1 = beat all
three on final points, 0.5 = even). ±2 SE is over games. Depth and nodes are per-move means;
ms/move columns appear when the run logged times.
"""

import json
import math
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def row(name: str, rs: list[dict]) -> str:
    n = len(rs)
    s = [r["score"] for r in rs]
    m = sum(s) / n
    se = math.sqrt(sum((x - m) ** 2 for x in s) / (n - 1) / n) if n > 1 else 0.0
    mean = lambda k: sum(r[k] for r in rs) / n
    wins = sum(1 for x in s if x > 0.999)
    out = (f"| {name} | {n} | {m:.3f} ± {2 * se:.3f} | {wins / n:.2f} | {mean('test_depth'):.2f} | "
           f"{mean('base_depth'):.2f} | {mean('test_nodes'):.0f} | {mean('base_nodes'):.0f} | {mean('turns'):.0f} |")
    if "test_ms" in rs[0]:
        out += f" {mean('test_ms'):.0f} | {mean('base_ms'):.0f} |"
    return out


def main(names: list[str]) -> None:
    files = [RES / f"{n}.jsonl" for n in names] if names else sorted(RES.glob("*.jsonl"))
    print("| run | games | test score ±2 SE | test 1st place | test depth | base depth | test nodes | "
          "base nodes | turns | test ms | base ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in files:
        rs = load(f)
        if rs:
            print(row(f.stem, rs))


if __name__ == "__main__":
    main(sys.argv[1:])
