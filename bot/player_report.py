"""One clean scrape log and one chart per scraped player.

    python -m bot.player_report arnav2007 Ihavethesauce taran0915
    python -m bot.player_report --all
    python -m bot.player_report --all --prune     # then delete the merged raw logs

For each player writes
  logs/players/<player>.log   every logs/scrape_<player>*.log run, oldest first, under a
                              "===== <file> | <date> | N games =====" header, keeping only
                              progress lines (saved games, archive summaries, run totals).
                              Tracebacks, retry notices and browser stack traces are dropped.
                              Runs already merged (same header file name) are not repeated,
                              so this can be re-run after --prune removed the raw files.
  logs/players/<player>.png   games by time control, plus the training curves of that
                              player's best warm-start model (models/*/meta.json with
                              data.player or args.player = <player>)
Training logs (logs/supervised_*.log, logs/<label>/) are left alone.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = LOGS / "players"
PLAYERS = ROOT / "scraper" / "data" / "players"
RUNS = ROOT / "logs" / "models"

GAME_LINE = re.compile(r"^\s+\S+: \d+ \(\d+ plies\) .*\[\d+/\d+\]\s*$")   # "  1+5: 113... (153 plies) ok [1/20]"
GAMES_CMD = re.compile(r"^\[\d+/\d+\] \d+")                                 # "[1/3] 113...: ok"
HEADER_RE = re.compile(r"^===== (\S+) ")


def scrape_logs(player: str) -> list[Path]:
    p = player.lower()
    files = [f for f in LOGS.glob("scrape_*.log")
             if f.stem.lower() == f"scrape_{p}" or f.stem.lower().startswith(f"scrape_{p}_")]
    return sorted(files, key=lambda f: (f.stat().st_mtime, f.name))


def clean_lines(text: str) -> list[str]:
    """Only the progress lines of a scrape log; run totals without zero counts."""
    out = []
    for line in text.splitlines():
        line = line.rstrip()
        if GAME_LINE.match(line) or GAMES_CMD.match(line):
            out.append(line)
        elif line.startswith("[") and ("archive rows read" in line or " window " in line):
            out.append(line)
        elif line.startswith("{") and line.endswith("}"):
            try:
                totals = {k: v for k, v in ast.literal_eval(line).items() if v}
            except (ValueError, SyntaxError):
                continue
            out.append(f"games saved: {totals}" if totals else "games saved: none (no new matching games)")
    return out


def combine_logs(player: str) -> Path | None:
    """Append every not-yet-merged scrape log of `player`, cleaned, to logs/players/<player>.log."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{player.lower()}.log"
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    merged = {m.group(1) for m in map(HEADER_RE.match, existing.splitlines()) if m}
    new = [f for f in scrape_logs(player) if f.name not in merged]
    if not new and not existing:
        return None
    with out.open("a", encoding="utf-8") as w:
        for f in new:
            lines = clean_lines(f.read_text(encoding="utf-8", errors="replace"))
            n = sum(1 for l in lines if GAME_LINE.match(l) or GAMES_CMD.match(l))
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            w.write(f"===== {f.name} | {date} | {n} games =====\n")
            w.write("".join(l + "\n" for l in lines) or "(no games saved)\n")
    return out


def prune_raw_logs(player: str) -> list[Path]:
    """Delete raw scrape logs that are already merged into logs/players/<player>.log."""
    out = OUT / f"{player.lower()}.log"
    if not out.exists():
        return []
    merged = {m.group(1) for m in map(HEADER_RE.match, out.read_text(encoding="utf-8").splitlines()) if m}
    gone = [f for f in scrape_logs(player) if f.name in merged]
    for f in gone:
        f.unlink()
    return gone


def games_by_tc(player: str) -> list[tuple[str, int]]:
    counts = []
    for idx in (PLAYERS / player.lower()).glob("*/index.jsonl"):
        rows = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            counts.append((rows[0].get("time_control") or idx.parent.name, len(rows)))
    return sorted(counts, key=lambda c: -c[1])


def player_models(player: str) -> list[dict]:
    out = []
    for meta_path in RUNS.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        who = (meta.get("data") or {}).get("player") or (meta.get("args") or {}).get("player")
        if meta.get("training") == "supervised" and who and who.lower() == player.lower() \
                and meta.get("history"):
            out.append(meta)
    return sorted(out, key=lambda m: m["label"])


def best_val_loss(meta: dict) -> float:
    losses = [h["val"].get("policy_loss") for h in meta["history"] if h.get("val")]
    return min((l for l in losses if l is not None), default=float("inf"))


def chart(player: str, out: Path | None = None) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    from . import plotstyle as ps

    tcs = games_by_tc(player)
    models = player_models(player)
    bench_file = RUNS / "benchmark.json"
    bench_now = json.loads(bench_file.read_text(encoding="utf-8")) if bench_file.exists() else None
    if not tcs and not models:
        return None
    ncols = 1 + 2 * bool(models)
    with ps.style_context():
        fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols + 1.2, 3.9),
                                 gridspec_kw={"width_ratios": [1.1] + [1] * (ncols - 1)})
        axes = list(axes) if ncols > 1 else [axes]
        letters = iter("abc")
        for i, ax in enumerate(axes):
            if i:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # a. games per time control (single series -> no legend)
        ax = axes[0]
        labels = [t for t, _ in tcs][::-1]
        values = [n for _, n in tcs][::-1]
        ax.barh(labels, values, color=ps.BLUE, height=0.6, edgecolor="white", linewidth=0.5)
        ps.finish(ax)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, max(len(labels), 4) - 0.4)   # keep bars thin when there are few
        for y, v in enumerate(values):
            ax.text(v, y, f" {v}", va="center", color="black", fontsize=7)
        ax.set_xlim(0, max(values or [1]) * 1.15)
        ps.panel(ax, next(letters), f"Games scraped: {sum(values)}")
        ax.set_xlabel("Games")

        if models:
            scored = [x for x in models if "pairwise" in (x.get("evaluation") or {})]
            # newest model scored with the pairwise rule; else the lowest validation loss
            m = max(scored, key=lambda x: x["label"]) if scored else min(models, key=best_val_loss)
            hist = m["history"]
            ep = [h["epoch"] + 1 for h in hist]
            best = m.get("best_epoch")

            # b. policy loss: train vs validation
            ax = axes[1]
            tr = [h["train_loss"]["policy"] for h in hist]
            va = [h["val"].get("policy_loss") for h in hist]
            ax.plot(ep, tr, color=ps.BLUE, label="Train")
            ax.plot(ep, va, color=ps.ORANGE, label="Validation")
            if best is not None:
                ax.axvline(best + 1, color=ps.RED, linewidth=0.8, linestyle="--")
                lo, hi = ax.get_ylim()
                ax.text(best + 1, hi - 0.05 * (hi - lo), f" kept: epoch {best + 1}", color="black",
                        va="top", fontsize=7)
            ax.set_xlim(0.5, ep[-1] + max(1.5, ep[-1] * 0.25))
            ax.legend(loc="best")
            ps.panel(ax, next(letters), "Policy loss (move prediction)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (cross-entropy)")
            ps.finish(ax)

            # c. validation top-1 move accuracy (single series)
            ax = axes[2]
            top1 = [100 * h["val"].get("top1", 0) for h in hist]
            ax.plot(ep, top1, color=ps.BLUE, marker="o", markersize=3)
            i = max(range(len(top1)), key=top1.__getitem__)
            ax.annotate(f"{top1[i]:.1f}%", (ep[i], top1[i]), textcoords="offset points",
                        xytext=(4, 4), ha="left", va="bottom", color="black", fontsize=7)
            ax.set_ylim(0, max(top1) * 1.25)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ps.panel(ax, next(letters), "Validation: played move ranked first")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Positions (%)")
            ps.finish(ax)
            d = m["data"]
            r = m.get("evaluation") or {}
            opp = (m.get("benchmark_eval") or {}).get("benchmark") or r.get("opponent")
            is_bench = (bench_now or {}).get("label") == m["label"]
            sub = (f"warm start {m['label']}  ·  {d['games']} games ({d['train_positions']:,} train / "
                   f"{d['val_positions']:,} val positions)")
            if "pairwise" in r:
                w, dr, l = r["wins_draws_losses"]
                sub += (f"\nvs {opp} ({r['games']} games, every seat): pairwise score "
                        f"{r['pairwise']:.3f} ± {2 * (r.get('pairwise_se') or 0):.3f} (fair 0.5)  ·  "
                        f"1v1 results W {w} / D {dr} / L {l}")
            if is_bench:
                sub += "\ncurrent benchmark model"
            sub = "\n".join(textwrap.fill(line, 120) for line in sub.split("\n"))
            fig.text(0.01, 0.005, sub, color=ps.SUBTLE, fontsize=7, ha="left", va="bottom")
        else:
            ps.finish(axes[0])

        fig.suptitle(player, x=0.01, ha="left", fontsize=9, fontweight="bold")
        fig.tight_layout(rect=(0, 0.14, 1, 0.95))
        out = Path(out) if out else OUT / f"{player.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("players", nargs="*")
    ap.add_argument("--all", action="store_true", help="every player under scraper/data/players")
    ap.add_argument("--prune", action="store_true",
                    help="delete raw logs/scrape_<player>*.log files once merged")
    args = ap.parse_args(argv)
    players = args.players or []
    if args.all:
        players += [p.name for p in sorted(PLAYERS.iterdir()) if p.is_dir()]
    for player in dict.fromkeys(p.lower() for p in players):
        print(player, "log:", combine_logs(player), "chart:", chart(player))
        if args.prune:
            print("  removed:", ", ".join(f.name for f in prune_raw_logs(player)) or "nothing")


if __name__ == "__main__":
    main()
