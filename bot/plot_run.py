"""Standard training-curves figure for a self-play NNUE run.

Drawn automatically by bot/selfplay_nnue.py at every check and at the end
(logs/<label>/curves.png, copied to models/<label>/curves.png). Also from the command line,
for a finished or running run:

    python -m bot.plot_run models/<label>            # or logs/<label>, or the label
    python -m bot.plot_run models/<label> -o out.png

Panels, all against hours of training (iterations from meta.json "history", checks from
"checks"):

1. Loss: training loss (mixed batches), validation loss on held-out earlier games, and on
   held-out self-play games. Reverts to the best weights are marked.
2. Learning rate.
3. Gradient norm (mean per iteration, before clipping at 1.0).
4. Checks: pairwise score ±2 SE vs the best weights and vs MCTS + MLP; 0.5 = even.
5. Mean game length (plies).
6. Mean ply of the first elimination.

Older runs (before validation and learning-rate logging) simply leave those panels empty.
Old meta.json files that still hold the retired predator/prey keys (prey_share, gang_rate)
are read fine: those keys are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import plotstyle as ps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PANEL_LETTERS = "abcdef"
SMOOTH_MIN_POINTS = 30    # longer per-iteration series get a raw line plus a moving average
SMOOTH_WINDOW = 9


def _series(hist, key):
    pts = [(h["hours"], h[key]) for h in hist if h.get(key) is not None]
    return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])


def _moving_average(y, window=SMOOTH_WINDOW):
    """Centred mean over `window` points, using fewer points at the ends (min_periods 1)."""
    half = window // 2
    return [sum(y[max(0, i - half): i + half + 1]) / len(y[max(0, i - half): i + half + 1])
            for i in range(len(y))]


def _curve(ax, x, y, color) -> bool:
    """Plot a series; long series get a thin raw line and a moving average on top."""
    if len(y) > SMOOTH_MIN_POINTS:
        ax.plot(x, y, color=color, lw=0.6, alpha=0.35, label="per iteration")
        ax.plot(x, _moving_average(y), color=color, lw=1.2, label=f"moving average ({SMOOTH_WINDOW})")
        return True
    ax.plot(x, y, color=color, lw=1.2)
    return False


def _config_line(meta) -> str:
    a = meta.get("args", {})
    bits = [f"depth {a.get('depth')}", f"q {a.get('qdepth')}", f"check ext {a.get('check_ext')}"]
    if a.get("width"):
        bits.append(f"width {a['width']} after {a.get('full_plies', 2)} plies")
    if a.get("discount", 1) < 1 or a.get("elim_bonus"):
        bits.append(f"discount {a.get('discount')}, elim bonus {a.get('elim_bonus')} to turn {a.get('elim_horizon')}")
    bits += [f"lr {a.get('lr')}", f"{a.get('games')} games x {a.get('steps')} steps x {a.get('batch')}",
             f"checks every {a.get('check_min')} min", f"{a.get('workers')} workers"]
    return " | ".join(str(b) for b in bits)


def _legend(ax, **kw) -> None:
    """Legend inside the axes, on an opaque white box so lines behind it (reverts) don't cut
    through the text; kept out of constrained layout so it cannot shrink the panels."""
    leg = ax.legend(frameon=True, framealpha=1.0, facecolor="white", edgecolor="none", **kw)
    leg.set_in_layout(False)


def _not_logged(ax) -> None:
    ax.text(0.5, 0.5, "not logged in this run", ha="center", va="center", transform=ax.transAxes,
            color=ps.PLACEHOLDER, fontsize=7)
    ps.blank(ax)


def plot_meta(meta: dict, out: str | Path) -> Path:
    hist = [h for h in meta.get("history", []) if "hours" in h]
    checks = meta.get("checks", [])
    reverts = [c["hours"] for c in checks if str(c.get("decision", "")).startswith("clearly worse")]
    hours = [h["hours"] for h in hist] + [c["hours"] for c in checks if "hours" in c]
    t_end = max(hours) if hours else 1.0
    with ps.style_context():
        fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
        # header row (run label and settings line), then the 2x3 panels
        gs = fig.add_gridspec(3, 3, height_ratios=[0.2, 1, 1])
        hdr = fig.add_subplot(gs[0, :])
        hdr.axis("off")
        hdr.text(0, 1.0, meta.get("label", ""), transform=hdr.transAxes, ha="left", va="top",
                 fontsize=9, fontweight="bold")
        # settings line directly under the title: 2 pt below the title's line box
        hdr.annotate(_config_line(meta), xy=(0, 1.0), xycoords="axes fraction", xytext=(0, -13),
                     textcoords="offset points", ha="left", va="top", fontsize=7, color=ps.SUBTLE)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in (1, 2)]
        letters = iter(PANEL_LETTERS)

        # a. loss
        ax = axes[0][0]
        for key, col, name in (("loss", ps.BLUE, "training"),
                               ("val_base_loss", ps.ORANGE, "validation: earlier games"),
                               ("val_selfplay_loss", ps.PURPLE, "validation: self-play")):
            x, y = _series(hist, key)
            if x:
                ax.plot(x, y, color=col, lw=1.2, label=name)
        for i, r in enumerate(reverts):
            ax.axvline(r, color=ps.RED, lw=0.8, ls="--", label="revert to best" if i == 0 else None)
        ps.panel(ax, next(letters), "Loss")
        ax.set_ylabel("Loss (cross-entropy)")
        if ax.get_legend_handles_labels()[0]:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 1.0 * (hi - lo))      # headroom above the data for the legend
            _legend(ax, loc="upper right")

        # b. learning rate: values scaled by a power of ten taken from the data
        ax = axes[0][1]
        x, y = _series(hist, "lr")
        if not x and hist and meta.get("args", {}).get("lr"):     # older runs: constant, from settings
            x, y = [hist[0]["hours"], hist[-1]["hours"]], [meta["args"]["lr"]] * 2
        if x:
            k = math.floor(math.log10(max(abs(v) for v in y))) if max(abs(v) for v in y) > 0 else 0
            scaled = [v / 10 ** k for v in y]
            flat = max(y) == min(y)
            ax.plot(x, scaled, color=ps.BLUE, lw=1.2, drawstyle="steps-post")
            ax.set_ylabel(f"Learning rate (×10$^{{{k}}}$)")
            ps.panel(ax, next(letters), "Learning rate (constant)" if flat else "Learning rate")
        else:
            _not_logged(ax)
            ps.panel(ax, next(letters), "Learning rate")
            ax.set_ylabel("Learning rate")

        # c. gradient norm
        ax = axes[0][2]
        x, y = _series(hist, "grad_norm")
        if x:
            if _curve(ax, x, y, ps.BLUE):
                _legend(ax, loc="upper right")
        else:
            _not_logged(ax)
        ps.panel(ax, next(letters), "Gradient norm")
        ax.set_ylabel("Gradient norm (before clipping at 1)")

        # d. checks
        ax = axes[1][0]
        ax.axhline(0.5, color=ps.GREY, lw=0.6, ls=":", label="even")
        for key, col, name in (("vs_best", ps.BLUE, "vs best (max^n)"), ("vs_mcts", ps.ORANGE, "vs MCTS + MLP")):
            pts = [(c["hours"], c[key]["pairwise"], 2 * c[key]["pairwise_se"]) for c in checks if key in c]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=col,
                            lw=1.2, marker="o", ms=3, mec=col, elinewidth=0.8, capsize=2, label=name)
        for i, r in enumerate(reverts):
            ax.axvline(r, color=ps.RED, lw=0.8, ls="--", label="revert" if i == 0 else None)
        ax.set_ylim(0, 1.45)                      # headroom above the data for the legend
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ps.panel(ax, next(letters), "Checks, ±2 SE")
        ax.set_ylabel("Pairwise score")
        if ax.get_legend_handles_labels()[0]:
            _legend(ax, loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.8, fontsize=6.5,
                    borderaxespad=0.3)

        # e. game length
        ax = axes[1][1]
        x, y = _series(hist, "mean_plies")
        if x:
            if _curve(ax, x, y, ps.BLUE):
                _legend(ax, loc="upper right")
        else:
            _not_logged(ax)
        ps.panel(ax, next(letters), "Game length")
        ax.set_ylabel("Game length (plies)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))   # headroom for the legend

        # f. first elimination
        ax = axes[1][2]
        x, y = _series(hist, "first_elim_ply")
        if x:
            ax.plot(x, y, color=ps.ORANGE, lw=1.2)
        else:
            _not_logged(ax)
        ps.panel(ax, next(letters), "First elimination")
        ax.set_ylabel("First elimination (ply)")

        for row in axes:
            for a in row:
                a.set_xlabel("Training time (h)")
                ps.finish(a)
                a.set_xlim(0, t_end)
        # blank panels: ps.finish() turns the ticks back on, so drop them again after it
        for row in axes:
            for a in row:
                if not a.get_lines() and not a.collections and not a.patches:
                    ps.blank(a)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    return out


def find_meta(target: str) -> Path:
    p = Path(target)
    for cand in (p / "meta.json", ROOT / "models" / target / "meta.json", ROOT / "models" / p.name / "meta.json"):
        if cand.is_file():
            return cand
    raise SystemExit(f"no meta.json for {target}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="models/<label>, logs/<label> or the label")
    ap.add_argument("-o", "--out", default=None, help="output png (default logs/<label>/curves.png)")
    a = ap.parse_args(argv)
    mp = find_meta(a.run)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    out = a.out or ROOT / "logs" / mp.parent.name / "curves.png"
    print(plot_meta(meta, out))


if __name__ == "__main__":
    main()
