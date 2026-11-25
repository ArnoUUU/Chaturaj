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
5. Predator and prey: prey share and gang rate.
6. Mean game length (plies).

Older runs (before validation and learning-rate logging) simply leave those panels empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
STYLE = {"font.family": "Arial", "font.size": 9, "text.color": INK, "axes.edgecolor": GRID,
         "axes.labelcolor": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "xtick.color": INK2,
         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}


def _series(hist, key):
    pts = [(h["hours"], h[key]) for h in hist if h.get(key) is not None]
    return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])


def _config_line(meta) -> str:
    a = meta.get("args", {})
    bits = [f"depth {a.get('depth')}", f"q {a.get('qdepth')}", f"check ext {a.get('check_ext')}"]
    if a.get("width"):
        bits.append(f"width {a['width']} after {a.get('full_plies', 2)} plies")
    bits += [f"lr {a.get('lr')}", f"{a.get('games')} games x {a.get('steps')} steps x {a.get('batch')}",
             f"checks every {a.get('check_min')} min", f"{a.get('workers')} workers"]
    return " | ".join(str(b) for b in bits)


def plot_meta(meta: dict, out: str | Path) -> Path:
    hist = [h for h in meta.get("history", []) if "hours" in h]
    checks = meta.get("checks", [])
    reverts = [c["hours"] for c in checks if str(c.get("decision", "")).startswith("clearly worse")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        ax = axes[0, 0]
        for key, col, name in (("loss", BLUE, "training (mixed batches)"),
                               ("val_base_loss", ORANGE, "validation: held-out earlier games"),
                               ("val_selfplay_loss", AQUA, "validation: held-out self-play games")):
            x, y = _series(hist, key)
            if x:
                ax.plot(x, y, color=col, lw=2 if key != "loss" else 1.2, label=name)
        for i, r in enumerate(reverts):
            ax.axvline(r, color=RED, lw=1, ls="--", label="back to best weights" if i == 0 else None)
        ax.set_title("Loss (pairwise cross-entropy)", loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="best")

        ax = axes[0, 1]
        x, y = _series(hist, "lr")
        if x:
            ax.plot(x, y, color=BLUE, lw=2, drawstyle="steps-post")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        elif hist and meta.get("args", {}).get("lr"):     # older runs: constant, from the settings
            ax.plot([hist[0]["hours"], hist[-1]["hours"]], [meta["args"]["lr"]] * 2, color=BLUE, lw=2)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax.text(0.5, 0.6, "constant (from the run's settings)", ha="center", transform=ax.transAxes,
                    color=INK2, fontsize=8)
        else:
            ax.text(0.5, 0.5, "not logged in this run", ha="center", transform=ax.transAxes, color=MUTED)
        ax.set_title("Learning rate", loc="left", fontsize=10)

        ax = axes[0, 2]
        x, y = _series(hist, "grad_norm")
        if x:
            ax.plot(x, y, color=BLUE, lw=1.2)
        else:
            ax.text(0.5, 0.5, "not logged in this run", ha="center", transform=ax.transAxes, color=MUTED)
        ax.set_title("Gradient norm (before clipping at 1)", loc="left", fontsize=10)

        ax = axes[1, 0]
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        for key, col, name in (("vs_best", BLUE, "vs best weights (max^n)"), ("vs_mcts", ORANGE, "vs MCTS + MLP")):
            pts = [(c["hours"], c[key]["pairwise"], 2 * c[key]["pairwise_se"]) for c in checks if key in c]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=col, lw=2,
                            marker="o", ms=6, mec=SURF, mew=1.2, capsize=3, label=name)
        for c in checks:
            if str(c.get("decision", "")).startswith("clearly worse"):
                ax.annotate("revert", (c["hours"], c["vs_best"]["pairwise"]), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=7, color=INK2)
        ax.set_ylim(0, 1)
        ax.set_title("Checks: pairwise score ±2 SE", loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")

        ax = axes[1, 1]
        for key, col, name in (("prey_share", BLUE, "prey share"), ("gang_rate", ORANGE, "gang rate")):
            x, y = _series(hist, key)
            if x:
                ax.plot(x, y, color=col, lw=1.2, label=name)
        ax.axhline(0.25, color=MUTED, lw=1, ls=":")
        ax.text(0.01, 0.255, "0.25 = captures spread evenly", transform=ax.get_yaxis_transform(), fontsize=7,
                color=INK2, va="bottom")
        ax.set_ylim(0, 1)
        ax.set_title("Predator and prey", loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")

        ax = axes[1, 2]
        x, y = _series(hist, "mean_plies")
        if x:
            ax.plot(x, y, color=BLUE, lw=1.2)
        ax.set_title("Mean game length (plies)", loc="left", fontsize=10)

        for a in axes.flat:
            a.set_xlabel("hours", fontsize=8)
        fig.suptitle(f"{meta.get('label', '')}\n{_config_line(meta)}", fontsize=9, color=INK2, x=0.01, ha="left")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
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
