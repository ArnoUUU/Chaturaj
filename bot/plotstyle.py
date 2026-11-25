"""Nature-style look for the bot's matplotlib figures.

Sans-serif type (Arial, else Helvetica, else DejaVu Sans), white background, thin black
spines, inward ticks on all four sides with minor ticks, a light grey major grid behind the
data, thin lines and small markers, a fixed colour order, frameless legends, and bold
lowercase panel letters (a, b, c, ...) with short bold titles.

Nothing is changed globally on import. Use `with style_context(): ...` for a figure, or
`apply()` to change matplotlib's defaults for the process. `panel()` and `finish()` set up
one axes; `finish()` also works on its own when the rc style is not active.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

BLUE = "#1f6fc5"
ORANGE = "#f28c28"
PURPLE = "#a61cbf"
GREEN = "#2a9d55"
GREY = "#7f7f7f"
RED = "#d62728"          # only for special events, e.g. a revert to the best weights
PALETTE = [BLUE, ORANGE, PURPLE, GREEN, GREY]
GRID = "#d9d9d9"
SUBTLE = "#555555"       # subtitles and secondary notes
PLACEHOLDER = "#7f7f7f"  # "not logged in this run"


def rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "errorbar.capsize": 2,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.loc": "best",
        "figure.facecolor": "white",
        "figure.titlesize": 9,
        "figure.titleweight": "bold",
        "figure.constrained_layout.use": True,
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    }


def apply() -> None:
    """Set the style as matplotlib's defaults for this process (explicit call only)."""
    plt.rcParams.update(rc())


def style_context():
    """Context manager: the style for the figures built inside the `with` block only."""
    return plt.rc_context(rc())


def panel(ax, letter: str, title: str) -> None:
    """Bold panel letter and short bold title on one baseline, above the top-left corner."""
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)   # baseline 4 pt above axes
    ax.annotate(letter, xy=(-0.13, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8, fontweight="bold", ha="left", va="baseline")


def blank(ax) -> None:
    """For a panel with no data: keep the frame, drop the ticks and tick labels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()


def resolved_font() -> str:
    """File name of the font matplotlib actually uses for the style (for checks)."""
    from matplotlib import font_manager
    return font_manager.findfont(font_manager.FontProperties(family=["Arial", "Helvetica", "DejaVu Sans"]))


def finish(ax) -> None:
    """Minor ticks on, inward ticks on all four sides, black spines, grid behind the data."""
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, left=True, bottom=True)
    ax.tick_params(which="major", length=3.5, width=0.6, labelsize=7)
    ax.tick_params(which="minor", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.6)
    ax.grid(False, which="minor")
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
