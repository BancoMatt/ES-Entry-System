"""
src/plots.py  -  one visual system for every chart in the project.

Colours come from a validated palette (categorical slots in fixed order, a blue<->red
diverging scale with a neutral grey midpoint). Validated with the dataviz skill's
checker: CVD separation and normal-vision floors pass; three light slots sit under
3:1 contrast, so lines are always DIRECT-LABELLED and every chart has a CSV twin.

Conventions used everywhere:
  - blue = better than Buy & Hold, red = worse, grey = about the same
  - thin marks, hairline grid, no dashed gridlines, one y-axis per chart
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#f0efec"
# diverging: red arm (worse than B&H) <- grey -> blue arm (better), equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list(
    "worse_better", ["#8f2424", "#e34948", "#f4b3b2", NEUTRAL, "#9ec5f4", "#2a78d6", "#104281"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "legend.frameon": False,
        "legend.fontsize": 8, "lines.linewidth": 2.0,
    })


def diverging_norm(limit):
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def note(fig, text):
    """Small grey footnote at the bottom-left of a figure (source / caveat)."""
    fig.text(0.01, -0.03, text, fontsize=7, color=MUTED, ha="left", va="top")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
