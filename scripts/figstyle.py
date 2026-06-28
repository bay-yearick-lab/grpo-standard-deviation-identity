"""Shared exhibit style for the figures."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY = "#1F4E79"   # GRPO
CLAY = "#B5654A"   # REINFORCE / Dr.GRPO
GREY = "#8C97A3"
INK = "#1A1A1A"
RULE = "#23527C"
GRID = "#E4E8EC"
AXIS = "#5A6470"


def use_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "text.color": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "figure.dpi": 150,
    })


def clean(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)


def header(fig, title, subtitle=None, x=0.065, top=0.965):
    # The floating in-figure action title and subtitle are intentionally not
    # drawn: each exhibit is labeled by its LaTeX caption instead. Kept as a
    # no-op so the existing call sites need no change.
    return None


def source(fig, text, x=0.065, y=0.018):
    # In-figure source/notes are intentionally not drawn: provenance lives in the
    # LaTeX caption and body. Kept as a no-op so call sites need no change.
    return None
