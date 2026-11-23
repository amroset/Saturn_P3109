"""Helpers for the figures: style, sizes, one color per concept, saving."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Column widths in inches
WIDTH = {"single": 3.5, "double": 7.16, "acm-single": 3.33, "acm-double": 7.0}

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

# One color and hatch per recurring concept; add new concepts here, not in figure scripts
COLORS = {
    "unscaled": "#BBBBBB",
    "ruleA": OKABE_ITO["blue"],
    "buf4": OKABE_ITO["blue"],
    "explicit": OKABE_ITO["orange"],
    "scalar": OKABE_ITO["green"],
    "software": "#444444",
    "accent": OKABE_ITO["vermillion"],
    "arithmetic": OKABE_ITO["skyblue"],
    "delivery": OKABE_ITO["purple"],
    "unused": "#E5E5E5",
}
HATCH = {"unscaled": "", "ruleA": "", "buf4": "xxxx", "explicit": "////", "scalar": "....", "software": "", "accent": "",
         "arithmetic": "", "delivery": "\\\\\\\\", "unused": ""}
LABEL = {"unscaled": "Unscaled", "ruleA": "Rule A (pairing)", "buf4": "Pairing, larger buffer", "explicit": "Explicit (best case)",
         "scalar": "Scalar operand (est.)", "software": "Software"}


def use():
    plt.style.use(os.path.join(HERE, "paper.mplstyle"))


def figure(width="single", height=2.0, **kw):
    """A figure at final printed size: width 'single'/'double' (or inches), height in inches."""
    w = WIDTH.get(width, width)
    return plt.subplots(figsize=(w, height), **kw)


def save(fig, path):
    """Write path.pdf (for the paper) and path.png (preview to check)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path + ".pdf", metadata={"CreationDate": None, "ModDate": None})  # reproducible bytes
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    return path + ".pdf", path + ".png"
