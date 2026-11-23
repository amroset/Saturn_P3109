"""Intra-block dequantization of 128 FP8 elements (E = 32, rule A) at LMUL 1 and LMUL 4:
the instruction stream, with the scale loads highlighted."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
KIND = {"ld": ("#BBBBBB", "", "Load data"), "sc": (pf.COLORS["accent"], "", "Load scales"),
        "cv": ("#444444", "", "Convert"), "st": ("white", "", "Store")}

pf.use()
fig, ax = pf.figure("single", height=1.0)
ax.set_xlim(-3.2, 21.0); ax.set_ylim(-2.3, 2.6); ax.axis("off")

def stream(y, seq):
    x = 0
    for kind, w in seq:
        fc, h, _ = KIND[kind]
        ax.add_patch(Rectangle((x, y), w - 0.12, 0.8, fc=fc, hatch=h, ec="black", lw=0.5))
        x += w
    return x

x1 = stream(1.3, [(k, 1) for _ in range(4) for k in ("ld", "sc", "cv", "st")])
ax.text(-0.3, 1.7, "LMUL 1", ha="right", va="center")
ax.text(x1 + 0.1, 1.7, "16 instr., 4 scale loads", ha="left", va="center")
x4 = stream(0.0, [("ld", 4), ("sc", 1), ("cv", 4), ("st", 4)])
ax.text(-0.3, 0.4, "LMUL 4", ha="right", va="center")
ax.text(x4 + 0.1, 0.4, "4 instr., 1 scale load", ha="left", va="center")
lx = 0.0
for k in ("ld", "sc", "cv", "st"):
    fc, h, lab = KIND[k]
    ax.add_patch(Rectangle((lx, -1.6), 0.7, 0.6, fc=fc, hatch=h, ec="black", lw=0.5))
    ax.text(lx + 0.9, -1.3, lab, va="center")
    lx += 0.9 + 0.5 * len(lab) + 1.0
print(pf.save(fig, os.path.join(HERE, "out", "lmul")))
