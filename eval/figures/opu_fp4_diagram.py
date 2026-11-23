"""FP4 pairs in one OPU cell (Section IV-C): each byte holds two steps; the four codes are decoded to
integers four times their value, their two-term dot product is formed exactly as a small integer,
converted exactly to the cell's 10-bit-exponent format and fed into the scaled FP8 path in place of a,
with b's data replaced by 1. Blocks name the operation, arrows carry its result. Colors as in fig.
opuunit: FP4-only logic light, the scaled multiply-add path blue."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = to_rgba(pf.COLORS["delivery"], 0.35)     # FP4-only logic
ARITH = pf.COLORS["arithmetic"]                 # the cell's scaled path, shared with FP8
DATA = "#D9D9D9"

pf.use()
fig, ax = pf.figure("single", height=2.95)
ax.set_xlim(0, 14); ax.set_ylim(0, 11.2); ax.axis("off")

def box(x, y, w, h, fc, lw=0.5, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 6.5)
    ax.text(x, y, s, va="center", **kw)

def block(x, y, w, h, fc, lines):
    box(x, y, w, h, fc)
    n = len(lines)
    for i, s in enumerate(lines):
        text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * 0.28, s, ha="center")

def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", ls=ls,
                                                         shrinkA=0, shrinkB=0, mutation_scale=6))

def path(xs, ys, ls="-"):
    ax.plot(xs[:-1], ys[:-1], color="black", lw=0.6, ls=ls)
    arrow(xs[-2], ys[-2], xs[-1], ys[-1], ls)

# inputs: one byte of each operand, step k in bits 3:0, step k+1 in bits 7:4
for y, n in ((9.5, "a"), (8.1, "b")):
    box(0.2, y, 1.2, 0.7, DATA); box(1.4, y, 1.2, 0.7, DATA)
    text(0.8, y + 0.35, f"${n}_{{k+1}}$", ha="center")
    text(2.0, y + 0.35, f"${n}_k$", ha="center")
    arrow(2.6, y + 0.35, 4.4, y + 0.35)
text(0.8, 10.6, "7:4", ha="center", fontsize=5.5); text(2.0, 10.6, "3:0", ha="center", fontsize=5.5)
text(0.2, 7.65, "a byte of $v_{s1}$ (row $i$)", fontsize=6)
text(0.2, 7.2, "and of $v_{s2}$ (column $j$)", fontsize=6)

# row 1: decode, then the integer dot product
block(4.4, 8.05, 1.9, 2.2, NEW, ["decode"])
arrow(6.3, 9.15, 9.4, 9.15)
text(7.95, 9.95, "$4a_k,\\ 4a_{k+1},$", ha="center")
text(7.95, 9.5, "$4b_k,\\ 4b_{k+1}$", ha="center")
text(7.95, 8.75, "integers, $|\\cdot| \\leq 12$", ha="center", fontsize=6)
block(9.4, 8.05, 3.6, 2.2, NEW, ["two integer ×,", "one +"])
arrow(11.2, 8.05, 11.2, 6.1)
text(11.4, 7.35, "$16P$, integer,", ha="left")
text(11.4, 6.85, "$|16P| \\leq 288$", ha="left")

# row 2: exact conversion, then slot a's scaling
block(9.4, 4.9, 3.6, 1.2, NEW, ["integer → float,", "exact"])
arrow(9.4, 5.5, 7.1, 5.5)
text(8.25, 5.95, "$16P$, float", ha="center")
text(8.25, 5.05, "10-bit exp.", ha="center", fontsize=6)
block(3.5, 4.9, 3.6, 1.2, ARITH, ["slot $a$: scale,", "exp. $+\\,s^A_i - 4$ (÷16)"])
arrow(5.3, 4.9, 5.3, 3.6)
text(5.5, 4.25, "$s^A_i\\,P$", ha="left")

# row 3: slot b carries only B's scale; the exact multiply-add with the accumulator
block(9.4, 2.4, 3.6, 1.2, ARITH, ["slot $b$: data $= 1$,", "apply only $s^B_j$"])
arrow(9.4, 3.0, 7.1, 3.0)
text(8.25, 3.42, "$1 \\cdot s^B_j = s^B_j$", ha="center")
block(3.5, 2.4, 3.6, 1.2, ARITH, ["$a \\cdot b + c$,", "exact"])
block(0.2, 2.5, 2.1, 1.0, "white", ["$c$ (FP32)"])
arrow(2.3, 3.0, 3.5, 3.0)
arrow(5.3, 2.4, 5.3, 1.35)
text(5.5, 1.9, "$c + s^A_i\\,s^B_j\\,P$, exact", ha="left")

# rounding and the result
block(3.5, 0.15, 3.6, 1.2, ARITH, ["round once to", "FP32, nearest even"])
arrow(7.1, 0.75, 8.9, 0.75)
text(8.0, 1.08, "$C_{ij}$", ha="center", fontsize=7)

# a NaN among the four codes forces a NaN result
path([5.0, 5.0, 3.1, 3.1, 3.5], [8.05, 6.75, 6.75, 0.55, 0.55], ls=(0, (2, 1.5)))
text(3.0, 4.2, "NaN", ha="right", fontsize=6)

# definition and legend
text(9.4, 1.75, "$P = a_k b_k + a_{k+1} b_{k+1}$", ha="left", fontsize=6.5)
box(9.4, 0.85, 0.5, 0.35, NEW, lw=0.4); text(10.05, 1.02, "FP4 only", fontsize=6)
box(9.4, 0.25, 0.5, 0.35, ARITH, lw=0.4); text(10.05, 0.42, "shared with FP8", fontsize=6)

print(pf.save(fig, os.path.join(HERE, "out", "opufp4")))
