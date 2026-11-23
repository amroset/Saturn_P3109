"""Scale delivery and arithmetic in the OPU (Section IV-C): the shared scale buffer of whole
registers, filled through the two data ports, and one cell's datapath; design choices numbered
as in the caption."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = to_rgba(pf.COLORS["delivery"], 0.35)
ARITH = pf.COLORS["arithmetic"]
SCALE = pf.COLORS["accent"]
DATA = "#D9D9D9"

pf.use()
fig, ax = pf.figure("single", height=2.75)
ax.set_xlim(0, 12); ax.set_ylim(0, 9.4); ax.axis("off")

def box(x, y, w, h, fc="white", lw=0.6, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                         shrinkA=0, shrinkB=0, mutation_scale=6))

def text(x, y, s, size=6.5, **kw):
    ax.text(x, y, s, fontsize=size, va="center", **kw)

def choice(x, y, n):
    ax.add_patch(Circle((x, y), 0.22, fc="black", ec="none"))
    ax.text(x, y - 0.02, str(n), color="white", fontsize=5.5, ha="center", va="center", fontweight="bold")

# ---- top: register file -> scale buffer -> tiles -------------------------------------
ax.text(0.0, 9.15, "(a)", fontsize=8, va="center")
box(0.1, 5.3, 2.2, 3.3)
text(1.2, 8.15, "Register", ha="center", size=7); text(1.2, 7.75, "file", ha="center", size=7)
for y in (6.95, 6.0):
    arrow(2.3, y, 3.6, y)
text(2.95, 7.3, "2 ports", ha="center")
text(2.95, 5.6, "fills", ha="center")
choice(2.95, 5.05, 2)

box(3.6, 5.3, 3.6, 3.3, NEW)
text(5.4, 8.2, "scale buffer", ha="center", size=7)
for e in range(4):
    y = 7.55 - e * 0.5
    for b in range(16):
        box(3.85 + b * 0.19, y, 0.19, 0.38, SCALE, lw=0.15)
text(5.4, 5.6, "4 whole registers", ha="center")
choice(7.0, 8.35, 1)

for y, lab in ((7.3, r"$s^A_i$"), (6.0, r"$s^B_j$")):
    arrow(7.2, y, 8.4, y)
    text(7.8, y + 0.35, lab, ha="center")
box(8.4, 5.3, 3.5, 3.3)
for i in range(2):
    for j in range(2):
        box(8.7 + j * 1.45, 5.55 + i * 1.3, 1.3, 1.15, "white", lw=0.4)
        for g in range(1, 4):
            ax.plot([8.7 + j * 1.45 + g * 0.325] * 2, [5.55 + i * 1.3, 6.7 + i * 1.3], color="#BBBBBB", lw=0.25)
            ax.plot([8.7 + j * 1.45, 10.0 + j * 1.45], [5.55 + i * 1.3 + g * 0.29] * 2, color="#BBBBBB", lw=0.25)
text(10.15, 8.25, r"$2\times2$ tiles of $32\times32$", ha="center")

# ---- bottom: one cell -----------------------------------------------------------------
ax.text(0.0, 4.55, "(b)", fontsize=8, va="center")
text(1.0, 4.55, "one cell", ha="left")
box(0.4, 3.05, 0.75, 0.7, DATA, lw=0.4); text(0.775, 3.4, r"$a_i$", ha="center")
box(1.35, 3.05, 0.75, 0.7, DATA, lw=0.4); text(1.725, 3.4, r"$b_j$", ha="center")
arrow(2.1, 3.4, 2.8, 3.4)
box(2.8, 2.9, 3.3, 1.0, ARITH)
text(4.45, 3.6, "decode; add the", ha="center"); text(4.45, 3.2, "exponents of $s^A_i$, $s^B_j$", ha="center")
arrow(6.1, 3.4, 6.8, 3.4)
box(6.8, 2.9, 3.4, 1.0, ARITH)
text(8.5, 3.6, r"$a\,b + c$, exact,", ha="center"); text(8.5, 3.2, "10-bit exponent", ha="center")
choice(10.2, 3.9, 3)
arrow(8.5, 2.9, 8.5, 2.3)
box(6.8, 1.3, 3.4, 1.0, ARITH)
text(8.5, 1.8, "round once to binary32", ha="center", size=6)
arrow(6.8, 1.8, 6.1, 1.8)
box(4.4, 1.45, 1.7, 0.7, "white", lw=0.4); text(5.25, 1.8, r"$c$", ha="center")
ax.plot([5.25, 5.25], [2.15, 2.55], color="black", lw=0.6)
ax.plot([5.25, 6.45], [2.55, 2.55], color="black", lw=0.6)
arrow(6.45, 2.55, 6.8, 3.05)
text(5.25, 1.1, "accumulator", ha="center")

# FP4: two steps per byte as a small integer dot product, in place of a (b = 1)
box(0.1, 0.1, 4.0, 1.6, NEW)
text(2.1, 1.2, "FP4: $16(a_k b_k + a_{k+1} b_{k+1})$,", ha="center", size=6)
text(2.1, 0.6, "an integer, in place of $a$", ha="center", size=6)
choice(4.1, 1.7, 4)
arrow(2.1, 1.7, 3.2, 2.9)

print(pf.save(fig, os.path.join(HERE, "out", "opuunit")))
