"""vfdot4.vx in scaled mode (E = 1), VLEN 256: (a) where data and scales sit and how they
reach the lanes, (b) what one FP32 lane computes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, SCALE, ACC, UNUSED = "#D9D9D9", pf.COLORS["accent"], "white", pf.COLORS["unused"]
ARITH = pf.COLORS["arithmetic"]
HL = 2  # the lane followed in (b)

pf.use()
fig, ax = pf.figure("double", height=2.55)
ax.set_xlim(0, 22.2); ax.set_ylim(0, 9.4); ax.axis("off")

def box(x, y, w, h, fc, lw=0.5, ec="black", **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))

def arrow(x0, y0, x1, y1, **kw):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                         shrinkA=0, shrinkB=0, mutation_scale=6, **kw))

def label(x, y, s, **kw):
    ax.text(x, y, s, va="center", **kw)

# ---- (a) placement and delivery -------------------------------------------------------
X0, LW = 2.6, 0.95          # register origin, lane width (32 bits)
ax.text(0.0, 9.15, "(a)", fontsize=9, va="center")

# vs2 = v8: eight FP32 lanes, each four FP8 values W[i, k..k+3]
yv = 7.3
label(X0 - 0.15, yv + 0.3, r"$v_{s2}$ = v8", ha="right")
for i in range(8):
    for j in range(4):
        box(X0 + i * LW + j * LW / 4, yv, LW / 4, 0.6, DATA, lw=0.3)
    box(X0 + i * LW, yv, LW, 0.6, "none", lw=1.2 if i == HL else 0.5)
    ax.text(X0 + i * LW + LW / 2, yv + 0.85, f"lane {i}" if i in (0, 7) else ("lane %d" % i if i == HL else ""),
            ha="center", va="center", fontsize=7)
label(X0 + 8 * LW + 0.15, yv + 0.3, r"$W[i,\,k..k{+}3]$", ha="left")

# v9 = vs2+1: 32 scale bytes, byte i scales lane i (E = 1); bytes 8..31 unused
ys = 5.6
label(X0 - 0.15, ys + 0.3, r"$v_{s2+1}$ = v9", ha="right")
BW = 8 * LW / 32
for b in range(32):
    box(X0 + b * BW, ys, BW, 0.6, SCALE if b < 8 else UNUSED, lw=0.3)
for i in range(8):
    ax.plot([X0 + i * LW + LW / 2, X0 + i * BW + BW / 2], [yv, ys + 0.6], color="black",
            lw=1.0 if i == HL else 0.35)
label(X0 + 8 * LW + 0.15, ys + 0.3, r"$s_W[i]$ = byte $i$", ha="left")

# scale buffer: holds v9 for the block's 8 instructions
ax.add_patch(FancyBboxPatch((X0 + 2.6, ys - 1.15), 2.0, 0.55, boxstyle="square,pad=0", fc="white",
                            ec="black", lw=0.5, ls=(0, (2, 1))))
ax.text(X0 + 3.6, ys - 0.875, "scale buffer", ha="center", va="center", fontsize=7)
arrow(X0 + 1.2, ys, X0 + 2.6, ys - 0.875)
label(X0 + 4.75, ys - 0.875, "read once per block,\nhit by its next 7 instructions", ha="left", fontsize=7)

# scalar register: x[k..k+3] in bits 31:0, its scale in bits 39:32
yf = 2.9
label(X0 - 0.15, yf + 0.3, r"$x[rs1]$", ha="right")
box(X0, yf, 1.2, 0.6, UNUSED, lw=0.3); ax.text(X0 + 0.6, yf + 0.3, "63:40", ha="center", va="center", fontsize=6)
box(X0 + 1.2, yf, 0.6, 0.6, SCALE, lw=0.3); ax.text(X0 + 1.5, yf + 0.3, r"$s_x$", ha="center", va="center", fontsize=7)
for j in range(4):
    box(X0 + 1.8 + j * 0.5, yf, 0.5, 0.6, DATA, lw=0.3)
box(X0 + 1.8, yf, 2.0, 0.6, "none", lw=0.5)
label(X0 + 3.95, yf + 0.3, r"$x[k..k{+}3]$, $s_x$: same for every lane", ha="left")

# vd: FP32 accumulators
yd = 1.2
label(X0 - 0.15, yd + 0.3, r"$v_d$", ha="right")
for i in range(8):
    box(X0 + i * LW, yd, LW, 0.6, ACC, lw=1.2 if i == HL else 0.5)
label(X0 + 8 * LW + 0.15, yd + 0.3, r"$y[i]$, FP32", ha="left")

# ---- (b) one lane ---------------------------------------------------------------------
B0 = 14.4
ax.text(B0 - 0.9, 9.15, "(b)", fontsize=9, va="center")
ax.text(B0 + 3.2, 9.15, f"lane {HL}", ha="center", va="center", fontsize=7)
# inputs: W and x elements, four multipliers
yi = 7.9
for j in range(4):
    cx = B0 + 0.4 + j * 1.6
    box(cx - 0.55, yi, 0.5, 0.5, DATA, lw=0.3); ax.text(cx - 0.3, yi + 0.25, r"$w_%d$" % j, ha="center", va="center", fontsize=7)
    box(cx + 0.05, yi, 0.5, 0.5, DATA, lw=0.3); ax.text(cx + 0.3, yi + 0.25, r"$x_%d$" % j, ha="center", va="center", fontsize=7)
    ax.add_patch(pf.plt.Circle((cx, 7.15), 0.28, fc=ARITH, ec="black", lw=0.5))
    ax.text(cx, 7.15, r"$\times$", ha="center", va="center")
    arrow(cx - 0.3, yi, cx - 0.12, 7.42); arrow(cx + 0.3, yi, cx + 0.12, 7.42)
    arrow(cx, 6.87, B0 + 2.8 + (j - 1.5) * 0.5, 6.3)

def stage(y, text, w=5.6):
    box(B0 + 2.8 - w / 2, y, w, 0.62, ARITH)
    ax.text(B0 + 2.8, y + 0.31, text, ha="center", va="center", fontsize=7)

stage(5.68, "exact fixed-point sum (69 bits)")
arrow(B0 + 2.8, 5.68, B0 + 2.8, 5.0)
stage(4.38, r"$\times\, s_W[i]\, s_x$: add the scale exponents")
arrow(B0 + 2.8, 4.38, B0 + 2.8, 3.7)
stage(3.08, r"$+\, y[i]$, exact")
arrow(B0 + 2.8, 3.08, B0 + 2.8, 2.4)
stage(1.78, r"round once, in frm $\rightarrow$ FP32")
arrow(B0 + 2.8, 1.78, B0 + 2.8, 1.2)
ax.text(B0 + 2.8, 0.95, r"$y[i]$", ha="center", va="center")
# side inputs: scales and accumulator
box(B0 - 1.1, 4.44, 0.5, 0.5, SCALE, lw=0.3); ax.text(B0 - 0.85, 4.69, r"$s_W$", ha="center", va="center", fontsize=6)
box(B0 + 6.7, 4.44, 0.5, 0.5, SCALE, lw=0.3); ax.text(B0 + 6.95, 4.69, r"$s_x$", ha="center", va="center", fontsize=6)
arrow(B0 - 0.6, 4.69, B0 - 0.0, 4.69); arrow(B0 + 6.7, 4.69, B0 + 5.6, 4.69)
box(B0 + 6.7, 3.14, 0.5, 0.5, ACC, lw=0.5); ax.text(B0 + 6.95, 3.39, r"$y$", ha="center", va="center", fontsize=6)
arrow(B0 + 6.7, 3.39, B0 + 5.6, 3.39)

print(pf.save(fig, os.path.join(HERE, "out", "dot4")))
