"""One scaled FMA lane (Section IV-B): the FP8 operands are decoded into the lane's format, whose exponent
is widened to 10 bits; each operand's scale (and the result's, for a and c) is added to its exponent;
the multiply-add is exact and the result is rounded once; a zero or NaN result scale is applied to the
rounded value. Circled numbers: the other places the scales could be applied (text). Colors as in
fig. vecunit: blue the arithmetic scaled mode adds, light blue what it widens, scale bytes vermillion.
RTL: SegmentedFMAPipe (laneType, scaleRec, MulAddRecFNPipeUnrounded, rawUnroundedToP3109)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
import matplotlib
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = pf.COLORS["arithmetic"]                       # added by scaled mode
WIDE = to_rgba(pf.COLORS["arithmetic"], 0.3)        # existing, widened to a 10-bit exponent
SCALE = pf.COLORS["accent"]                         # scale bytes
DATA = "#D9D9D9"

pf.use()
matplotlib.rcParams["savefig.bbox"] = "standard"    # exactly column width
fig, ax = pf.figure("single", height=2.75)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 14); ax.set_ylim(0, 11.2); ax.axis("off")

def box(x, y, w, h, fc, lw=0.5):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw))

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 6.5)
    ax.text(x, y, s, va="center", **kw)

def block(x, y, w, h, fc, lines, fs=6.5):
    box(x, y, w, h, fc)
    n = len(lines)
    for i, s in enumerate(lines):
        text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * 0.3, s, ha="center", fontsize=fs)

def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                         shrinkA=0, shrinkB=0, mutation_scale=6))

def line(xs, ys):
    ax.plot(xs, ys, color="black", lw=0.6)

def alt(x, y, n):                                   # an alternative placement of the scales
    ax.add_patch(Circle((x, y), 0.32, fc="white", ec="#777777", lw=0.5))
    text(x, y, str(n), ha="center", fontsize=5.5, color="#555555")

COLS = [(0.6, "a", "+\\,e_a - e_r"), (3.6, "b", "+\\,e_b"), (6.6, "c", "+\\,e_c - e_r")]
W = 2.5
for x, n, add in COLS:
    xc = x + W / 2
    block(x + 0.45, 10.0, W - 0.9, 0.8, DATA, [f"${n}$ (FP8)"])
    arrow(xc, 10.0, xc, 9.05)
    block(x, 7.85, W, 1.2, WIDE, ["decode,", "10-bit exp."])
    arrow(xc - 0.45, 7.85, xc - 0.45, 6.75)
    block(x, 5.55, W, 1.2, NEW, ["exponent", f"${add}$"])
    arrow(xc, 5.55, xc, 4.65)

# scale bytes, from the scale buffers, onto a bus above the exponent adders
block(10.2, 7.85, 3.4, 1.2, SCALE, ["scale bytes", "$s_a, s_b, s_c, s_r$"])
text(11.9, 9.45, "from the scale buffers", ha="center", fontsize=5.5)
xs = sorted([10.2, COLS[0][0] + W / 2 + 0.6] + [v for x, _, _ in COLS[1:] for v in (x + W / 2 - 0.6, x + W / 2 - 0.3)])
for x0, x1 in zip(xs[::2], xs[1::2]):              # broken where the data arrows cross it
    line([x0, x1], [7.35, 7.35])
for x, _, _ in COLS:
    arrow(x + W / 2 + 0.6, 7.35, x + W / 2 + 0.6, 6.75)

# exact multiply-add, single rounding, special result scales
block(0.6, 3.45, 8.5, 1.2, WIDE, ["$a\\,b + c$, exact (no rounding)"])
text(9.3, 4.05, "operands from", ha="left", fontsize=5.5)
text(9.3, 3.6, "$2^{-270}$ to $2^{269}$", ha="left", fontsize=5.5)
arrow(4.85, 3.45, 4.85, 2.55)
block(2.6, 1.95, 4.5, 0.6, WIDE, ["round once (frm)"])
arrow(4.85, 1.95, 4.85, 1.35)
block(2.6, 0.75, 4.5, 0.6, NEW, ["zero or NaN $s_r$"])
arrow(7.1, 1.05, 8.6, 1.05)
text(8.75, 1.05, "$Z$ (FP8 or BF16)", ha="left")
line([13.1, 13.1, 6.3], [7.85, 0.3, 0.3])           # s_r to the special-case logic
arrow(6.3, 0.3, 6.3, 0.75)

# where else the scales could go
alt(0.3, 9.45, 1)        # before the lane: on the FP8 values
alt(0.3, 3.0, 2)         # between the multiply-add and the rounding
alt(0.3, 1.65, 3)        # after the rounding

# legend
box(9.3, 2.45, 0.5, 0.32, NEW, lw=0.4); text(9.95, 2.61, "added", fontsize=6)
box(11.3, 2.45, 0.5, 0.32, WIDE, lw=0.4); text(11.95, 2.61, "widened", fontsize=6)

print(pf.save(fig, os.path.join(HERE, "out", "fmalane")))
