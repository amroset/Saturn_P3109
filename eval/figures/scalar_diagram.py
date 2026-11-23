"""How a dot product's scalar operand can carry its scale (Section IV-A): (a) in an FP register, a
SEW-32 instruction receives only a 32-bit FP value, so a scale in bits 39:32 does not arrive;
(b) VMXDOTP names a second FP register for the scale, a fifth register operand; (c) vfdot4.vx
packs four FP8 values and their scale into one integer register, which arrives whole."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, SCALE, UNUSED = "#D9D9D9", pf.COLORS["accent"], pf.COLORS["unused"]
X0, U, H = 3.6, 9.4 / 64, 0.7           # bar origin, width of one bit, bar height

pf.use()
fig, ax = pf.figure("single", height=2.45)
ax.set_xlim(0, 14); ax.set_ylim(0, 10.4); ax.axis("off")

def box(x, y, w, h, fc, lw=0.4, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 7)
    ax.text(x, y, s, va="center", **kw)

def field(y, hi, lo, fc, label, **kw):
    x = X0 + (63 - hi) * U
    box(x, y, (hi - lo + 1) * U, H, fc, **kw)
    text(x + (hi - lo + 1) * U / 2, y + H / 2, label, ha="center", fontsize=6.5)

def bits(y):
    for hi, lo in ((63, 40), (39, 32), (31, 0)):
        text(X0 + (63 - hi + (hi - lo + 1) / 2) * U, y + H + 0.25, f"{hi}:{lo}", ha="center", fontsize=5.5)

def bracket(y, hi, lo, label):
    x0, x1 = X0 + (63 - hi) * U, X0 + (64 - lo) * U
    yb = y + H + 0.75
    ax.plot([x0, x0, x1, x1], [yb - 0.15, yb, yb, yb - 0.15], color="black", lw=0.5)
    text((x0 + x1) / 2, yb + 0.3, label, ha="center", fontsize=6.5)

def packed(y, scale_fc=SCALE, unused="unused"):
    field(y, 63, 40, UNUSED, unused)
    field(y, 39, 32, scale_fc, "$s^B$")
    for j in range(4):
        field(y, 31 - 8 * j, 24 - 8 * j, DATA, f"$b_{3 - j}$")

# (a) FP register: only bits 31:0 reach a SEW-32 instruction
ya = 7.6
text(0.0, ya + H / 2 + 0.25, "(a) $f[rs1]$", ha="left", fontsize=7.5)
text(0.0, ya + H / 2 - 0.4, "FP register", ha="left", fontsize=6.5)
packed(ya)
bits(ya)
bracket(ya, 31, 0, "reaches the vector unit: a 32-bit FP value")
text(X0 + (63 - 35.5) * U, ya - 0.35, "lost", ha="center", fontsize=6.5)

# (b) VMXDOTP: the scale in a second FP register
yb = 4.45
text(0.0, yb + H / 2 + 0.25, "(b) VMXDOTP", ha="left", fontsize=7.5)
text(0.0, yb + H / 2 - 0.4, "two FP registers", ha="left", fontsize=6.5)
box(X0, yb, 5.8, H, DATA)
text(X0 + 2.9, yb + H / 2, "packed values", ha="center", fontsize=6.5)
text(X0 + 2.9, yb + H + 0.3, "$f[rs1]$", ha="center", fontsize=6.5)
box(X0 + 6.3, yb, 3.1, H, SCALE)
text(X0 + 7.85, yb + H / 2, "scale", ha="center", fontsize=6.5)
text(X0 + 7.85, yb + H + 0.3, "$f[rs3]$", ha="center", fontsize=6.5)
text(X0 + 4.7, yb - 0.4, "a fifth register operand: a 48- or 64-bit encoding", ha="center", fontsize=6.5)

# (c) vfdot4.vx: values and scale in one integer register, all 64 bits arrive
yc = 1.25
text(0.0, yc + H / 2 + 0.25, "(c) $x[rs1]$", ha="left", fontsize=7.5)
text(0.0, yc + H / 2 - 0.4, "integer register", ha="left", fontsize=6.5)
packed(yc, unused="ignored")
bits(yc)
bracket(yc, 63, 0, "reaches the vector unit whole: values and scale")
text(X0 + 4.7, yc - 0.4, "three register operands: a 32-bit encoding", ha="center", fontsize=6.5)

print(pf.save(fig, os.path.join(HERE, "out", "scalarop")))
