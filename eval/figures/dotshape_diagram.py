"""What one dot-product instruction reads of y = W x (Section IV-A), with scales paired: (a) Zvdota,
a register of 32 consecutive k of one row at SEW 8, one scale per instruction; (b) vfdot4, four k
of each of 8 rows per 32-bit element at SEW 32, eight scales kept for the block's eight
instructions. One block of 32 along k, VLEN 256."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
READ, SCALE, IDLE = "#A9CCE8", pf.COLORS["accent"], pf.COLORS["unused"]
N, CW, CH = 8, 0.5, 0.42      # rows and 4-value cells drawn, cell size

pf.use()
fig, ax = pf.figure("single", height=2.75)
ax.set_xlim(0, 14); ax.set_ylim(0, 10.6); ax.axis("off")

def box(x, y, w, h, fc, lw=0.3, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 7)
    ax.text(x, y, s, va="center", **kw)

def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="black",
                                                         shrinkA=0, shrinkB=0, mutation_scale=5))

def panel(px, label, per_element):
    x0, top = px + 1.3, 9.3
    W = N * CW
    ax.text(px, 10.25, label, fontsize=8, va="center")
    reads = (lambda r, c: c == 0) if per_element else (lambda r, c: r == 0)

    # W: one block of 32 k (8 cells of 4 values) for 8 rows
    for r in range(N):
        for c in range(N):
            box(x0 + c * CW, top - (r + 1) * CH, CW, CH, READ if reads(r, c) else "white", lw=0.25)
    box(x0, top - N * CH, W, N * CH, "none", lw=0.6)
    arrow(x0, top + 0.3, x0 + W, top + 0.3)
    text(x0 + W + 0.1, top + 0.3, "$k$", ha="left")
    text(x0 - 0.12, top - CH / 2, "0", ha="right", fontsize=6)
    text(x0 - 0.12, top - N * CH + CH / 2, "7", ha="right", fontsize=6)
    text(x0 - 0.55, top - N * CH / 2, "row", ha="center", rotation=90, fontsize=6.5)
    arrow(x0 + W / 2, top - N * CH - 0.1, x0 + W / 2, 5.3)

    # data register: 32 FP8 values, grouped into 32-bit elements in (b)
    ry, sy = 4.7, 3.95
    for v in range(32):
        box(x0 + v * CW / 4, ry, CW / 4, 0.5, READ, lw=0.15)
    if per_element:
        for e in range(N):
            box(x0 + e * CW, ry, CW, 0.5, "none", lw=0.6)
    box(x0, ry, W, 0.5, "none", lw=0.6)
    text(x0 - 0.12, ry + 0.25, r"$v_{s2}$", ha="right")
    text(x0 + W + 0.15, ry + 0.25, "SEW 32" if per_element else "SEW 8", ha="left", fontsize=6.5)

    # paired scale register: byte floor(i/E) for element i
    used = N if per_element else 1
    for b in range(32):
        box(x0 + b * CW / 4, sy, CW / 4, 0.5, SCALE if b < used else IDLE, lw=0.15)
    box(x0, sy, W, 0.5, "none", lw=0.6)
    text(x0 - 0.12, sy + 0.25, r"$v_{s2+1}$", ha="right")
    text(x0 + W + 0.15, sy + 0.25, "$E = 1$" if per_element else "$E = 32$", ha="left", fontsize=6.5)
    text(x0 + used * CW / 4 + 0.1, sy - 0.32, "8 row scales" if per_element else "1 scale",
         ha="left" if per_element else "left", fontsize=6.5)

    # eight consecutive instructions; a dot marks each one that needs new scales loaded
    text(x0 + W / 2, 2.95, "8 consecutive instructions:", ha="center", fontsize=6.5)
    for n in range(N):
        box(x0 + n * CW, 2.15, CW, 0.45, "white", lw=0.3)
        text(x0 + n * CW + CW / 2, 2.375, str(n + 1), ha="center", fontsize=5.5)
        if n == 0 or not per_element:
            ax.plot(x0 + n * CW + CW / 2, 1.85, "o", ms=2.2, mfc=SCALE, mec="black", mew=0.3)
    lines = (["new scales loaded once,", "reused by the next 7"] if per_element
             else ["new scales loaded", "for every instruction"])
    for n, s in enumerate(lines):
        text(x0 + W / 2, 1.2 - n * 0.6, s, ha="center")

panel(0.0, "(a) Zvdota", False)
panel(7.25, "(b) vfdot4", True)

print(pf.save(fig, os.path.join(HERE, "out", "dotshape")))
