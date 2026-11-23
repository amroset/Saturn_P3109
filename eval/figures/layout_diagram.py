"""Inter-block (across, E = 1) and intra-block (along, E = B) register layouts: the data matrix A
(8 rows x 16 k, blocks of B = 8), its scale matrix S, the slices one instruction loads, and which
scale byte each register element uses (element i -> byte i >> log2 E). Schematic sizes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R, K, B = 8, 16, 8
ACC, ACC_L = pf.COLORS["accent"], "#F5C9A8"
AY, SX = 5.6, 18.2              # bottom of A and S; x of S
CW = 1.75                       # register box pitch

def cell(ax, x, y, fc="white", w=1, h=0.62, lw=0.3, ec="#9A9A9A"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))

def draw(ax, mode):
    ax.set_xlim(-1.6, 21.3); ax.set_ylim(-1.15, AY + R * 0.62 + 1.05); ax.set_aspect("equal"); ax.axis("off")
    sel_a = (lambda r, c: c == 0) if mode == "across" else (lambda r, c: r == 3 and c < B)
    sel_s = (lambda r, c: c == 0) if mode == "across" else (lambda r, c: r == 3 and c == 0)
    top = AY + R * 0.62
    for r in range(R):
        y = top - (r + 1) * 0.62
        if r in (0, R - 1) or (mode == "along" and r == 3):
            ax.text(-0.25, y + 0.31, str(r), ha="right", va="center", fontsize=7)
        for c in range(K):
            cell(ax, c, y, ACC if sel_a(r, c) else "white")
        for c in range(2):
            cell(ax, SX + c, y, ACC_L if sel_s(r, c) else "white")
    ax.add_patch(Rectangle((0, AY), K, R * 0.62, fill=False, ec="black", lw=0.6))
    ax.plot([B, B], [AY, top], "k-", lw=1.0)
    ax.add_patch(Rectangle((SX, AY), 2, R * 0.62, fill=False, ec="black", lw=0.6))
    ax.text(K / 2, top + 0.15, "$A$: rows $\\times$ $K$, blocks of $B{=}8$", ha="center", va="bottom")
    ax.text(SX + 1, top + 0.15, "$S$", ha="center", va="bottom")
    ax.text(-1.25, top - 0.31 * R, "row", rotation=90, ha="center", va="center", fontsize=7)

    # registers: data (top) and its scales (bottom), 8 elements each
    x0, yd, ys = 2.0, 2.3, -0.7
    lg = 0 if mode == "across" else 3
    for i in range(8):
        x = x0 + i * CW
        cell(ax, x, yd, ACC, w=CW - 0.2, h=0.8, lw=0.5, ec="black")
        lab = f"$a_{{{i},0}}$" if mode == "across" else f"$a_{{3,{i}}}$"
        ax.text(x + (CW - 0.2) / 2, yd + 0.4, lab, ha="center", va="center", fontsize=7, color="white")
        used = mode == "across" or i == 0
        cell(ax, x, ys, ACC_L if used else "white", w=CW - 0.2, h=0.8, lw=0.5, ec="black" if used else "#9A9A9A")
        if used:
            lab = f"$s_{{{i},0}}$" if mode == "across" else "$s_{3,0}$"
            ax.text(x + (CW - 0.2) / 2, ys + 0.4, lab, ha="center", va="center", fontsize=7)
    for i in range(8):                       # element i uses byte i >> lg
        xs = x0 + i * CW + (CW - 0.2) / 2
        xb = x0 + (i >> lg) * CW + (CW - 0.2) / 2
        ax.plot([xs, xb], [yd, ys + 0.8], color="black", lw=0.5)
    ax.text(x0 - 0.3, yd + 0.4, "data\nregister", ha="right", va="center", fontsize=7, linespacing=0.95)
    ax.text(x0 - 0.3, ys + 0.4, "scale\nregister", ha="right", va="center", fontsize=7, linespacing=0.95)
    ax.text(x0 + 8 * CW + 0.1, (yd + ys + 0.8) / 2, f"$E{{=}}{1 << lg}$", ha="left", va="center")
    # loads: slice of A -> data register, slice of S -> scale register
    arr = dict(arrowstyle="-|>", lw=0.5, color="#555555", mutation_scale=6, shrinkA=1, shrinkB=1)
    src = (0.5, AY) if mode == "across" else (B / 2, top - 4 * 0.62)
    ax.annotate("", xy=(x0 + 0.2, yd + 0.8), xytext=src, arrowprops={**arr, "connectionstyle": "arc3,rad=0.2"})
    ax.annotate("", xy=(x0 + 8 * CW - 0.1, ys + 0.4), xytext=(SX + 0.5, AY if mode == "across" else top - 4 * 0.62),
                arrowprops={**arr, "connectionstyle": "angle,angleA=-90,angleB=0,rad=0"})

pf.use()
for mode in ("across", "along"):            # single-column versions (slides, standalone use)
    fig, ax = pf.figure("single", height=2.15)
    draw(ax, mode)
    pf.save(fig, os.path.join(HERE, "out", f"layout-{mode}"))
fig, (a, b) = pf.figure("double", height=2.15, ncols=2, gridspec_kw={"wspace": 0.06})
draw(a, "across"); draw(b, "along")
a.set_title("(a) Inter-block (across): one $k$ of every row", loc="left")
b.set_title("(b) Intra-block (along): consecutive $k$ of one row", loc="left")
print(pf.save(fig, os.path.join(HERE, "out", "layouts")))
