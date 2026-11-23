"""The pairing rule at LMUL L (Section IV-A): the scales of data group v_N .. v_{N+L-1} occupy the
group that starts at v_{N+L}; element i uses scale byte floor(i / E). Drawn for L = 2 with four
elements per register (Saturn: 32), for E = 1 and E = 4. Colors as in fig. layouts."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ACC, ACC_L = pf.COLORS["accent"], "#F5C9A8"
EPR, L, W, GAP = 4, 2, 1.0, 0.35          # elements per register, LMUL, box width, gap between registers

def xpos(i):                              # left edge of element i of a group
    return 2.6 + i * W + (i // EPR) * GAP

def panel(ax, E, title, labels):
    x0 = 0 if labels else 2.3
    ax.set_xlim(x0, 12.0); ax.set_ylim(-0.9, 4.35); ax.set_aspect("equal"); ax.axis("off")
    yd, ys, h = 2.0, 0.0, 0.62
    n = EPR * L
    for i in range(n):
        ax.add_patch(Rectangle((xpos(i), yd), W - 0.1, h, fc=ACC, ec="black", lw=0.5))
        ax.text(xpos(i) + (W - 0.1) / 2, yd + h / 2, f"${i}$", ha="center", va="center", color="white", fontsize=7)
        used = i < -(-n // E)
        ax.add_patch(Rectangle((xpos(i), ys), W - 0.1, h, fc=ACC_L if used else "white", ec="black" if used else "#9A9A9A", lw=0.5))
        if used:
            ax.text(xpos(i) + (W - 0.1) / 2, ys + h / 2, f"${i}$", ha="center", va="center", fontsize=7)
    for i in range(n):
        ax.plot([xpos(i) + 0.45, xpos(i // E) + 0.45], [yd, ys + h], color="black", lw=0.5)
    for r, (dl, sl) in enumerate([("$v_N$", "$v_{N+L}$"), ("$v_{N+L-1}$", "$v_{N+2L-1}$")]):
        xc = xpos(r * EPR) + (EPR * W - 0.1) / 2
        ax.text(xc, yd + h + 0.12, dl, ha="center", va="bottom", fontsize=7)
        ax.text(xc, ys - 0.12, sl, ha="center", va="top", fontsize=7)
    if labels:
        ax.text(2.35, yd + h / 2, "data\ngroup", ha="right", va="center", fontsize=7, linespacing=0.95)
        ax.text(2.35, ys + h / 2, "scale\ngroup", ha="right", va="center", fontsize=7, linespacing=0.95)
    ax.text(xpos(0), 4.3, f"{title} $E{{=}}{E}$", ha="left", va="top")

pf.use()
fig, (a, b) = pf.figure("single", height=1.5, ncols=2, gridspec_kw={"wspace": 0.02, "width_ratios": [12.0, 9.7]})
panel(a, 1, "(a)", True); panel(b, 4, "(b)", False)
fig.subplots_adjust(left=0.01, right=0.99)
print(pf.save(fig, os.path.join(HERE, "out", "pairing")))
