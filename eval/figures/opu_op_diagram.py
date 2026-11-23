"""Scaled OPFMACC as software sees it (Section IV-A): (a) the outer product of a column a and a
row b into a 32x32 tile, element C_ij scaled by s^A_i and s^B_j; (b) an FP4 byte, two steps
that share their block's scales. Drawn with 8 of the 32 rows and columns."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, SCALE, HL = "#D9D9D9", pf.COLORS["accent"], pf.COLORS["arithmetic"]
N, C = 8, 0.55          # drawn rows/columns, cell size
I, J = 2, 5             # the element followed

pf.use()
fig, ax = pf.figure("single", height=2.05)
ax.set_xlim(0, 14); ax.set_ylim(0, 8.8); ax.axis("off")

def box(x, y, w, h, fc, lw=0.3, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 7)
    ax.text(x, y, s, va="center", **kw)

# ---- (a) ------------------------------------------------------------------------------
ax.text(0.0, 8.5, "(a)", fontsize=8, va="center")
TX, TY = 3.0, 1.35                     # tile origin (bottom left)
top = TY + N * C
row_y = lambda r: top - (r + 1) * C    # row r, counted from the top

# tile, with the followed element
box(TX, TY, N * C, N * C, "white", lw=0.6)
for g in range(1, N):
    ax.plot([TX + g * C] * 2, [TY, top], color="#BBBBBB", lw=0.25)
    ax.plot([TX, TX + N * C], [TY + g * C] * 2, color="#BBBBBB", lw=0.25)
box(TX + J * C, row_y(I), C, C, HL, lw=0.8)
text(TX + N * C / 2, TY - 0.4, r"$32\times32$ FP32 tile", ha="center")

# a and its row scales (v_s1, v_s1+1): one per tile row
for r in range(N):
    box(TX - 0.8, row_y(r), C, C, DATA, lw=0.8 if r == I else 0.3)
    box(TX - 1.5, row_y(r), C, C, SCALE, lw=0.8 if r == I else 0.3)
text(TX - 0.8 + C / 2, top + 0.35, r"$a$", ha="center")
text(TX - 1.5 + C / 2, top + 0.35, r"$s^A$", ha="center")
text(TX - 1.6, row_y(I) + C / 2, r"$i$", ha="right")

# b and its column scales (v_s2, v_s2+1): one per tile column
BY = top + 0.95
for c in range(N):
    box(TX + c * C, BY, C, C, DATA, lw=0.8 if c == J else 0.3)
    box(TX + c * C, BY + 0.7, C, C, SCALE, lw=0.8 if c == J else 0.3)
text(TX + N * C + 0.15, BY + C / 2, r"$b$", ha="left")
text(TX + N * C + 0.15, BY + 0.7 + C / 2, r"$s^B$", ha="left")
text(TX + J * C + C / 2, BY + 1.65, r"$j$", ha="center")

# the element's two inputs
ax.plot([TX - 0.8 + C, TX + J * C], [row_y(I) + C / 2] * 2, color="black", lw=0.5, ls=(0, (2, 1.5)))
ax.plot([TX + J * C + C / 2] * 2, [BY, row_y(I) + C], color="black", lw=0.5, ls=(0, (2, 1.5)))
text(TX + N * C / 2, TY - 1.0, r"$C_{ij}$ += $s^A_i a_i \cdot s^B_j b_j$", ha="center")

# ---- (b) ------------------------------------------------------------------------------
ax.text(9.3, 8.5, "(b)", fontsize=8, va="center")
BX, BYB, BW, BH = 9.5, 5.2, 2.0, 0.75
box(BX, BYB, BW, BH, DATA, lw=0.5)
box(BX + BW, BYB, BW, BH, DATA, lw=0.5)
text(BX + BW / 2, BYB + BH / 2, r"step $k{+}1$", ha="center")
text(BX + 1.5 * BW, BYB + BH / 2, r"step $k$", ha="center")
text(BX + BW / 2, BYB + BH + 0.35, "bits 7:4", ha="center", fontsize=6.5)
text(BX + 1.5 * BW, BYB + BH + 0.35, "bits 3:0", ha="center", fontsize=6.5)
text(BX + BW, BYB - 0.45, "one FP4 byte of $a$ or $b$:", ha="center")
text(BX + BW, BYB - 1.2, "both steps share", ha="center")
text(BX + BW, BYB - 1.9, "their block's scale", ha="center")

print(pf.save(fig, os.path.join(HERE, "out", "opuop")))
