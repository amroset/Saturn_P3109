"""Magnitude ranges (log2, smallest subnormal to largest finite) of the values in a scaled
FP8 FMA and of the formats that could hold them: why the scaled lanes need a wider exponent."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
import matplotlib
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

def ieee(exp, sig):  # (smallest subnormal, largest finite) as log2, sig counts the hidden bit
    bias = 2 ** (exp - 1) - 1
    return 1 - bias - (sig - 1), bias + math.log2(2 - 2.0 ** (1 - sig))

# P3109 binary8p3 (extended domain, signed): widest 8-bit range, 2^-17 to 49152
P3 = (-17, math.log2(49152))
SCALE = (-127, 126)                     # Binary8p1uf codes 1..254: 2^(c-128)
OPND = (P3[0] + SCALE[0], P3[1] + SCALE[1])                            # a s_a
LANE_IN = (P3[0] + SCALE[0] - SCALE[1], P3[1] + SCALE[1] - SCALE[0])   # a s_a / s_r
FP32 = ieee(8, 24)
W = ieee(10, 8)

ROWS = [  # label, (lo, hi), kind
    ("FP8 element", P3, "value"),
    ("Scale factor", SCALE, "value"),
    (r"Scaled operand $a\,s_a$", OPND, "value"),
    (r"Lane input $a\,s_a/s_r$", LANE_IN, "value"),
    ("FP32/BF16 lane (8-bit exp.)", FP32, "lane"),
    ("Scaled lane (10-bit exp.)", W, "lane"),
]

pf.use()
matplotlib.rcParams["savefig.bbox"] = "standard"  # exactly column width
fig, ax = pf.figure("single", height=1.55)
fig.subplots_adjust(left=0.42, right=0.98, bottom=0.25, top=0.98)
for y, (label, (lo, hi), kind) in enumerate(reversed(ROWS)):
    fc = pf.COLORS["software"] if kind == "value" else pf.COLORS["arithmetic"]
    ax.add_patch(Rectangle((lo, y - 0.3), hi - lo, 0.6, fc=fc, ec="black", lw=0.3))
for x in FP32:
    ax.axvline(x, color="black", lw=0.5, ls=(0, (3, 2)), zorder=0)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in reversed(ROWS)])
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(-560, 560)
ax.set_xticks([-512, -256, 0, 256, 512])
ax.set_xlabel(r"Magnitude ($\log_2$)")
ax.spines["left"].set_visible(False)
print(pf.save(fig, os.path.join(HERE, "out", "ranges")))
