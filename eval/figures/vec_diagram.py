"""Scale delivery in the vector unit (Section IV-B): the path of a scaled instruction's scales
from dispatch to the lanes, with the design choices numbered as in the caption."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = to_rgba(pf.COLORS["delivery"], 0.35)     # new delivery logic
ARITH = pf.COLORS["arithmetic"]                 # new arithmetic
SCALE = pf.COLORS["accent"]                     # scale bytes
DASH = (0, (2.5, 1.5))

pf.use()
fig, ax = pf.figure("double", height=2.3)
ax.set_xlim(0, 24); ax.set_ylim(0, 7.7); ax.axis("off")

def box(x, y, w, h, fc="white", lw=0.6, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=lw, **kw))

def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", ls=ls,
                                                         shrinkA=0, shrinkB=0, mutation_scale=6))

def text(x, y, s, size=6.5, **kw):
    ax.text(x, y, s, fontsize=size, va="center", **kw)

def choice(x, y, n):
    ax.add_patch(Circle((x, y), 0.26, fc="black", ec="none"))
    ax.text(x, y - 0.02, str(n), color="white", fontsize=6, ha="center", va="center", fontweight="bold")

# dispatch: scale groups join the instruction's reads, so existing hazard checks cover them
box(0.2, 5.75, 6.4, 1.75)
text(0.45, 7.1, "Dispatch", size=7.5)
text(0.45, 6.5, "adds each block operand's scale", ha="left")
text(0.45, 6.1, "groups to the instruction's reads", ha="left")
choice(6.2, 7.05, 1)
arrow(3.4, 5.75, 3.4, 5.05)
text(3.55, 5.4, "existing hazard checks", ha="left")

# FP execute sequencer: per element group, hit or miss
box(0.2, 0.3, 6.4, 4.75)
text(0.45, 4.6, "FP sequencer, per element group", size=7.5)
box(0.5, 3.35, 5.8, 0.75, NEW)
text(3.4, 3.72, "scale groups in the buffers?", ha="center")
box(0.5, 1.05, 2.6, 1.5)
text(1.8, 2.05, "hit:", ha="center"); text(1.8, 1.55, "issue the data", ha="center")
box(3.7, 1.05, 2.6, 1.5, NEW)
text(5.0, 2.05, "miss: +1 cycle", ha="center"); text(5.0, 1.55, "to read scales", ha="center")
arrow(1.8, 3.35, 1.8, 2.55); arrow(5.0, 3.35, 5.0, 2.55)
text(1.65, 2.95, "yes", ha="right"); text(5.15, 2.95, "no", ha="left")
choice(6.0, 0.7, 2)
arrow(6.6, 2.6, 8.6, 2.6)
text(7.6, 2.95, "addresses", ha="center")

# register file: three read ports into the scale buffers
box(8.6, 0.3, 2.7, 4.75)
text(9.95, 4.6, "Register file", size=7.5, ha="center")
ports = [("$v_{s1}$", 3.75), ("$v_{s2}$", 2.75), ("$v_d$", 1.75)]
for name, y in ports:
    text(11.15, y, name, ha="right")
    arrow(11.3, y, 12.7, y)

# scale buffers: one register's scale groups per port, kept across instructions
for name, y in ports:
    box(12.7, y - 0.38, 3.4, 0.76, NEW)
    text(12.85, y, "buffer", ha="left")
    for b in range(8):
        box(14.25 + b * 0.2, y - 0.22, 0.2, 0.44, SCALE, lw=0.2)
text(14.4, 5.05, "1 register per port,", ha="center")
text(14.4, 4.65, "kept across instructions", ha="center")

# write broadcast: every write's address invalidates matching entries
ax.plot([11.3, 14.4], [0.75, 0.75], color="black", lw=0.6, ls=DASH)
arrow(14.4, 0.75, 14.4, 1.37, ls=DASH)
text(11.45, 0.42, "write addresses", ha="left")
choice(14.95, 1.05, 3)

# data + scales to the execute units
for _, y in ports:
    ax.plot([16.1, 17.0], [y, y], color="black", lw=0.6)
ax.plot([17.0, 17.0], [1.0, 3.95], color="black", lw=0.6)
arrow(17.0, 3.95, 17.9, 3.95); arrow(17.0, 1.0, 17.9, 1.0)

# execute units: conversions and the widened FMA lanes apply scales as exponent adds
box(17.9, 3.35, 5.9, 1.25)
text(20.85, 4.2, "Conversions", size=7.5, ha="center")
text(20.85, 3.7, "scales as exponent additions", ha="center")
box(17.9, 0.3, 5.9, 2.75)
text(20.85, 2.65, "FMA unit", size=7.5, ha="center")
box(18.15, 1.3, 5.4, 0.7)
text(20.85, 1.65, "16 existing lanes, 5 formats", ha="center")
box(18.15, 0.5, 5.4, 0.7, ARITH)
text(20.85, 0.85, "exponents widened to 10 bits", ha="center")
choice(23.4, 2.65, 4)

print(pf.save(fig, os.path.join(HERE, "out", "vecunit")))
