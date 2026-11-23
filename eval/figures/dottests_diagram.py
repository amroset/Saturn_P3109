"""How vfdot4 was verified, by scope: (1) the lane alone and (2) the instruction in the SoC against
the exact reference; (3) the register-use rules, (4) whole kernels and (5) the existing builds by
self-checks. Counts from notes/experiments.md (D1, D2, D3, M6)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REF, BOX = "#D9D9D9", "white"
W, GAP, X0, Y0, H = 6.4, 0.6, 0.8, 0.4, 7.3

pf.use()
fig, ax = pf.figure("double", height=2.45)
ax.set_xlim(0, 36); ax.set_ylim(0, 12.3); ax.axis("off")

def text(x, y, s, **kw):
    kw.setdefault("fontsize", 6.5)
    ax.text(x, y, s, va="center", **kw)

def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                         shrinkA=0, shrinkB=0, mutation_scale=6))

LEVELS = [
    ("(1) Lane", "ScaledDotLane, Verilator",
     ["every code pair", "random, cancelling sums", "FP32 range limits", "2 formats, 5 roundings"],
     ["2,141,790 ops, 0 wrong;", "5/5 planted errors caught"]),
    ("(2) Instruction", "vfdot4 in the SoC",
     [".vv, .vx; scaled, unscaled", "E 1, 2, 8; LMUL 1, 2, 4", "masks; scale-buffer hits", "random bits 63:40"],
     ["34 of 34 tests", "on 3 builds"]),
    ("(3) Register rules", "pairing's constraints",
     ["misaligned groups,", "scale groups past v31,", "destinations on scales", "must trap; legal ones run"],
     ["22 of 22 cases,", "4 with vfdot4"]),
    ("(4) Kernels", "GEMV and GEMMs",
     ["pairing, 4-register", "buffers, explicit", "operand: one checksum", "per kernel; LMUL 1 = 2"],
     ["identical on", "every build"]),
    ("(5) Regression", "existing builds",
     ["build without vfdot4:", "Verilog compared;", "element-wise tests on", "the vfdot4 build"],
     ["unchanged;", "137 of 137 pass"]),
]

for n, (title, dut, checks, result) in enumerate(LEVELS):
    x = X0 + n * (W + GAP)
    ax.add_patch(Rectangle((x, Y0), W, H, fc=BOX, ec="black", lw=0.6))
    text(x + W / 2, Y0 + H - 0.55, title, ha="center", fontsize=7.5)
    text(x + W / 2, Y0 + H - 1.35, dut, ha="center", style="italic")
    for i, s in enumerate(checks):
        text(x + 0.3, Y0 + H - 2.35 - i * 0.75, s, ha="left", fontsize=6.2)
    ax.plot([x + 0.3, x + W - 0.3], [Y0 + 1.95, Y0 + 1.95], color="#999999", lw=0.4)
    for i, s in enumerate(result):
        text(x + W / 2, Y0 + 1.5 - i * 0.6, s, ha="center")

# exact reference feeds (1) and (2); (3)-(5) check themselves
rx1 = X0 + 2 * W + GAP
ax.add_patch(Rectangle((X0, 9.3), rx1 - X0, 2.4, fc=REF, ec="black", lw=0.6))
text((X0 + rx1) / 2, 10.85, "exact reference (Python, rational arithmetic):", ha="center", fontsize=7)
text((X0 + rx1) / 2, 10.1, "P3109 block decoding, exact sum, one rounding to FP32", ha="center")
for n in range(2):
    xc = X0 + n * (W + GAP) + W / 2
    arrow(xc, 9.3, xc, Y0 + H)
text(X0 + W + GAP / 2, 8.5, "expected results, bit for bit", ha="center")
x3 = X0 + 2 * (W + GAP)
x5 = X0 + 5 * W + 4 * GAP
text((x3 + x5) / 2, 10.5, "self-checks: trap or not, checksums, Verilog diff", ha="center", fontsize=7)
ax.plot([x3 + W / 2, x5 - W / 2], [9.7, 9.7], color="black", lw=0.6)
for n in range(2, 5):
    xc = X0 + n * (W + GAP) + W / 2
    arrow(xc, 9.7, xc, Y0 + H)

# scope
arrow(X0, 12.0, X0 + 5 * W + 4 * GAP - 2.2, 12.0)
text(X0 + 5 * W + 4 * GAP - 2.0, 12.0, "scope", ha="left")

print(pf.save(fig, os.path.join(HERE, "out", "dottests")))
