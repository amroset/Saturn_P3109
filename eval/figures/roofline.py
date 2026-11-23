"""Roofline of the inter-block workloads: (a) whole system, (b) vector unit, (c) OPU GEMM.
Vector kernels: median of 8 layouts, every loop head aligned to 16 bytes in both binaries (element-wise:
A1b; vfdot4.vx: A1; vfdot4.vv attention, four accumulators: V1b). OPU: median of 8 layouts (Q4, Q4b, Q4c).
Operations: elements for the element-wise kernels, multiply-adds for the others. Bytes: the data each
call must move at least once (inputs, outputs, scales), as for the element-wise kernels."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from constants import OPU, OPU_MACS, OPU_AI, BW, BW32, BW32LD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
E = json.load(open(os.path.join(DATA, "across-a1b.json")))["kernels"]
D = json.load(open(os.path.join(DATA, "dot-a1.json")))["results"]
A = json.load(open(os.path.join(DATA, "attn-v1b.json")))["results"]

# element-wise: operations per byte = 1 / (element bytes in + out + one scale byte per 32 elements per block operand)
EW = [("quantrow", "Quant", 1 / (2 + 1 + 1 / 32)), ("dequantrow", "Deq", 1 / (1 + 2 + 1 / 32)),
      ("gemv", "GEMV", 1 / (1 + 1 / 32)), ("smulrow", "SMul", 1 / (1 + 1 + 2 + 2 / 32)), ("addrow", "BAdd", 1 / (3 + 3 / 32))]
def ew(k):
    return {"unscaled": 8192 / E[k]["unscaled"]["median"], "explicit": 8192 / E[k]["explicit"]["median"],
            "ruleA": 8192 / E[k]["pairing"]["median"]}
# vfdot4.vx: 64 x 256 weights and their scales (64 x 8), T tokens of 256 values and 8 scales, 64 x T FP32 outputs
def dot_ai(t):
    return 64 * 256 * t / (64 * 256 + 64 * 8 + t * (256 + 8) + t * 64 * 4)
DOT = [("gemv", "Dot 1", 1), ("gemm", "Dot 4", 4), ("gemm8m1", "Dot 8 m1", 8), ("gemm8m2", "Dot 8 m2", 8)]
def dot(k, R=D, v="unroll"):      # each scheme's fastest kernel: unrolled (.vx), one register (.vv)
    return {"unscaled": R[f"{k}|pairing|unscaled"]["mac_per_cycle"], "explicit": R[f"{k}|explicit|{v}"]["mac_per_cycle"],
            "ruleA": R[f"{k}|pairing|{v}"]["mac_per_cycle"]}
# vfdot4.vv attention: query 32 x 96 and its scales, keys 64 x 32 x 96 and theirs, 64 x 32 FP32 scores
ATT_AI = 64 * 32 * 96 / (32 * 96 + 32 * 3 + 64 * 32 * 96 + 64 * 32 * 3 + 64 * 32 * 4)
opu = {"unscaled": OPU_MACS / OPU["unscaled_db"], "explicit": OPU_MACS / OPU["explicit"],
       "ruleA": OPU_MACS / OPU["ruleA"], "buf4": OPU_MACS / OPU["ruleA_db8_copy"]}

VAR = [("unscaled", "o"), ("explicit", "D"), ("ruleA", ".")]
def marks(ax, x, vals, big=1.0):
    for concept, m in VAR:
        v = vals[concept]
        if m == "o":
            ax.plot(x, v, "o", mfc="none", mec=pf.COLORS[concept], ms=5.5 * big, mew=0.9)
        elif m == "D":
            ax.plot(x, v, "D", mfc="none", mec=pf.COLORS[concept], ms=4 * big, mew=0.9)
        else:
            ax.plot(x, v, "o", color=pf.COLORS[concept], ms=2.2 * big)

POINTS = ([(ai, ew(k), name) for k, name, ai in EW] + [(dot_ai(t), dot(k), name) for k, name, t in DOT]
          + [(ATT_AI, dot("attn", A, "single"), "Attn .vv")])

pf.use()
fig, (a, b, c) = pf.figure("double", height=2.1, ncols=3, gridspec_kw={"width_ratios": [1.1, 1.35, 0.8], "wspace": 0.42})
# (a) whole system
x = np.logspace(-1, 2, 300)
a.loglog(x, np.minimum(BW * x, 256), "k-", lw=0.9)
a.loglog(x, np.minimum(BW * x, 16), "k-", lw=0.9)
a.text(0.12, 300, "OPU peak 256 MAC/cycle", va="bottom")
a.text(90, 18, "Vector FP8\npeak 16/cycle", ha="right", va="bottom")
for ai, vals, _ in POINTS:
    marks(a, ai, vals, 0.7)
marks(a, OPU_AI, opu, 0.7)
a.set_xlim(0.1, 100); a.set_ylim(1, 600)
a.set_xlabel("Operations per byte")
a.set_ylabel("Operations per cycle")
a.set_title("(a)", loc="left")
# (b) vector unit
x = np.logspace(np.log10(0.2), 1, 200)
b.loglog(x, np.minimum(BW * x, 16), "k-", lw=0.9)
b.loglog(x, np.minimum(BW32 * x, 16), "k--", lw=0.7)
b.loglog(x, np.minimum(BW32LD * x, 16), "k:", lw=0.8)
b.set_xlim(0.2, 10); b.set_ylim(1.5, 22)
from matplotlib.lines import Line2D
b.legend([Line2D([], [], color="k", lw=0.9), Line2D([], [], color="k", lw=0.7, ls="--"), Line2D([], [], color="k", lw=0.8, ls=":")],
         ["LMUL-8 loads", "32-byte loads + stores", "32-byte loads only"], loc="lower right", fontsize=6,
         handlelength=1.6, title="Memory roofs", title_fontsize=6, borderaxespad=0.2)
b.text(1.05, 17, "peak 16/cycle", ha="left", va="bottom")
OFF = {"Quant": (5, 3), "Deq": (5, -7), "GEMV": (-6, -6), "SMul": (0, 6), "BAdd": (0, -11),
       "Dot 1": (5, -8), "Dot 4": (-5, 3), "Dot 8 m1": (5, -7), "Dot 8 m2": (5, -6), "Attn .vv": (-6, -2)}
for ai, vals, name in POINTS:
    marks(b, ai, vals)
    off = OFF[name]
    b.annotate(name, (ai, vals["ruleA"]), xytext=off, textcoords="offset points", fontsize=6.5,
               ha="right" if off[0] < 0 else "left" if off[0] > 0 else "center",
               bbox=dict(fc="white", ec="none", pad=0.4))
b.set_xlim(0.2, 10); b.set_ylim(1.5, 22)
b.set_xticks([0.25, 0.5, 1, 2, 5, 10], ["0.25", "0.5", "1", "2", "5", "10"]); b.set_xticks([], minor=True)
b.set_yticks([2, 4, 8, 16], ["2", "4", "8", "16"]); b.set_yticks([], minor=True)
b.set_xlabel("Operations per byte")
b.set_title("(b)", loc="left")
# (c) OPU, linear: a dot plot, the axis does not start at zero
OPUV = [("unscaled", "Unsc."), ("explicit", "Expl."), ("ruleA", "Pair."), ("buf4", "Pair.,\n8 entr.")]
for i, (concept, label) in enumerate(OPUV):
    v = opu[concept]
    if concept == "unscaled":
        c.plot(i, v, "o", mfc="none", mec=pf.COLORS[concept], ms=6.5, mew=1.0)
    elif concept == "explicit":
        c.plot(i, v, "D", mfc="none", mec=pf.COLORS[concept], ms=5, mew=1.0)
    elif concept == "ruleA":
        c.plot(i, v, "o", color=pf.COLORS[concept], ms=4)
    else:
        c.plot(i, v, "s", mfc="white", mec=pf.COLORS[concept], ms=4.5, mew=1.0)
    c.text(i + 0.16, v, f"{v:.0f}", va="center", fontsize=6.5)
c.axhline(256, color="black", lw=0.9)
c.text(3.45, 254.5, "Peak", ha="right", va="top")
c.set_xlim(-0.5, 3.5)
c.set_ylim(225, 260)
c.set_xticks(range(4), [l for _, l in OPUV], fontsize=6.5)
c.tick_params(axis="x", length=0)
c.yaxis.grid(True)
c.set_ylabel("MAC per cycle")
c.set_title("(c)", loc="left")
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="none", marker="o", mfc="none", mec=pf.COLORS["unscaled"], ms=5.5, mew=0.9),
     Line2D([], [], ls="none", marker="D", mfc="none", mec=pf.COLORS["explicit"], ms=4, mew=0.9),
     Line2D([], [], ls="none", marker="o", color=pf.COLORS["ruleA"], ms=2.2)]
fig.legend(h, [pf.LABEL["unscaled"], pf.LABEL["explicit"], pf.LABEL["ruleA"]], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98))
print(pf.save(fig, os.path.join(HERE, "out", "roofline")))
