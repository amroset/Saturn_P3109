"""Inter-block kernels, LMUL 1: cycles per element (vector kernels, median and min-max
over 8 code layouts; M3) and the OPU GEMM (cycles per 32-k block, median of 8 layouts; Q4)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf

SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "across-m3.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "results-across")
E = json.load(open(SUMMARY))["kernels"]
KERNELS = [("quantrow", "Quant"), ("dequantrow", "Deq"), ("gemv", "GEMV"), ("smulrow", "SMul"), ("addrow", "BAdd")]
SERIES = [("unscaled", "unscaled"), ("pairing", "ruleA"), ("pairing_buf4", "buf4"), ("explicit", "explicit"), ("software", "software")]
from constants import OPU as O
OPU = {"unscaled": O["unscaled_db"], "ruleA": O["ruleA"], "buf4": O["ruleA_db8"], "explicit": O["explicit"], "software": O["software"]}

pf.use()
fig, (ax, ax2) = pf.figure("single", height=1.75, ncols=2, gridspec_kw={"width_ratios": [5, 1.25], "wspace": 0.35})
w = 0.16
for j, (lab, concept) in enumerate(SERIES):
    xs, ys, lo, hi = [], [], [], []
    for i, (k, _) in enumerate(KERNELS):
        d = E[k][lab]
        m = d["median"] / 8192
        xs.append(i + (j - 2) * w); ys.append(m)
        lo.append(m - d["min"] / 8192); hi.append(d["max"] / 8192 - m)
    ax.bar(xs, ys, w, color=pf.COLORS[concept], hatch=pf.HATCH[concept], edgecolor="black", label=pf.LABEL[concept])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6)
ax.set_xticks(range(len(KERNELS)), [n for _, n in KERNELS])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Cycles per element")
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True)
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.45))

for j, concept in enumerate(OPU):
    ax2.bar(j, OPU[concept], 0.8, color=pf.COLORS[concept], hatch=pf.HATCH[concept], edgecolor="black")
ax2.set_xticks([2], ["OPU GEMM"])
ax2.tick_params(axis="x", length=0)
ax2.set_ylabel("Cycles per 32-$k$ block")
ax2.set_ylim(0, 3200)
ax2.yaxis.grid(True)
print(pf.save(fig, OUT))
