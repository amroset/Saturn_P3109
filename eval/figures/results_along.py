"""Along the blocks: cycles per element at LMUL 1 and LMUL 4 (LMUL benchmark: the rule A
and explicit binaries differ only in register names), median and min-max over 8 layouts."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "data", "sweep3-summary.json")))["lmul"]
KERNELS = [("dequant", "Deq\n$E{=}32$"), ("quant", "Quant\n$E{=}32$"), ("smul", "SMul\n$E{=}32$"),
           ("dequant16", "Deq\n$E{=}16$"), ("quant16", "Quant\n$E{=}16$")]
SERIES = [("unscaled_ruleA", "unscaled"), ("ruleA", "ruleA"), ("shared", "explicit"), ("scalar", "scalar")]

pf.use()
fig, axes = pf.figure("single", height=2.6, nrows=2, sharex=True, gridspec_kw={"hspace": 0.15})
w = 0.19
for ax, lm in zip(axes, ("1", "4")):
    for j, (lab, concept) in enumerate(SERIES):
        xs, ys, lo, hi = [], [], [], []
        for i, (k, _) in enumerate(KERNELS):
            d = L[k][lm][lab]
            m = d["median"] / 2048
            xs.append(i + (j - 1.5) * w); ys.append(m)
            lo.append(m - d["min"] / 2048); hi.append(d["max"] / 2048 - m)
        ax.bar(xs, ys, w, color=pf.COLORS[concept], hatch=pf.HATCH[concept], edgecolor="black",
               label=pf.LABEL[concept] if lm == "1" else None)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.yaxis.grid(True)
    ax.text(0.99, 0.95, f"LMUL {lm}", transform=ax.transAxes, ha="right", va="top")
axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
axes[1].set_xticks(range(len(KERNELS)), [n for _, n in KERNELS])
axes[1].tick_params(axis="x", length=0)
fig.supylabel("Cycles per element", fontsize=8, x=0.02)
print(pf.save(fig, os.path.join(HERE, "out", "results-along")))
