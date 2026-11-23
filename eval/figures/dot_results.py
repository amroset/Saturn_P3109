"""Dot-product kernels (vfdot4.vx): fraction of the instruction's peak (16 multiply-adds per
cycle), median and min-max over 8 code layouts (M3)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "dot.json")))
R, MACS = D["results"], D["macs"]
KERNELS = [("gemv", "GEMV\n1 token\nLMUL 2"), ("gemm", "GEMM\n4 tokens\nLMUL 2"), ("gemm8m1", "GEMM\n8 tokens\nLMUL 1"), ("gemm8m2", "GEMM\n8 tokens\nLMUL 2")]
SERIES = [("pairing", "unscaled", "unscaled"), ("pairing", "unroll", "ruleA"),
          ("pairing+buf4", "unroll", "buf4"), ("explicit", "unroll", "explicit")]
PEAK = 16.0

pf.use()
fig, ax = pf.figure("single", height=1.9)
w = 0.19
for j, (build, var, concept) in enumerate(SERIES):
    xs, ys, lo, hi = [], [], [], []
    for i, (k, _) in enumerate(KERNELS):
        d = R[f"{k}|{build}|{var}"]
        f = lambda c: MACS[k] / c / PEAK * 100
        m = f(d["median"])
        xs.append(i + (j - 1.5) * w); ys.append(m)
        lo.append(m - f(d["max"])); hi.append(f(d["min"]) - m)
    ax.bar(xs, ys, w, color=pf.COLORS[concept], hatch=pf.HATCH[concept], edgecolor="black", label=pf.LABEL[concept])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6)
ax.set_xticks(range(len(KERNELS)), [n for _, n in KERNELS])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Fraction of peak (%)")
ax.set_ylim(0, 100)
ax.yaxis.grid(True)
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.38))
print(pf.save(fig, os.path.join(HERE, "out", "dot-results")))
