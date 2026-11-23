"""Compute-only microbenchmark (vec-block-timing): cycles per LMUL-1 instruction, destinations rotating
over four registers, no memory accesses; median and min-max over 8 code layouts. Pairing and unscaled:
timing-summary.json (ruleA build); explicit: same file (shared build); pairing with four registers per
port: timing-buf4.json (E11, P3109BlockBuf4). E = 1 reads two scale groups per instruction, E = 32 one."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
T = json.load(open(os.path.join(DATA, "timing-summary.json")))
B = json.load(open(os.path.join(DATA, "timing-buf4.json")))
INSNS = [("mul", "vfmul"), ("macc", "vfmacc"), ("ncvt", "vfncvtbf16"), ("wmul", "vfwmul")]
SERIES = [  # label, value source, color, hatch
    ("Unscaled", lambda k: T["ruleA"][k]["unscaled"], pf.COLORS["unscaled"], ""),
    ("Pairing, $E = 1$", lambda k: T["ruleA"][k]["E1"], pf.COLORS["ruleA"], ""),
    ("Pairing, $E = 32$", lambda k: T["ruleA"][k]["E32"], to_rgba(pf.COLORS["ruleA"], 0.4), ""),
    ("Pairing, larger buffer, $E = 1$", lambda k: B[k]["E1"], pf.COLORS["buf4"], pf.HATCH["buf4"]),
    ("Explicit, $E = 1$", lambda k: T["shared"][k]["E1"], pf.COLORS["explicit"], pf.HATCH["explicit"]),
]

pf.use()
fig, ax = pf.figure("single", height=2.0)
w = 0.16
for j, (label, f, color, hatch) in enumerate(SERIES):
    xs = [i + (j - 2) * w for i in range(len(INSNS))]
    v = [f(k) for k, _ in INSNS]
    ys = [x["median"] for x in v]
    err = [[x["median"] - x["min"] for x in v], [x["max"] - x["median"] for x in v]]
    ax.bar(xs, ys, w, color=color, hatch=hatch, edgecolor="black", lw=0.3, label=label, zorder=2)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2, zorder=3)
ax.set_xticks(range(len(INSNS)), [n for _, n in INSNS], family="monospace", fontsize=6.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Cycles per instruction")
ax.set_ylim(0, 6.5)
ax.yaxis.grid(True, zorder=0)
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=6, columnspacing=0.8, handlelength=1.4)
for k, _ in INSNS:
    print(k, [round(f(k)["median"], 2) for _, f, _, _ in SERIES])
print(pf.save(fig, os.path.join(HERE, "out", "compute-micro")))
