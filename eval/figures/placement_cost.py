"""Inter-block kernels: time of pairing, with one-register and with larger buffers, relative to the
best-case explicit operand (per layout, median and min-max over 8 code layouts), grouped by instruction,
with the scale copies per block that pairing needs. Element-wise kernels: A1b; vfdot4.vx kernels: A1; vfdot4.vv attention:
V1b (all with every loop head aligned to 16 bytes in both binaries); each scheme's fastest kernel (unrolled for
vfdot4.vx, one register for the .vv kernel, for every scheme). OPU: cycles per 32-k block, FP8 (Q4, Q4c) and FP4
(Q4e), within 1.2 cycle over layouts: pairing's fastest kernel on the 4-entry buffer, and on 8 entries with
copied scales. No copies (C1): the same pairing binaries without their scale copies, timing only, on the same
buffer as the bar it marks (element-wise and vfdot4.vx kernels; the fastest .vv and OPU kernels have none)."""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from constants import OPU, OPU_FP4

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
E = json.load(open(os.path.join(DATA, "across-a1b.json")))["kernels"]
D = json.load(open(os.path.join(DATA, "dot-a1.json")))["results"]
A = json.load(open(os.path.join(DATA, "attn-v1b.json")))["results"]
C = json.load(open(os.path.join(DATA, "c1-nocopy.json")))["results"]

def rel(num, den):            # per-layout ratios, in percent: median, min, max
    r = [100 * (a / b - 1) for a, b in zip(num, den)]
    return st.median(r), min(r), max(r)

rows = []                     # (label, pairing, larger buffer, group, copies)
nocopy = {}                   # row index -> (no copies on one-register buffers, on the larger ones), medians
for k, name, c in [("quantrow", "Quant", 3), ("dequantrow", "Deq", 3), ("gemv", "GEMV", 3), ("smulrow", "SMul", 6), ("addrow", "BAdd", 9)]:
    ex = E[k]["explicit"]["per_layout"]
    nocopy[len(rows)] = tuple(rel(C[f"ew/{k}|{b}|ruleA"]["per_layout"], ex)[0] for b in ("nocopy", "nocopy+buf4"))
    rows.append((name, rel(E[k]["pairing"]["per_layout"], ex), rel(E[k]["pairing_buf4"]["per_layout"], ex), 0, str(c)))
for k, name in [("gemv", "Dot\n1"), ("gemm", "Dot\n4"), ("gemm8m1", "Dot 8\nm1"), ("gemm8m2", "Dot 8\nm2")]:
    ex = D[f"{k}|explicit|unroll"]["per_layout"]
    nocopy[len(rows)] = tuple(rel(C[f"dot/{k}|{b}|unroll"]["per_layout"], ex)[0] for b in ("nocopy", "nocopy+buf4"))
    rows.append((name, rel(D[f"{k}|pairing|unroll"]["per_layout"], ex),
                 rel(D[f"{k}|pairing+buf4|unroll"]["per_layout"], ex), 1, "3"))
ea = A["attn|explicit|single"]["per_layout"]
rows.append(("Attn", rel(A["attn|pairing|single"]["per_layout"], ea), rel(A["attn|pairing+buf4|single"]["per_layout"], ea), 2, "0"))
for name, O in (("FP8", OPU), ("FP4", OPU_FP4)):
    opu = lambda v: (100 * (v / O["explicit"] - 1),) * 3
    rows.append((name, opu(O["ruleA"]), opu(O["ruleA_db8_copy"]), 3, "0/4"))
GROUPS = ["Element-wise", "vfdot4.vx", ".vv", "OPU"]

pf.use()
fig, ax = pf.figure("single", height=2.05)
w = 0.36
for g in range(len(GROUPS)):
    idx = [i for i, r in enumerate(rows) if r[3] == g]
    if g % 2:
        ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color="#EEEEEE", zorder=0)
    ax.text((idx[0] + idx[-1]) / 2, 1.03, GROUPS[g], ha="center", fontsize=6.5, transform=ax.get_xaxis_transform())
for j, (concept, idx) in enumerate([("ruleA", 1), ("buf4", 2)]):
    xs = [i + (j - 0.5) * w for i in range(len(rows))]
    ys = [r[idx][0] for r in rows]
    err = [[r[idx][0] - r[idx][1] for r in rows], [r[idx][2] - r[idx][0] for r in rows]]
    ax.bar(xs, ys, w, color=pf.COLORS[concept], hatch=pf.HATCH[concept], edgecolor="black",
           label=pf.LABEL[concept], zorder=2)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="black", elinewidth=0.6, zorder=3)
for i, nc in nocopy.items():         # where each bar would end without the copies
    ax.plot([i - w / 2, i + w / 2], nc, ls="none", marker="o", ms=3.4, mew=0.8, mfc="white", mec="black", zorder=4,
            label="No copies" if i == 0 else None)
ax.axhline(0, color="black", lw=0.6, zorder=1)
ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
ax.tick_params(axis="x", length=0)
for i, r in enumerate(rows):         # copies per block under pairing, under the tick labels
    ax.text(i, -0.205, r[4], ha="center", va="top", fontsize=6, transform=ax.get_xaxis_transform())
ax.text(-0.75, -0.205, "Copies", ha="right", va="top", fontsize=6, transform=ax.get_xaxis_transform())
ax.set_ylabel("Time vs explicit (%)")
ax.set_ylim(-4.5, 7.5)
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.yaxis.grid(True, zorder=0)
h, l = ax.get_legend_handles_labels()
ax.legend(h[1:] + h[:1], l[1:] + l[:1], ncol=3, loc="upper left", bbox_to_anchor=(-0.02, 1.32), columnspacing=0.9, handletextpad=0.4)
for r in rows:
    print(f"{r[0].replace(chr(10), ' '):14s} pairing {r[1][0]:+5.2f} ({r[1][1]:+.2f}..{r[1][2]:+.2f})  larger {r[2][0]:+5.2f} ({r[2][1]:+.2f}..{r[2][2]:+.2f})")
for i, nc in nocopy.items():
    print(f"{rows[i][0].replace(chr(10), ' '):14s} no copies {nc[0]:+5.2f}  larger {nc[1]:+5.2f}")
print(pf.save(fig, os.path.join(HERE, "out", "placement-cost")))
