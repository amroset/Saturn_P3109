"""Qwen2.5-0.5B decoder layer (linear part, prefill of 64 tokens), hardware vs software scaling
on the same chip (Q1 method, SB1 software GEMM): (a) cycles per phase at full size, (b) speedup at three layer sizes."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "data", "layer.json")))["sizes"]
PHASES = [("qkv", "$qkv$"), ("o", "$o$"), ("gate_up", "gate+up"), ("down", "down"),
          ("quantize", "quant."), ("elementwise", "elem.")]
GEMMS = ["qkv", "o", "gate_up", "down"]

pf.use()
fig, (ax, ax2) = pf.figure("single", height=1.95, ncols=2, gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.45})
full = S["full-896"]
w = 0.38
for j, (side, concept, label) in enumerate([("hw", "ruleA", "Hardware scaling"), ("sw", "software", "Software scaling")]):
    ys = [full[side][p] / 1e6 for p, _ in PHASES]
    ax.bar([i + (j - 0.5) * w for i in range(len(PHASES))], ys, w, color=pf.COLORS[concept],
           hatch=pf.HATCH[concept], edgecolor="black", label=label)
for i, (p, _) in enumerate(PHASES):
    ax.text(i, full["sw"][p] / 1e6 + 0.3, f"{full['sw'][p] / full['hw'][p]:.1f}$\\times$", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(range(len(PHASES)), [n for _, n in PHASES], rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Cycles ($10^6$)")
ax.set_ylim(0, 16)
ax.yaxis.grid(True)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top")
fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(top=0.84)

sizes = ["small-256", "med-448", "full-896"]
hs = [S[k]["hidden_kv_ffn"][0] for k in sizes]
tot = [S[k]["sw"]["total"] / S[k]["hw"]["total"] for k in sizes]
gem = [sum(S[k]["sw"][g] for g in GEMMS) / sum(S[k]["hw"][g] for g in GEMMS) for k in sizes]
xs = range(len(hs))
ax2.plot(xs, gem, marker="s", color="black", lw=0.8, ms=3.5, mfc="white")
ax2.plot(xs, tot, marker="o", color=pf.COLORS["ruleA"], lw=0.8, ms=3.5)
ax2.text(2, gem[-1] + 0.15, "GEMMs", ha="right", va="bottom", fontsize=6.5)
ax2.text(1.1, tot[1] - 0.15, "total", ha="left", va="top", fontsize=6.5)
ax2.set_xticks(xs, [str(h) for h in hs])
ax2.set_xlabel("Hidden size")
ax2.set_ylabel("Speedup")
ax2.set_ylim(1, 4.5)
ax2.set_xlim(-0.4, 2.4)
ax2.yaxis.grid(True)
ax2.text(0.05, 0.97, "(b)", transform=ax2.transAxes, va="top")
print(pf.save(fig, os.path.join(HERE, "out", "layer")))
