"""Area added by block scaling (Sky130, Yosys + ABC, cell area): vector unit vs OPU,
split into arithmetic, scale delivery and unused logic; labels give the relative growth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
from constants import VEC, OPU_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
d = lambda t: (t[1] - t[0]) / 1e6
vec = [("Scaled arithmetic", d(VEC["FMA pipe"]) + d(VEC["FPConv"])),
       ("Scale delivery", d(VEC["Backend logic"]) + d(VEC["FP issue queues"]) + d(VEC["FP sequencer"])),
       ("Unused", d(VEC["Integer sequencer"]))]
opu = [("Scaled arithmetic", 256 * d(OPU_AREA["cell"])),
       ("Scale delivery", d(OPU_AREA["sequencer"]) + d(OPU_AREA["top"])),
       ("Unused", 0.0)]
base_vec = sum(v[0] for v in VEC.values()) / 1e6
base_opu = (256 * OPU_AREA["cell"][0] + OPU_AREA["sequencer"][0] + OPU_AREA["top"][0]) / 1e6
style = {"Scaled arithmetic": (pf.COLORS["arithmetic"], pf.HATCH["arithmetic"]), "Scale delivery": (pf.COLORS["delivery"], pf.HATCH["delivery"]), "Unused": (pf.COLORS["unused"], pf.HATCH["unused"])}

pf.use()
fig, ax = pf.figure("single", height=1.05)
for y, parts, base in ((1, vec, base_vec), (0, opu, base_opu)):
    left = 0
    for name, v in parts:
        col, h = style[name]
        ax.barh(y, v, 0.55, left=left, color=col, hatch=h, edgecolor="black", label=name if y == 1 else None)
        left += v
    ax.text(left + 0.01, y, f"+{left:.2f} mm$^2$ (+{100 * left / base:.1f}%)" if left / base < 0.1 else
            f"+{left:.2f} mm$^2$ (+{100 * left / base:.0f}%)", va="center")
ax.set_yticks([1, 0], ["Vector unit", "OPU"])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.8)
ax.set_xlabel("Area added (mm$^2$)")
ax.xaxis.grid(True)
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2)
print(pf.save(fig, os.path.join(HERE, "out", "area")))
