"""The precision sweep's results in one table per unit.

    python sweep_table.py <baseline CONFIG> <CONFIG>...

For each config: the flattened totals from synth_unit.sh (area after
buffering, critical path) and, from synth_hier.sh, the area split by what
each module does, so the cost of a precision can be traced to the part that
grew. Everything is relative to the baseline config.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breakdown import totals  # noqa: E402


def flat(config, unit):
    p = os.path.join(HERE, "runs", config, unit, "summary.txt")
    if not os.path.exists(p):
        return None, None
    s = open(p).read()
    area = float(re.search(r"area\s+([0-9.]+)", s).group(1))
    delay = float(re.search(r"critical path\s+([0-9.]+)", s).group(1))
    return area, delay


def core_shape(name):
    m = re.search(r"_e(\d+)_s(\d+)$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def role(unit, mod):
    if mod.startswith("P3109Rounder"):
        return "8-bit rounders"
    m = re.match(r"RoundAnyRawFNToRecFN_ie\d+_is\d+_oe(\d+)_os(\d+)$", mod)
    if m and int(m.group(1)) + int(m.group(2)) <= 9:      # OCP's rounders into an 8-bit format
        return "8-bit rounders"
    if unit == "FPFMAPipe" and mod.startswith("MulAddRecFN"):
        e, s = core_shape(mod)
        if (e, s) in ((11, 53), (8, 24), (8, 8)):
            return "FP64/FP32/BF16 cores"
        if s == 11:
            return "FP16 cores"
        return "8-bit cores"
    if mod in ("SegmentedFMAPipe", "FPFMAPipe", "FPConvBlock", "FPConvPipe"):
        return "unit's own logic (readers, muxes, pipeline)"
    return "other (wide rounders, converters)"


def short(config):
    m = re.match(r"P3109Sweep(\d+)", config)
    if m:
        return "{" + ",".join(m.group(1)) + "}"
    return {"MXV256D128ShuttleConfig": "OCP",
            "P3109V256D128ShuttleConfig": "{3,4} pair"}.get(config, config)


if __name__ == "__main__":
    base, configs = sys.argv[1], sys.argv[2:]
    builds = [base] + configs
    W = 11
    for unit in ("FPConvPipe", "FPFMAPipe"):
        flats = {c: flat(c, unit) for c in builds}
        roles = {}
        for c in builds:
            try:
                per_mod, _ = totals(unit, c)
            except FileNotFoundError:
                per_mod = {}
            r = {}
            for m, a in per_mod.items():
                r[role(unit, m)] = r.get(role(unit, m), 0) + a
            roles[c] = r
        names = sorted({k for r in roles.values() for k in r})

        def line(label, vals, fmt):
            print(f"{label:34}" + "".join(f"{fmt(v):>{W}}" for v in vals))

        print(f"\n{unit}  (um^2, Sky130 tt; deltas against {short(base)})")
        line("", [short(c) for c in builds], str)
        line("total area, flattened", [flats[c][0] for c in builds],
             lambda v: f"{v:.0f}" if v else "-")
        line("  vs baseline", [flats[c][0] for c in builds],
             lambda v: f"{100 * (v - flats[base][0]) / flats[base][0]:+.1f}%" if v and flats[base][0] else "-")
        line("critical path (ns)", [flats[c][1] for c in builds],
             lambda v: f"{v:.2f}" if v else "-")
        print("per module (hierarchical, before buffering):")
        for n in names:
            line("  " + n[:32], [roles[c].get(n, 0) for c in builds], lambda v: f"{v:.0f}")
        print("change against the baseline:")
        for n in names:
            line("  " + n[:32], [roles[c].get(n, 0) - roles[base].get(n, 0) for c in builds],
                 lambda v: f"{v:+.0f}")
