"""Where a unit's area sits, and how two configs differ, module by module.

    python breakdown.py <unit> <CONFIG_A> <CONFIG_B>

Reads runs/<CONFIG>/<unit>.hier/area.txt (from synth_hier.sh). Yosys reports
each module's own cells and, separately, which submodules it instantiates; this
multiplies the tree out, so every module's total is its own area times the
number of copies in the whole unit.

firtool numbers duplicate module names (RecFNToRecFN_15, _18, ...), which
differ between configs for no design reason, so a trailing _<n> is dropped and
same-named modules are added together.
"""
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(path):
    own, children, cur = {}, defaultdict(dict), None
    for line in open(path):
        m = re.match(r"=== (\S+) ===", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s+Chip area for module '\\(\S+)': ([0-9.]+)", line)
        if m:
            own[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"\s+(\S+)\s+(\d+)$", line)
        if cur and m and not m.group(1).startswith("sky130_") and m.group(1) != "cells:":
            children[cur][m.group(1)] = int(m.group(2))
        if line.startswith("=== design hierarchy ==="):
            cur = None
    return own, children


def copies(top, children):
    """How many instances of each module the whole unit holds."""
    n = defaultdict(int)
    def walk(mod, k):
        n[mod] += k
        for c, cnt in children.get(mod, {}).items():
            walk(c, k * cnt)
    walk(top, 1)
    return n


def norm(name):
    return re.sub(r"_\d+$", "", name)


def totals(unit, config):
    own, children = parse(os.path.join(HERE, "runs", config, unit + ".hier", "area.txt"))
    n = copies(unit, {k: v for k, v in children.items() if k in own})
    area, count = defaultdict(float), defaultdict(int)
    for mod, k in n.items():
        if mod in own:
            area[norm(mod)] += own[mod] * k
            count[norm(mod)] += k
    return area, count


unit, a, b = sys.argv[1:4]
A, nA = totals(unit, a)
B, nB = totals(unit, b)
tA, tB = sum(A.values()), sum(B.values())

rows = sorted(set(A) | set(B), key=lambda m: -abs(B.get(m, 0) - A.get(m, 0)))
print(f"{unit}: {a} -> {b}")
print(f"{'module':46} {'copies':>11} {'area A':>10} {'area B':>10} {'B - A':>10}")
for m in rows:
    d = B.get(m, 0) - A.get(m, 0)
    if abs(d) < 1 and A.get(m, 0) < 0.01 * tA:
        continue
    print(f"{m:46} {nA.get(m, 0):>5}->{nB.get(m, 0):<5} {A.get(m, 0):>10.0f} {B.get(m, 0):>10.0f} {d:>+10.0f}")
print(f"{'TOTAL (hierarchical, before buffering)':46} {'':11} {tA:>10.0f} {tB:>10.0f} {tB - tA:>+10.0f}"
      f"  ({100 * (tB - tA) / tA:+.1f}%)")
