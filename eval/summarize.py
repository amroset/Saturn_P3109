#!/usr/bin/env python3
"""Summarize timing runs over the eight code layouts.

    eval/summarize.py LABEL=GLOB [LABEL=GLOB ...] [--compare LABEL:TOKEN LABEL:TOKEN]

Each GLOB selects one log per layout; the layout number comes from `-l<i>` in
the file name. For every result, the script prints the median, minimum and
maximum over the layouts, and how many distinct checksums the layouts gave
(one, if they all computed the same thing).

It reads three kinds of output:
  RESULT <name ...> <n> cycles [... checksum <x>]   most benchmarks
  <op> unscaled <x> B=1 <y> B=32 <z> cycles/...      vec-block-timing
  # <tag> <name> / <tag> <iterations> <cycles>       opu-timing: the fitted
                                                     cycles per iteration (fit.py)

--compare A:ruleA B:shared matches every result of A whose name contains the
token ruleA with the result of B whose name has shared in its place, and prints
the difference layout by layout: median [min, max] in percent. A difference
counts only if the two ranges of cycles do not overlap; otherwise the line says
"no measurable difference".
"""
import glob
import re
import statistics
import sys
from collections import defaultdict


def layout(path):
    m = re.search(r"-l(\d+)[.-]", path)
    return int(m[1]) if m else path


def fit(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def parse(path):
    """{name tuple: (value, checksum or None)} for one log."""
    out, names, pts = {}, {}, defaultdict(list)
    text = open(path, errors="replace").read()
    if "finish" not in text:
        print(f"warning: {path} did not finish", file=sys.stderr)
    for line in text.splitlines():
        m = re.match(r"RESULT\s+(.*?)\s+(\d+) cycles(.*)", line)
        if m:
            c = re.search(r"checksum (\S+)", m[3])
            out[tuple(m[1].split())] = (int(m[2]), c[1] if c else None)
            continue
        m = re.match(r"(\S+)\s+unscaled ([\d.]+)\s+B=1 ([\d.]+)\s+B=32 ([\d.]+) cycles", line)
        if m:
            for k, v in zip(("unscaled", "B=1", "B=32"), m.groups()[1:]):
                out[(m[1], k)] = (float(v), None)
            continue
        m = re.match(r"# (\S+) (.*)", line)
        if m:
            names[m[1]] = m[2].strip()
            continue
        p = line.split()
        if len(p) == 3 and p[0] in names and p[1].isdigit() and p[2].isdigit():
            pts[p[0]].append((int(p[1]), int(p[2])))
    for tag, ps in pts.items():
        out[(names[tag],)] = (fit(ps), None)
    return out


def load(pattern):
    """{name: {layout: (value, checksum)}}"""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no logs match {pattern}")
    res = defaultdict(dict)
    for f in files:
        for k, v in parse(f).items():
            res[k][layout(f)] = v
    return res


def fmt(xs):
    return f"{statistics.median(xs):10.2f} [{min(xs):.2f}, {max(xs):.2f}]"


def main(argv):
    sets, compare = {}, None
    i = 0
    while i < len(argv):
        if argv[i] == "--compare":
            compare = argv[i + 1:i + 3]
            i += 3
            continue
        label, _, pattern = argv[i].partition("=")
        if not pattern:
            sys.exit(__doc__)
        sets[label] = load(pattern)
        i += 1
    if not sets:
        sys.exit(__doc__)

    for label, res in sets.items():
        print(f"== {label}: median [min, max] over layouts")
        for name in sorted(res):
            vals = [v for v, _ in res[name].values()]
            sums = {c for _, c in res[name].values() if c}
            extra = f"  {len(sums)} checksum{'s' if len(sums) != 1 else ''}" if sums else ""
            print(f"  {' '.join(name):40s} {fmt(vals)}  n={len(vals)}{extra}")

    if compare:
        (la, ta), (lb, tb) = (c.split(":", 1) for c in compare)
        a, b = sets[la], sets[lb]
        print(f"== {la}:{ta} against {lb}:{tb}, per layout, percent")
        for name in sorted(a):
            if ta not in name:
                continue
            other = tuple(tb if t == ta else t for t in name)
            if other not in b:
                continue
            common = sorted(set(a[name]) & set(b[other]), key=str)
            if not common:
                continue
            d = [100 * (a[name][l][0] / b[other][l][0] - 1) for l in common]
            va, vb = [a[name][l][0] for l in common], [b[other][l][0] for l in common]
            overlap = min(va) <= max(vb) and min(vb) <= max(va)
            verdict = "no measurable difference" if overlap else ""
            print(f"  {' '.join(name):40s} {statistics.median(d):+6.1f} [{min(d):+.1f}, {max(d):+.1f}]  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
