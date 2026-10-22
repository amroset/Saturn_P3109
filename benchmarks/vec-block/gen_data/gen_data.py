#!/usr/bin/env python3
"""Generate vec-block/data.S and vec-block/tests.h: scaled mode (docs/scaled-mode.md).

  ./gen_data.py --std p3109 -o ..        # writes ../data.S and ../tests.h

Each test runs one instruction with vtype's scaled bit set and a block size B
(vtype bits 14:12, log2), on COUNT elements. Every 8-bit register operand vN is
a block, its Binary8p1uf scales in vN+1: element i takes byte i >> log2(B); a
scalar (.vf) or BF16 operand is not a block. The expected results follow P3109
v4.0.3:

  operands  omegaBlockDecode (5.4.1): x times 2^(s-128); scale 0 gives zero,
            NaN for an infinite or NaN x; scale 255 gives NaN
  result    to BF16: rounded once (frm); P3109's zero is +0 (4.8.2)
            to 8 bits: omegaBlockProject (5.4.2) with the scale in vd+1: the
            exact result divided by 2^(sr-128), rounded once; sr 255 gives NaN,
            sr 0 gives zero for anything but NaN

The exact arithmetic and the single rounding are fma_ref's (validated against
the Spike vectors of vec-mx-binary); this file adds only the scales.
"""

import argparse
import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common-data-gen"))

import fma_ref as R  # noqa: E402
from gfloat import RoundMode, encode_float, round_float  # noqa: E402
from gfloat_ref import BF16, FP8_STANDARDS, FRM, external_result, print_array, print_header  # noqa: E402

COUNT = 128      # elements per test; main.c's N
VL = 32          # elements per strip (VLEN 256, SEW 8, LMUL 1); main.c's VL
EDGE_SCALES = (0, 255, 1, 254, 127, 128, 129)
ZERO = ("num", 0, Fraction(0))


def scale_value(s):
    return Fraction(2) ** (s - 128)


def block_decode(x, s):
    """omegaBlockDecode of one element x (an exact value) with scale code s."""
    if s == 255:
        return ("nan",)
    if s == 0:
        return ("nan",) if x[0] != "num" else ZERO
    if x[0] != "num":
        return x
    return ("num", x[1], x[2] * scale_value(s))


def project8(fi, z, sr, rnd, sat=False):
    """omegaBlockProject of the exact result z with scale code sr, encoded in fi."""
    if sr == 255 or z[0] == "nan":
        return R.project(fi, ("nan",), rnd)
    if sr == 0:
        return R.project(fi, ZERO, rnd)
    if z[0] == "inf":
        return R.project(fi, z, rnd, sat)
    return R.project(fi, ("num", z[1], z[2] / scale_value(sr)), rnd, sat)


def project16(std, z, rnd):
    return external_result(std, R.project(BF16, z, rnd), 16)


def neg(x):
    return R._negate(x)


# Each test: kind, instruction, and the exact result Z from the decoded operands
# A2 (vs2), A1 (vs1), D (vd), F (the .vf scalar) and the rounding mode.
#   K8     8-bit result: vs2 v2/v3, vs1 v4/v5, vd v8 with its scales in v9
#   KW     BF16 result in v8-v9 (also the BF16 accumulator); vs2 v2/v3, vs1 v4/v5
#   KWW    BF16 result; vs2 BF16 in v16-v17, vs1 v4/v5
#   KWCVT  vfwcvtbf16.f.f.v v8, v2: block in v2/v3 to BF16 (ConvertFromBlock)
#   KNCVT  vfncvtbf16[.sat].f.f.w v8, v16: BF16 to a block in v8/v9 (ConvertToBlock)
def mul(x, y, r): return R.mul(x, y, r)
def add(x, y, r): return R.add(x, y, r)


TESTS = {
    "add_vv":   ("K8", "vfadd.vv v8, v2, v4", lambda a2, a1, d, f, r: add(a2, a1, r)),
    "sub_vv":   ("K8", "vfsub.vv v8, v2, v4", lambda a2, a1, d, f, r: add(a2, neg(a1), r)),
    "mul_vv":   ("K8", "vfmul.vv v8, v2, v4", lambda a2, a1, d, f, r: mul(a2, a1, r)),
    "macc_vv":  ("K8", "vfmacc.vv v8, v4, v2", lambda a2, a1, d, f, r: add(mul(a1, a2, r), d, r)),
    "nmsac_vv": ("K8", "vfnmsac.vv v8, v4, v2", lambda a2, a1, d, f, r: add(neg(mul(a1, a2, r)), d, r)),
    "madd_vv":  ("K8", "vfmadd.vv v8, v4, v2", lambda a2, a1, d, f, r: add(mul(a1, d, r), a2, r)),
    "mul_vf":   ("K8", "vfmul.vf v8, v2, f0", lambda a2, a1, d, f, r: mul(a2, f, r)),
    "rsub_vf":  ("K8", "vfrsub.vf v8, v2, f0", lambda a2, a1, d, f, r: add(f, neg(a2), r)),
    "wadd_vv":  ("KW", "vfwadd.vv v8, v2, v4", lambda a2, a1, d, f, r: add(a2, a1, r)),
    "wsub_vv":  ("KW", "vfwsub.vv v8, v2, v4", lambda a2, a1, d, f, r: add(a2, neg(a1), r)),
    "wmul_vv":  ("KW", "vfwmul.vv v8, v2, v4", lambda a2, a1, d, f, r: mul(a2, a1, r)),
    "wmacc_vv": ("KW", "vfwmacc.vv v8, v4, v2", lambda a2, a1, d, f, r: add(mul(a1, a2, r), d, r)),
    "wmul_vf":  ("KW", "vfwmul.vf v8, v2, f0", lambda a2, a1, d, f, r: mul(a2, f, r)),
    "wadd_wv":  ("KWW", "vfwadd.wv v8, v16, v4", lambda a2, a1, d, f, r: add(a2, a1, r)),
    "wcvt":     ("KWCVT", ".insn r 0x57, 0x1, 0x25, x8, x13, x2", None),
    "ncvt":     ("KNCVT", ".insn r 0x57, 0x1, 0x25, x8, x29, x16", None),
    "ncvt_sat": ("KNCVT", ".insn r 0x57, 0x1, 0x25, x8, x31, x16", None),
}
# Every test in RNE; these also in the four other modes
ALL_MODES = ("mul_vv", "macc_vv", "wmul_vv", "wmacc_vv", "wcvt", "ncvt")


def rand_scale(rng, centre):
    """Mostly near `centre` (results in range), sometimes anywhere or an edge code."""
    u = rng.random()
    if u < 0.12:
        return rng.choice(EDGE_SCALES)
    if u < 0.35:
        return rng.randint(1, 254)
    return max(1, min(254, centre + rng.randint(-6, 6)))


def rand_bf16(rng, lo=-30, hi=30):
    u = rng.random()
    if u < 0.04:
        return rng.choice([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0])
    v = 2.0 ** rng.uniform(lo, hi) * (1 if rng.random() < 0.5 else -1)
    return encode_float(BF16, round_float(BF16, v, rnd=RoundMode.TiesToEven))


# Block sizes (log2): every test runs with B = 1, and once more with each of
# these, in RNE, alternating the 8-bit format
EXTRA_LB = (1, 3, 5)


def emit(outdir, std, count):
    rng = random.Random(1)
    lines = []
    runs = []
    for alt in (0, 1):
        for name in TESTS:
            for mode in (FRM if name in ALL_MODES else ["rne"]):
                runs.append((name, alt, mode, 0))
    for k, lb in enumerate(EXTRA_LB):
        for n, name in enumerate(TESTS):
            runs.append((name, (k + n) % 2, "rne", lb))
    with open(os.path.join(outdir, "data.S"), "w") as out:
        print_header(out, f"Generated by gen_data.py --std {std} (gfloat reference). Do not edit.")
        for name, alt, mode, lb in runs:
            kind, insn, op = TESTS[name]
            frm, rnd = FRM[mode]
            fi = FP8_STANDARDS[std][f"altfmt{alt}"]
            t = f"{name}_a{alt}_{mode}_b{1 << lb}"
            lines.append(f'{kind}({t}, {frm}, {alt}, {lb}, "{insn}")')
            for suffix, (vals, esize) in make(rng, name, kind, fi, std, op, rnd, lb, count).items():
                print_array(out, t, suffix, vals, esize)
    with open(os.path.join(outdir, "tests.h"), "w") as h:
        h.write(f"// Generated by gen_data.py --std {std}. Do not edit.\n")
        h.write("// kind(name, frm, altfmt, log2 of the block size, instruction)\n")
        h.write("\n".join(lines) + "\n")


def make(rng, name, kind, fi, std, op, rnd, lb, count):
    """The arrays of one test. Scales are compact: element i of a strip takes
    byte i >> lb of its strip's 32 scale bytes; the bytes past 32 >> lb are
    random codes the hardware must ignore."""
    code8 = lambda: rng.randrange(256)
    ex = lambda c: R.exact(fi, c)
    nblk = VL >> lb

    def scales(centres):
        """A strip-major scale array: block b of strip q from centres(q, b)."""
        arr = [code8() for _ in range(count)]                 # unused bytes: anything
        for q in range(count // VL):
            for b in range(nblk):
                arr[q * VL + b] = centres(q, b)
        return arr

    def at(arr, i):
        return arr[(i // VL) * VL + ((i % VL) >> lb)]

    if kind == "KWCVT":
        a = [code8() for _ in range(count)]
        sa = scales(lambda q, b: rand_scale(rng, 128))
        out = [project16(std, block_decode(ex(a[i]), at(sa, i)), rnd) for i in range(count)]
        return {"_a": (a, 1), "_sa": (sa, 1), "_out": (out, 2)}
    if kind == "KNCVT":
        # each block's scale near its largest magnitude, as a quantizer's
        a = [rand_bf16(rng, -140, 140) for _ in range(count)]
        def top(q, b):
            ms = [R.exact(BF16, a[q * VL + i]) for i in range(VL) if i >> lb == b]
            ms = [math.floor(math.log2(m[2])) for m in ms if m[0] == "num" and m[2]]
            return max(ms) if ms else 0
        sr = scales(lambda q, b: rand_scale(rng, 128 + top(q, b) - rng.randint(0, 10)))
        out = [project8(fi, R.exact(BF16, a[i]), at(sr, i), rnd, sat=name.endswith("_sat")) for i in range(count)]
        return {"_a": (a, 2), "_sr": (sr, 1), "_out": (out, 1)}

    sa = scales(lambda q, b: rand_scale(rng, 128))
    sb = scales(lambda q, b: rand_scale(rng, 128))
    adds = name in ("add_vv", "sub_vv", "rsub_vf")
    # result scale near the product's (or the larger addend's) scale
    sr = scales(lambda q, b: rand_scale(rng, max(sa[q * VL + b], sb[q * VL + b]) if adds
                                        else sa[q * VL + b] + sb[q * VL + b] - 128))
    f = [code8() for _ in range(count // VL)]
    a, b, c, out = [], [], [], []
    for i in range(count):
        x2 = rand_bf16(rng) if kind == "KWW" else code8()
        x1 = code8()
        A2 = R.exact(BF16, x2) if kind == "KWW" else block_decode(ex(x2), at(sa, i))
        A1 = block_decode(ex(x1), at(sb, i))
        F = ex(f[i // VL])
        if kind == "K8":
            d = code8()
            D = block_decode(ex(d), at(sr, i))
            out.append(project8(fi, op(A2, A1, D, F, rnd), at(sr, i), rnd))
        else:
            d = rand_bf16(rng) if kind == "KW" else 0
            out.append(project16(std, op(A2, A1, R.exact(BF16, d), F, rnd), rnd))
        a.append(x2); b.append(x1); c.append(d)
    f += [0] * (-len(f) % 4)
    if kind == "K8":
        return {"_a": (a, 1), "_sa": (sa, 1), "_b": (b, 1), "_sb": (sb, 1), "_c": (c, 1),
                "_sr": (sr, 1), "_f": (f, 1), "_out": (out, 1)}
    if kind == "KW":
        return {"_a": (a, 1), "_sa": (sa, 1), "_b": (b, 1), "_sb": (sb, 1), "_c": (c, 2),
                "_f": (f, 1), "_out": (out, 2)}
    return {"_a": (a, 2), "_b": (b, 1), "_sb": (sb, 1), "_out": (out, 2)}  # KWW


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--std", choices=["p3109", "p3109-finite"], default="p3109")
    ap.add_argument("-n", "--count", type=int, default=COUNT)
    ap.add_argument("-o", "--outdir", default=os.path.join(os.path.dirname(__file__), ".."))
    args = ap.parse_args()
    assert args.count % VL == 0
    emit(args.outdir, args.std, args.count)


if __name__ == "__main__":
    main()
