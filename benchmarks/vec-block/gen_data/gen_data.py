#!/usr/bin/env python3
"""Generate vec-block/data.S and vec-block/tests.h: scaled mode (docs/scaled-mode.md).

  ./gen_data.py --std p3109 -o ..        # writes ../data.S and ../tests.h

Each test runs one instruction with vtype's scaled bit set, on COUNT elements.
Every 8-bit register operand vN is a block, its Binary8p1uf scales in vN+1; a
scalar (.vf) or BF16 operand is not. The expected results follow P3109 v4.0.3:

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


def emit(outdir, std, count):
    rng = random.Random(1)
    lines = []
    with open(os.path.join(outdir, "data.S"), "w") as out:
        print_header(out, f"Generated by gen_data.py --std {std} (gfloat reference). Do not edit.")
        for alt in (0, 1):
            fi = FP8_STANDARDS[std][f"altfmt{alt}"]
            for name, (kind, insn, op) in TESTS.items():
                modes = FRM.items() if name in ALL_MODES else [("rne", FRM["rne"])]
                for mode, (frm, rnd) in modes:
                    t = f"{name}_a{alt}_{mode}"
                    lines.append(f'{kind}({t}, {frm}, {alt}, "{insn}")')
                    arrays = make(rng, name, kind, fi, std, op, rnd, count)
                    for suffix, (vals, esize) in arrays.items():
                        print_array(out, t, suffix, vals, esize)
    with open(os.path.join(outdir, "tests.h"), "w") as h:
        h.write(f"// Generated by gen_data.py --std {std}. Do not edit.\n")
        h.write("\n".join(lines) + "\n")


def make(rng, name, kind, fi, std, op, rnd, count):
    code8 = lambda: rng.randrange(256)
    ex = lambda c: R.exact(fi, c)
    if kind == "KWCVT":
        a = [code8() for _ in range(count)]
        sa = [rand_scale(rng, 128) for _ in range(count)]
        out = [project16(std, block_decode(ex(x), s), rnd) for x, s in zip(a, sa)]
        return {"_a": (a, 1), "_sa": (sa, 1), "_out": (out, 2)}
    if kind == "KNCVT":
        # the scale puts most results inside the 8-bit range, as a quantizer would
        a = [rand_bf16(rng, -140, 140) for _ in range(count)]
        sr, out = [], []
        for x in a:
            e = R.exact(BF16, x)
            top = math.floor(math.log2(e[2])) if e[0] == "num" and e[2] else 0
            s = rand_scale(rng, 128 + top - rng.randint(0, 10))
            sr.append(s)
            out.append(project8(fi, e, s, rnd, sat=name.endswith("_sat")))
        return {"_a": (a, 2), "_sr": (sr, 1), "_out": (out, 1)}
    a, sa, b, sb, c, sr, out = [], [], [], [], [], [], []
    f = [code8() for _ in range(count // VL)]
    for i in range(count):
        x2 = rand_bf16(rng) if kind == "KWW" else code8()
        x1 = code8()
        s2, s1 = rand_scale(rng, 128), rand_scale(rng, 128)
        A2 = R.exact(BF16, x2) if kind == "KWW" else block_decode(ex(x2), s2)
        A1 = block_decode(ex(x1), s1)
        F = ex(f[i // VL])
        if kind == "K8":
            d = code8()
            # result scale near the product's (or the larger addend's) scale
            s_r = rand_scale(rng, max(s2, s1) if name in ("add_vv", "sub_vv", "rsub_vf") else s2 + s1 - 128)
            D = block_decode(ex(d), s_r)
            z = op(A2, A1, D, F, rnd)
            c.append(d); sr.append(s_r); out.append(project8(fi, z, s_r, rnd))
        else:
            d = rand_bf16(rng) if kind == "KW" else 0
            z = op(A2, A1, R.exact(BF16, d), F, rnd)
            c.append(d); out.append(project16(std, z, rnd))
        a.append(x2); sa.append(s2); b.append(x1); sb.append(s1)
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
