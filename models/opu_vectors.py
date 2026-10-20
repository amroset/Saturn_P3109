"""Test vectors for the 8-bit multiply-accumulate of one OuterProductCell.

    opu_vectors.py <outdir> <std: ocp | p3109 | p3109-finite> [fp4]

Writes opu.bin for tb_opu_cell.cpp, one 16-byte record per multiply-accumulate,

    u8 fmt, u8 a, u8 b, u8 sa, u8 sb, 3 x u8 0, u32 c, u32 expect

fmt is the element-format code: 0 and 1 are the build's two FP8 formats
(altfmt), 6 is P3109 Binary4p2sf pairs (fp4, configs with opuFP4), where a
and b each hold two FP4 values, step k in bits 3:0 and step k+1 in bits 7:4,
and the product is the exact dot product a0*b0 + a1*b1.

c is the FP32 accumulator before, expect after: c + (sa x a) x (sb x b),
rounded once to FP32 with round-to-nearest-even, which the cell always uses
(a zero result is +0 in a P3109 build).
sa and sb are P3109 Binary8p1uf block scales (0 is zero, 255 is NaN, otherwise
2^(code - 128)), applied exactly as P3109's BlockDecode does.

Unscaled (sa = sb = 128, as OPFMACC): every (a, b) pair of both formats, each with
  * fixed accumulators: +-0, +-1, +-maxFinite, the smallest subnormal, +-Inf, NaN
  * one random accumulator within 2^+-26 of the product, which makes the
    addition round.
Scaled (OPFMACC.S): every (a, b) pair of both formats, each with six scale
pairs: two from a list of edge codes, two random, and two that put the scaled
product at FP32's overflow and underflow boundaries; each with one of the
accumulators above.
FP4 (fp4): every (a0, a1, b0, b1), unscaled and scaled as above.
"""
import math
import os
import random
import struct
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))

from gfloat import RoundMode  # noqa: E402
from gfloat.formats import format_info_binary32 as FP32  # noqa: E402
from gfloat_ref import FP8_STANDARDS, binary32_result  # noqa: E402
from fractions import Fraction  # noqa: E402
from fma_ref import add, exact, mul, project  # noqa: E402
from gfloat.formats import format_info_p3109  # noqa: E402
from gfloat.types import Domain, Signedness  # noqa: E402

FP4 = format_info_p3109(4, 2, Signedness.Signed, Domain.Finite)  # Binary4p2sf
FMT_FP4 = 6

RNE = RoundMode.TiesToEven
FIXED_C = (0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
           0xFF7FFFFF, 0x00000001, 0x7F800000, 0xFF800000, 0x7FC00000)


def near_c(rng, p):
    """A random FP32 value within 2^+-26 of the product p."""
    # a product of two 8-bit values is exact in a float
    e = math.frexp(float(p[2]))[1] - 1 if p[0] == "num" and p[2] != 0 else 0
    e = max(-126, min(127, e + rng.randint(-26, 26)))
    return rng.getrandbits(1) << 31 | (e + 127) << 23 | rng.getrandbits(23)


EDGE_S = ((0, 128), (128, 0), (255, 128), (128, 255), (0, 255), (1, 1), (254, 254),
          (1, 254), (254, 1), (127, 129), (129, 127), (64, 200), (200, 64), (2, 128))


def scale(code):
    """Decode a Binary8p1uf block scale exactly."""
    if code == 255:
        return ("nan",)
    return ("num", 0, Fraction(0) if code == 0 else Fraction(2) ** (code - 128))


def boundary_scales(rng, p, target):
    """A scale pair putting |p| x 2^(sa + sb - 256) near 2^target, or random."""
    if p[0] != "num" or p[2] == 0:
        return rng.randint(0, 255), rng.randint(0, 255)
    total = target - (math.frexp(float(p[2]))[1] - 1) + 256 + rng.randint(-1, 1)
    lo = max(1, total - 254)
    hi = min(254, total - 1)
    if lo > hi:
        return rng.randint(1, 254), rng.randint(1, 254)
    sa = rng.randint(lo, hi)
    return sa, total - sa


def records(args):
    std, altfmt, a_hi, scaled = args
    fi = FP8_STANDARDS[std][f"altfmt{altfmt}"]
    rng = random.Random(altfmt << 9 | scaled << 8 | a_hi)
    out = bytearray()
    for a in range(a_hi << 4, (a_hi + 1) << 4):
        for b in range(256):
            if not scaled:
                p = mul(exact(fi, a), exact(fi, b), RNE)
                for c in FIXED_C + (near_c(rng, p),):
                    want = binary32_result(std, project(FP32, add(p, exact(FP32, c), RNE), RNE))
                    out += struct.pack("<BBBBBxxxII", altfmt, a, b, 128, 128, c, want)
                continue
            p0 = mul(exact(fi, a), exact(fi, b), RNE)
            k = a << 8 | b
            pairs = [EDGE_S[(2 * k) % len(EDGE_S)], EDGE_S[(2 * k + 1) % len(EDGE_S)],
                     (rng.randint(0, 255), rng.randint(0, 255)),
                     (rng.randint(0, 255), rng.randint(0, 255)),
                     boundary_scales(rng, p0, 127), boundary_scales(rng, p0, -149)]
            for i, (sa, sb) in enumerate(pairs):
                p = mul(mul(scale(sa), exact(fi, a), RNE), mul(scale(sb), exact(fi, b), RNE), RNE)
                c = (FIXED_C + (near_c(rng, p),))[(k * 6 + i) % (len(FIXED_C) + 1)]
                want = binary32_result(std, project(FP32, add(p, exact(FP32, c), RNE), RNE))
                out += struct.pack("<BBBBBxxxII", altfmt, a, b, sa, sb, c, want)
    return bytes(out)


def records4(args):
    """FP4 pairs: a = a0 | a1 << 4, b = b0 | b1 << 4, every (a0, a1, b0, b1)."""
    std, a0, scaled = args
    rng = random.Random(FMT_FP4 << 9 | scaled << 8 | a0)
    out = bytearray()
    for a1 in range(16):
        for b in range(256):
            a, b0, b1 = a0 | a1 << 4, b & 15, b >> 4
            def dot(sa, sb):
                x0, x1 = mul(scale(sa), exact(FP4, a0), RNE), mul(scale(sa), exact(FP4, a1), RNE)
                y0, y1 = mul(scale(sb), exact(FP4, b0), RNE), mul(scale(sb), exact(FP4, b1), RNE)
                return add(mul(x0, y0, RNE), mul(x1, y1, RNE), RNE)  # exact
            if not scaled:
                p = dot(128, 128)
                for c in FIXED_C + (near_c(rng, p),):
                    want = binary32_result(std, project(FP32, add(p, exact(FP32, c), RNE), RNE))
                    out += struct.pack("<BBBBBxxxII", FMT_FP4, a, b, 128, 128, c, want)
                continue
            p0 = dot(128, 128)
            k = a << 8 | b
            pairs = [EDGE_S[(2 * k) % len(EDGE_S)], EDGE_S[(2 * k + 1) % len(EDGE_S)],
                     (rng.randint(0, 255), rng.randint(0, 255)),
                     (rng.randint(0, 255), rng.randint(0, 255)),
                     boundary_scales(rng, p0, 127), boundary_scales(rng, p0, -149)]
            for i, (sa, sb) in enumerate(pairs):
                p = dot(sa, sb)
                c = (FIXED_C + (near_c(rng, p),))[(k * 6 + i) % (len(FIXED_C) + 1)]
                want = binary32_result(std, project(FP32, add(p, exact(FP32, c), RNE), RNE))
                out += struct.pack("<BBBBBxxxII", FMT_FP4, a, b, sa, sb, c, want)
    return bytes(out)


if __name__ == "__main__":
    fp4 = sys.argv[3:] == ["fp4"]
    if len(sys.argv) not in (3, 4) or sys.argv[2] not in FP8_STANDARDS or (len(sys.argv) == 4 and not fp4):
        sys.exit(f"usage: opu_vectors.py <outdir> <{' | '.join(FP8_STANDARDS)}> [fp4]")
    with Pool() as pool, open(os.path.join(sys.argv[1], "opu.bin"), "wb") as f:
        for chunk in pool.map(records, [(sys.argv[2], f, a_hi, scaled)
                                        for scaled in (0, 1) for f in (0, 1) for a_hi in range(16)]):
            f.write(chunk)
        if fp4:
            for chunk in pool.map(records4, [(sys.argv[2], a0, scaled) for scaled in (0, 1) for a0 in range(16)]):
                f.write(chunk)
