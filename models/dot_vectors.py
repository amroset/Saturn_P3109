"""Test vectors for one ScaledDotLane (vfdot4, docs/vfdot4.md).

    dot_vectors.py <outdir> <std: p3109 | p3109-finite>

Writes dot.bin for tb_dot_lane.cpp, one 20-byte record per lane operation,

    u8 altfmt, u8 frm, u8 sa, u8 sb, u32 a, u32 b, u32 c, u32 expect

a and b hold four 8-bit codes each (element j in bits 8j+7:8j), sa and sb
their Binary8p1uf scales, c the FP32 accumulator. expect is
c + (sa x a0)(sb x b0) + ... + (sa x a3)(sb x b3), each element decoded as
P3109's BlockDecode does, rounded once to FP32 in frm (a zero result is +0).

Per format (altfmt 0, 1):
  pairs   every (x, y) code pair in one lane position (rotating over the four),
          the other elements zero, each with the 11 accumulators of
          opu_vectors.py; frm rotates over the five modes; unscaled
  random  four random codes each (any code, mostly finite), scales from edge
          codes, near 2^0 or random, any frm, c fixed or near the sum
  cancel  three random products and a fourth chosen to cancel most of them,
          sometimes c the exact negative of the sum
  bound   scales that put the scaled sum at FP32's overflow or underflow
          threshold
"""
import math
import os
import random
import struct
import sys
from fractions import Fraction
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))

from gfloat.formats import format_info_binary32 as FP32  # noqa: E402
from gfloat_ref import FP8_STANDARDS, FRM, binary32_result  # noqa: E402
from fma_ref import add, exact, mul, project  # noqa: E402

REC = struct.Struct("<BBBBIIII")
MODES = [rnd for _, rnd in sorted(FRM.values())]  # by frm code 0..4
FIXED_C = (0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
           0xFF7FFFFF, 0x00000001, 0x7F800000, 0xFF800000, 0x7FC00000)
EDGE_S = ((0, 128), (128, 0), (255, 128), (128, 255), (1, 1), (254, 254), (1, 254),
          (254, 1), (127, 129), (129, 127), (64, 200), (200, 64), (2, 128), (128, 128))
N_RANDOM, N_CANCEL, N_BOUND = 200_000, 100_000, 50_000
CHUNK = 10_000


def scale(code):
    if code == 255:
        return ("nan",)
    return ("num", 0, Fraction(0) if code == 0 else Fraction(2) ** (code - 128))


def dot(fi, a, b, sa, sb):
    """The exact scaled sum of the four products."""
    z = ("num", 0, Fraction(0))
    for j in range(4):
        x = mul(scale(sa), exact(fi, a >> 8 * j & 255), None)
        y = mul(scale(sb), exact(fi, b >> 8 * j & 255), None)
        z = add(z, mul(x, y, None), MODES[0])
    return z


def expect(std, fi, a, b, c, sa, sb, frm):
    rnd = MODES[frm]
    z = add(dot(fi, a, b, sa, sb), exact(FP32, c), rnd)
    return binary32_result(std, project(FP32, z, rnd))


def as_float(v):
    if v[0] != "num":
        return None
    return float(v[2]) * (-1 if v[1] else 1)


def near_c(rng, v):
    """A random FP32 value within 2^+-26 of v (an exact value), else anywhere."""
    f = as_float(v)
    e = math.frexp(f)[1] - 1 if f else rng.randint(-20, 20)
    e = max(-126, min(127, e + rng.randint(-26, 26)))
    return rng.getrandbits(1) << 31 | (e + 127) << 23 | rng.getrandbits(23)


def finite_codes(fi):
    return [c for c in range(256) if exact(fi, c)[0] == "num"]


def pack(codes):
    return sum(c << 8 * j for j, c in enumerate(codes))


def rec(std, fi, alt, frm, sa, sb, a, b, c):
    return REC.pack(alt, frm, sa, sb, a, b, c, expect(std, fi, a, b, c, sa, sb, frm))


def job_pairs(args):
    std, alt, x = args
    fi = FP8_STANDARDS[std][f"altfmt{alt}"]
    rng = random.Random(alt << 9 | x)
    out = bytearray()
    for y in range(256):
        k = x << 8 | y
        j = k % 4
        a, b = x << 8 * j, y << 8 * j
        p = dot(fi, a, b, 128, 128)
        for i, c in enumerate(FIXED_C + (near_c(rng, p),)):
            out += rec(std, fi, alt, (k + i) % 5, 128, 128, a, b, c)
    return bytes(out)


def rand_scales(rng):
    u = rng.random()
    if u < 0.2:
        return rng.choice(EDGE_S)
    if u < 0.6:
        return rng.randint(122, 134), rng.randint(122, 134)
    return rng.randint(0, 255), rng.randint(0, 255)


def job_random(args):
    std, alt, part, n = args
    fi = FP8_STANDARDS[std][f"altfmt{alt}"]
    fin = finite_codes(fi)
    rng = random.Random(1 << 20 | alt << 16 | part)
    code = lambda: rng.randrange(256) if rng.random() < 0.15 else rng.choice(fin)
    out = bytearray()
    for _ in range(n):
        a, b = pack([code() for _ in range(4)]), pack([code() for _ in range(4)])
        sa, sb = rand_scales(rng)
        z = dot(fi, a, b, sa, sb)
        c = rng.choice(FIXED_C) if rng.random() < 0.3 else near_c(rng, z)
        out += rec(std, fi, alt, rng.randrange(5), sa, sb, a, b, c)
    return bytes(out)


def job_cancel(args):
    std, alt, part, n = args
    fi = FP8_STANDARDS[std][f"altfmt{alt}"]
    fin = [c for c in finite_codes(fi) if exact(fi, c)[2] != 0]
    val = {c: exact(fi, c) for c in fin}
    rng = random.Random(2 << 20 | alt << 16 | part)
    out = bytearray()
    for _ in range(n):
        xs = [rng.choice(fin) for _ in range(3)]
        ys = [rng.choice(fin) for _ in range(3)]
        partial = dot(fi, pack(xs + [0]), pack(ys + [0]), 128, 128)
        # the fourth product: y at random, x the code closest to -partial / y
        y3 = rng.choice(fin)
        t = -as_float(partial) / as_float(val[y3]) if as_float(partial) else 0.0
        x3 = min(fin, key=lambda c: abs(as_float(val[c]) - t))
        a, b = pack(xs + [x3]), pack(ys + [y3])
        sa, sb = (128, 128) if rng.random() < 0.5 else (rng.randint(100, 156), rng.randint(100, 156))
        z = dot(fi, a, b, sa, sb)
        f = as_float(z)
        f32 = struct.unpack("<f", struct.pack("<f", f))[0] if f is not None and abs(f) < 3e38 else None
        if f32 is not None and rng.random() < 0.2 and Fraction(f32) == Fraction(f):
            c = struct.unpack("<I", struct.pack("<f", -f32))[0]    # exact cancellation: a zero result
        else:
            c = rng.choice((0, 0x80000000)) if rng.random() < 0.3 else near_c(rng, z)
        out += rec(std, fi, alt, rng.randrange(5), sa, sb, a, b, c)
    return bytes(out)


def job_bound(args):
    std, alt, part, n = args
    fi = FP8_STANDARDS[std][f"altfmt{alt}"]
    fin = [c for c in finite_codes(fi) if exact(fi, c)[2] != 0]
    rng = random.Random(3 << 20 | alt << 16 | part)
    out = bytearray()
    for _ in range(n):
        a, b = pack([rng.choice(fin) for _ in range(4)]), pack([rng.choice(fin) for _ in range(4)])
        f = as_float(dot(fi, a, b, 128, 128))
        if not f:
            continue
        target = rng.choice((127, 128, -126, -149, -150))
        total = target - (math.frexp(abs(f))[1] - 1) + 256 + rng.randint(-1, 1)
        lo, hi = max(1, total - 254), min(254, total - 1)
        if lo > hi:
            continue
        sa = rng.randint(lo, hi)
        sb = total - sa
        c = rng.choice((0, 0x80000000, 0x00000001, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF))
        out += rec(std, fi, alt, rng.randrange(5), sa, sb, a, b, c)
    return bytes(out)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("p3109", "p3109-finite"):
        sys.exit("usage: dot_vectors.py <outdir> <p3109 | p3109-finite>")
    std = sys.argv[2]
    split = lambda n: [CHUNK] * (n // CHUNK)
    with Pool() as pool, open(os.path.join(sys.argv[1], "dot.bin"), "wb") as f:
        for alt in (0, 1):
            for chunk in pool.map(job_pairs, [(std, alt, x) for x in range(256)]):
                f.write(chunk)
            for job, n in ((job_random, N_RANDOM), (job_cancel, N_CANCEL), (job_bound, N_BOUND)):
                for chunk in pool.map(job, [(std, alt, i, m) for i, m in enumerate(split(n))]):
                    f.write(chunk)
