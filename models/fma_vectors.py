"""Test vectors for SegmentedFMAPipe in scaled mode (docs/scaled-mode.md): the
lanes, widened to 10 exponent bits, that compute 8-bit and BF16 results of
block operands.

    fma_vectors.py <outdir> <std: p3109 | p3109-finite>

Writes fma.bin for tb_fma_lane.cpp, one 80-byte record per cycle,

    u8 cat, alt, frm, op, addsub, mul, a_eew, b_eew, c_eew, out_eew, widen,
       chkexc, 4 x u8 0
    u64 a, b, c, expect, exc
    u8 scale[8][4]       lane k's (sa, sb, sc, sr), the pins io_scale_k_*

and fma.cats, the category names by cat. An 8-bit result has 8 lanes (byte k
of a, b, c and the result); a BF16 result has 4 (lane j's 8-bit operands in
byte 2j, its scales in scale[2j]). exc holds lane k's flags in byte k, checked
where chkexc bit k is set: 8-bit results whose scales are all 1..254 (the
flags of a zero or NaN result scale are not architectural).

The lanes compute, with s0 = op bit 0 and s1 = op bit 1,
    fma     (-1)^s1 A B + (-1)^s0 C
    addsub  (-1)^s1 A + (-1)^s0 C     (the pipe supplies b = 1)
    mul     A B                       (the pipe supplies c = 0)
each 8-bit operand decoded as P3109's BlockDecode does (5.4.1), and round once:
an 8-bit result as BlockProject (5.4.2) with sr, a BF16 result to BF16 (sr
unused, zero +0). The unused inputs (b of addsub, c of mul, their scales, sr
of a BF16 result, the odd scale slots of a BF16 result) hold random values.

Per format (altfmt 0, 1), every (x, y) code pair, rotated over the lanes:
  8-bit result   x, y as a, b of an fma with 8 kinds of c and scales
                 (unscaled with special or nearby c, edge scale codes, random
                 scales with the result in range, scales at the overflow and
                 underflow thresholds), as a, c of an addsub, as a, b of a mul
  BF16 result    x, y as a, b of a mul, as a, c of an addsub, as a, b of an
                 fma with a BF16 c (vfwmul, vfwadd, vfwmacc)
plus a sample of the .wv form, a BF16 a and an 8-bit c (vfwadd.wv), with
zeros and exact cancellations.
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
sys.path.insert(0, HERE)

from gfloat import RoundMode, encode_float, round_float  # noqa: E402
from gfloat_ref import BF16, FP8_STANDARDS, FRM, external_result  # noqa: E402
from fma_ref import _negate as neg, add, exact, mul, project  # noqa: E402
from flags_ref import flags, pack as pack_flags  # noqa: E402

REC = struct.Struct("<16BQQQQQ32B")
MODES = [rnd for _, rnd in sorted(FRM.values())]  # by frm code 0..4
EDGE = (0, 1, 2, 127, 128, 129, 253, 254, 255)
ZERO = ("num", 0, Fraction(0))
N_WV = 20_000
CATS = ["fma8 unscaled, special c", "fma8 unscaled, nearby c", "fma8 edge scales",
        "fma8 edge scales", "fma8 random scales", "fma8 random scales",
        "fma8 overflow threshold", "fma8 underflow threshold", "add8", "mul8",
        "wmul (BF16)", "wadd (BF16)", "wmacc (BF16)", "wadd.wv (BF16)"]
NARROW, WIDE = range(10), range(10, 13)


def block_decode(x, s):
    if s == 255:
        return ("nan",)
    if s == 0:
        return ("nan",) if x[0] != "num" else ZERO
    return x if x[0] != "num" else ("num", x[1], x[2] * Fraction(2) ** (s - 128))


def project8(fi, z, sr, rnd):
    if sr == 255 or z[0] == "nan":
        return project(fi, ("nan",), rnd)
    if sr == 0:
        return project(fi, ZERO, rnd)
    if z[0] == "inf":
        return project(fi, z, rnd)
    return project(fi, ("num", z[1], z[2] / Fraction(2) ** (sr - 128)), rnd)


def lane_z(kind, op, A, B, C, rnd):
    if kind == "mul":
        return mul(A, B, rnd)
    p = A if kind == "addsub" else mul(A, B, rnd)
    return add(neg(p) if op & 2 else p, neg(C) if op & 1 else C, rnd)


def lg(v):
    return None if v[0] != "num" or v[2] == 0 else math.floor(math.log2(v[2]))


class Fmt:
    def __init__(self, std, alt):
        self.std, self.fi = std, FP8_STANDARDS[std][f"altfmt{alt}"]
        self.val = [exact(self.fi, c) for c in range(256)]
        self.fin = [c for c in range(256) if self.val[c][0] == "num" and self.val[c][2]]
        self.by_lg = {}
        for c in self.fin:
            self.by_lg.setdefault(lg(self.val[c]), []).append(c)
        self.special = [0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x7E, 0xFE, 0x40, 0xC0]
        self.lg_max, self.lg_min = math.floor(math.log2(self.fi.max)), round(math.log2(self.fi.smallest_subnormal))

    def near(self, rng, e):
        """An 8-bit code of random sign with exponent within 3 of e, else any finite code."""
        cs = [c for d in range(-3, 4) for c in self.by_lg.get(e + d, [])] if e is not None else []
        return rng.choice(cs or self.fin) ^ (rng.getrandbits(1) << 7)


def rand_scale(rng):
    return rng.choice(EDGE) if rng.random() < 0.1 else rng.randint(1, 254)


def clip(s):
    return max(1, min(254, s))


def rand_bf16(rng, e=None):
    u = rng.random()
    if u < 0.05:
        return rng.choice([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x7F7F])
    e = rng.randint(-40, 40) if e is None else e + rng.randint(-4, 4)
    v = 2.0 ** e * (1 + rng.random()) * (1 if rng.random() < 0.5 else -1)
    return encode_float(BF16, round_float(BF16, v, rnd=RoundMode.TiesToEven))


def bf16_of(v):
    """The BF16 pattern of the exact value v, or None if BF16 cannot hold it."""
    if v[0] != "num":
        return None
    f = float(v[2]) * (-1 if v[1] else 1)
    try:
        r = round_float(BF16, f)
    except (OverflowError, ValueError):
        return None
    if not (math.isfinite(f) and math.isfinite(r)) or Fraction(r) != v[2] * (-1 if v[1] else 1):
        return None
    return encode_float(BF16, r)


def narrow_lane(F, rng, j, x, y):
    """One lane of 8-bit-result category j for the pair (x, y):
    (kind, a, b, c, scales, the scales the result depends on)."""
    v = F.val
    kind = "addsub" if j == 8 else "mul" if j == 9 else "fma"
    a, b = x, y
    c = rng.randrange(256)
    s = [128, 128, 128, 128]
    if j == 0:
        c = F.special[(x + y) % len(F.special)]
    elif j == 1:
        p = mul(v[a], v[b], None)
        c = F.near(rng, lg(p))
        if rng.random() < 0.1 and lg(p) is not None:
            cs = [k for k in F.fin if v[k][2] == p[2] and v[k][1] != p[1]]
            c = cs[0] if cs else c                              # exact cancellation
    elif j in (2, 3):
        s = [rng.choice(EDGE) for _ in range(4)]
    elif j in (4, 5):
        s[0], s[1] = rng.randint(100, 156), rng.randint(100, 156)
        p = mul(v[a], v[b], None)
        e = lg(p)
        s[3] = clip(s[0] + s[1] - 128 + rng.randint(-4, 4))
        s[2] = rng.randint(100, 156)
        c = F.near(rng, None if e is None else e + s[0] + s[1] - s[2] - 128)
    elif j in (6, 7):
        s[0], s[1], s[2] = rng.randint(100, 156), rng.randint(100, 156), rng.randint(1, 254)
        e = lg(mul(v[a], v[b], None))
        target = F.lg_max if j == 6 else F.lg_min
        s[3] = rand_scale(rng) if e is None else clip(e + s[0] + s[1] - 128 - target + 128 + rng.randint(-1, 1))
        c = rng.choice((0x00, 0x01, 0x81, F.near(rng, None if e is None else e + s[0] + s[1] - s[2] + rng.randint(-2, 0) - 128)))
    elif j == 8:
        a, c, b = x, y, rng.randrange(256)
        s = [rand_scale(rng) for _ in range(3)] + [0]
        s[3] = clip(max(s[0], s[2]) + rng.randint(-3, 3)) if rng.random() < 0.9 else rng.choice(EDGE)
    else:
        s = [rand_scale(rng), rand_scale(rng), rng.randrange(256), 0]
        s[3] = clip(s[0] + s[1] - 128 + rng.randint(-3, 3)) if rng.random() < 0.9 else rng.choice(EDGE)
    used = [0, 2, 3] if kind == "addsub" else [0, 1, 3] if kind == "mul" else [0, 1, 2, 3]
    if kind == "addsub":
        s[1] = rng.randrange(256)
    return kind, a, b, c, s, used


def narrow_job(args):
    std, alt, j, g0, g1 = args
    F, out = Fmt(std, alt), bytearray()
    rng = random.Random(alt << 24 | j << 18 | g0)
    for g in range(g0, g1):
        frm, op = (g + j) % 5, rng.randrange(4)
        lanes = [None] * 8
        for i in range(8):
            p = 8 * g + i
            lanes[(i + j) % 8] = narrow_lane(F, rng, j, p >> 8, p & 255)
        kind = lanes[0][0]
        lop = 0 if kind == "mul" else op
        wa = wb = wc = we = wx = 0
        chk = 0
        scales = []
        for k, (_, a, b, c, s, used) in enumerate(lanes):
            v = F.val
            A, B, C = block_decode(v[a], s[0]), block_decode(v[b], s[1]), block_decode(v[c], s[2])
            z = lane_z(kind, lop, A, B, C, MODES[frm])
            sr = s[3]
            e = project8(F.fi, z, sr, MODES[frm])
            if all(1 <= s[u] <= 254 for u in used):
                srv = z if z[0] != "num" else ("num", z[1], z[2] / Fraction(2) ** (sr - 128))
                ops = [A, B] if kind == "mul" else [A, C] if kind == "addsub" else [A, B, C]
                invalid = z[0] == "nan" and all(o[0] != "nan" for o in ops)
                wx |= pack_flags(flags(srv, F.fi, frm, invalid=invalid)) << 8 * k
                chk |= 1 << k
            wa |= a << 8 * k; wb |= b << 8 * k; wc |= c << 8 * k; we |= e << 8 * k
            scales += s
        out += REC.pack(j, alt, frm, lop, kind == "addsub", kind == "mul", 0, 0, 0, 0, 0, chk, 0, 0, 0, 0,
                        wa, wb, wc, we, wx, *scales)
    return bytes(out)


def wide_rec(F, rng, cat, frm, op, kind, lanes):
    """lanes: 4 tuples (a, b, c, sa, sb, sc) with a and c 8-bit or BF16 per a_eew, c_eew."""
    a_eew, b_eew, c_eew = {"mul": (0, 0, 0), "addsub": (0, 1, 0), "fma": (0, 0, 1), "wv": (1, 1, 0)}[kind]
    rnd = MODES[frm]
    wa = wb = wc = we = 0
    scales = [rng.randrange(256) for _ in range(32)]
    for j, (a, b, c, sa, sb, sc) in enumerate(lanes):
        A = exact(BF16, a) if a_eew else block_decode(F.val[a], sa)
        B = block_decode(F.val[b], sb)
        C = exact(BF16, c) if c_eew else block_decode(F.val[c], sc)
        z = lane_z("addsub" if kind == "wv" else kind, op, A, B, C, rnd)
        e = external_result(F.std, project(BF16, z, rnd), 16)
        wa |= a << 16 * j
        wb |= (0x3F80 if b_eew else b) << 16 * j
        wc |= c << 16 * j
        we |= e << 16 * j
        scales[8 * j:8 * j + 3] = [sa, sb, sc]
    return REC.pack(cat, F.alt, frm, op, kind in ("addsub", "wv"), kind == "mul", a_eew, b_eew, c_eew, 1, 1,
                    0, 0, 0, 0, 0, wa, wb, wc, we, 0, *scales)


def wide_scales(rng):
    if rng.random() < 0.5:
        return [rng.randint(118, 138) for _ in range(3)]
    return [rand_scale(rng) for _ in range(3)]


def wide_job(args):
    std, alt, j, g0, g1 = args
    F, out = Fmt(std, alt), bytearray()
    F.alt = alt
    rng = random.Random(1 << 30 | alt << 24 | j << 18 | g0)
    kind = {10: "mul", 11: "addsub", 12: "fma"}[j]
    for g in range(g0, g1):
        frm, op = (g + j) % 5, 0 if kind == "mul" else rng.randrange(4)
        lanes = [None] * 4
        for i in range(4):
            p = 4 * g + i
            x, y = p >> 8, p & 255
            sa, sb, sc = wide_scales(rng)
            if kind == "mul":
                t = (x, y, 0, sa, sb, 128)
            elif kind == "addsub":
                t = (x, 0, y, sa, rng.randrange(256), sb)
            else:
                prod = mul(block_decode(F.val[x], sa), block_decode(F.val[y], sb), None)
                c = bf16_of(neg(prod)) if rng.random() < 0.05 else None
                t = (x, y, c if c is not None else rand_bf16(rng, lg(prod)), sa, sb, 128)
            lanes[(i + j) % 4] = t
        out += wide_rec(F, rng, j, frm, op, kind, lanes)
    return bytes(out)


def wv_job(args):
    std, alt, part, n = args
    F, out = Fmt(std, alt), bytearray()
    F.alt = alt
    rng = random.Random(2 << 30 | alt << 24 | part)
    for _ in range(n // 4):
        lanes = []
        for _ in range(4):
            c, sc = rng.randrange(256), rand_scale(rng) if rng.random() < 0.5 else rng.randint(118, 138)
            u = rng.random()
            C = block_decode(F.val[c], sc)
            if u < 0.15:
                a = rng.choice((0x0000, 0x8000))
                c = rng.choice((0x00, c))
            elif u < 0.35 and bf16_of(neg(C)) is not None:
                a = bf16_of(neg(C))                                # exact cancellation
            else:
                a = rand_bf16(rng, lg(C))
            lanes.append((a, 0, c, 128, rng.randrange(256), sc))
        out += wide_rec(F, rng, 13, rng.randrange(5), rng.randrange(4), "wv", lanes)
    return bytes(out)


def run_job(job):
    fn, args = job
    return fn(args)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("p3109", "p3109-finite"):
        sys.exit("usage: fma_vectors.py <outdir> <p3109 | p3109-finite>")
    outdir, std = sys.argv[1], sys.argv[2]
    jobs = []
    for alt in (0, 1):
        jobs += [(narrow_job, (std, alt, j, g, g + 256)) for j in NARROW for g in range(0, 8192, 256)]
        jobs += [(wide_job, (std, alt, j, g, g + 512)) for j in WIDE for g in range(0, 16384, 512)]
        jobs += [(wv_job, (std, alt, part, 2000)) for part in range(N_WV // 2000)]
    with Pool(int(os.environ.get("JOBS", "8"))) as pool, open(os.path.join(outdir, "fma.bin"), "wb") as f:
        for chunk in pool.imap(run_job, jobs):
            f.write(chunk)
    with open(os.path.join(outdir, "fma.cats"), "w") as f:
        f.write("\n".join(CATS) + "\n")
