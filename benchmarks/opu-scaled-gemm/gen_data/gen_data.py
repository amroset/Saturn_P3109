"""Data for opu-scaled-gemm: C = sum over k of (sA x A) x (sB x B).

    gen_data.py [--std p3109 | p3109-finite | ocp] <data.S>

M = N = 64 (a 2x2 kernel of 32x32 tiles), K = 96. A and B are packed for the
OPU (k-major); scales are P3109 Binary8p1uf in every build. The expected C
rounds to FP32 once per k step, as OPFMACC.S does, in the kernel's k order.
Both 8-bit formats of the build. Three tests:

  gemm  block 32: 64 row and 64 column scales per block, reloaded at each
        block, one by vmv1r.v; a zero scale and a NaN scale
  evict two register sets, used on alternate k steps, each with its own
        scales: eight scale registers through a four-entry scale cache, so
        every OPFMACC.S evicts and refills (a stress test of the cache)
  gram  C = (sA x A)(sA x A)^T: A is both operands, so two of the four
        tiles read the same register as vs1 and vs2
  gemm4 (--fp4) P3109 Binary4p2sf pairs, element-format code 6: each byte
        holds steps k (bits 3:0) and k+1 (bits 7:4), each instruction adds
        the exact dot product of a pair, rounded once; block 32, a zero
        scale and one NaN element
"""
import argparse
import os
import random
import sys
from fractions import Fraction
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common-data-gen"))

from gfloat import RoundMode  # noqa: E402
from gfloat.formats import format_info_binary32 as FP32  # noqa: E402
from gfloat_ref import FP8_STANDARDS, binary32_result  # noqa: E402
from fma_ref import add, exact, mul, project  # noqa: E402
from gfloat.formats import format_info_p3109  # noqa: E402
from gfloat.types import Domain, Signedness  # noqa: E402

FP4 = format_info_p3109(4, 2, Signedness.Signed, Domain.Finite)  # Binary4p2sf

M, N, K, BLK = 64, 64, 96, 32
RNE = RoundMode.TiesToEven


def scale(code):
    if code == 255:
        return ("nan",)
    return ("num", 0, Fraction(0) if code == 0 else Fraction(2) ** (code - 128))


def reference(std, fi, a, b, sa, sb):
    """C[i][j] = sum over k of (sa[k][i] a[k][i]) (sb[k][j] b[k][j]), rounded per k."""
    c = []
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                p = mul(mul(scale(sa[k][i]), exact(fi, a[k][i]), RNE),
                        mul(scale(sb[k][j]), exact(fi, b[k][j]), RNE), RNE)
                acc = binary32_result(std, project(FP32, add(p, exact(FP32, acc), RNE), RNE))
            c.append(acc)
    return c


def reference4(std, a, b, sa, sb):
    """FP4 pairs: per instruction kp, the exact sum over k = 2kp, 2kp+1 of
    (sa a[k][i]) (sb b[k][j]), added to C and rounded once."""
    c = []
    for i in range(M):
        for j in range(N):
            acc = 0
            for kp in range(K // 2):
                blk = 2 * kp // BLK
                p0, p1 = (mul(mul(scale(sa[blk][i]), exact(FP4, a[k][i]), RNE),
                              mul(scale(sb[blk][j]), exact(FP4, b[k][j]), RNE), RNE)
                          for k in (2 * kp, 2 * kp + 1))
                p = add(p0, p1, RNE)  # exact
                acc = binary32_result(std, project(FP32, add(p, exact(FP32, acc), RNE), RNE))
            c.append(acc)
    return c


def arrays4(std):
    rng = random.Random("gemm4")
    finite = [c for c in range(16) if c != 8]
    a = [[rng.choice(finite) for _ in range(M)] for _ in range(K)]
    b = [[rng.choice(finite) for _ in range(N)] for _ in range(K)]
    a[50][7] = 8      # one NaN element: row 7 is NaN from k = 50 on
    sa, sb = scales(rng, M, K // BLK), scales(rng, N, K // BLK)
    sa[1][5] = 0      # a zero scale: row 5 adds nothing in block 1
    c = reference4(std, a, b, sa, sb)
    pack = lambda x, n: [x[2 * kp][i] | x[2 * kp + 1][i] << 4 for kp in range(K // 2) for i in range(n)]
    return pack(a, M), pack(b, N), sa, sb, c


def scales(rng, n, blocks):
    return [[rng.randint(118, 138) for _ in range(n)] for _ in range(blocks)]


def arrays(args):
    std, test, altfmt = args
    fi = FP8_STANDARDS[std][f"altfmt{altfmt}"]
    rng = random.Random(f"{test}{altfmt}")
    finite = [c for c in range(256) if exact(fi, c)[0] == "num"]
    a = [[rng.choice(finite) for _ in range(M)] for _ in range(K)]
    b = [[rng.choice(finite) for _ in range(N)] for _ in range(K)]
    if test == "gemm":
        sa, sb = scales(rng, M, K // BLK), scales(rng, N, K // BLK)
        sa[1][5] = 0      # a zero scale: row 5 adds nothing in block 1
        sb[2][40] = 255   # a NaN scale: column 40 is NaN from block 2 on
        ka = [sa[k // BLK] for k in range(K)]
        kb = [sb[k // BLK] for k in range(K)]
    elif test == "evict":
        # sa[2 * blk + parity]: set 0 on even k, set 1 on odd k
        sa, sb = scales(rng, M, 2 * K // BLK), scales(rng, N, 2 * K // BLK)
        ka = [sa[2 * (k // BLK) + k % 2] for k in range(K)]
        kb = [sb[2 * (k // BLK) + k % 2] for k in range(K)]
    else:  # gram
        b, sa = a, scales(rng, M, K // BLK)
        sb = sa
        ka = kb = [sa[k // BLK] for k in range(K)]
    return test, altfmt, a, b, sa, sb, reference(std, fi, a, b, ka, kb)


def emit(f, name, data, kind):
    f.write(f".global {name}\n.balign 64\n{name}:\n")
    per = 32 if kind == "byte" else 8
    for i in range(0, len(data), per):
        f.write(f"    .{kind} " + ", ".join(f"0x{x:0{2 if kind == 'byte' else 8}x}" for x in data[i:i + per]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", choices=sorted(FP8_STANDARDS), default="p3109")
    ap.add_argument("--fp4", action="store_true", help="also the FP4 test, for a config with opuFP4")
    ap.add_argument("output")
    args = ap.parse_args()
    jobs = [(args.std, t, f) for t in ("gemm", "evict", "gram") for f in (0, 1)]
    with Pool() as pool:
        results = pool.map(arrays, jobs)
    with open(args.output, "w") as f:
        f.write(f"# Generated by gen_data.py --std {args.std}. Do not edit.\n"
                ".section .data,\"aw\",@progbits\n")
        for test, altfmt, a, b, sa, sb, c in results:
            p = f"{test}{altfmt}_"
            emit(f, p + "a", [x for row in a for x in row], "byte")
            if test != "gram":
                emit(f, p + "b", [x for row in b for x in row], "byte")
            emit(f, p + "sa", [x for row in sa for x in row], "byte")
            if test != "gram":
                emit(f, p + "sb", [x for row in sb for x in row], "byte")
            emit(f, p + "c", c, "word")
            f.write(f".global {p}out\n.balign 64\n{p}out:\n    .zero {4 * M * N}\n")
        if args.fp4:
            a4, b4, sa, sb, c = arrays4(args.std)
            emit(f, "gemm4_a", a4, "byte")
            emit(f, "gemm4_b", b4, "byte")
            emit(f, "gemm4_sa", [x for row in sa for x in row], "byte")
            emit(f, "gemm4_sb", [x for row in sb for x in row], "byte")
            emit(f, "gemm4_c", c, "word")
            f.write(f".global gemm4_out\n.balign 64\ngemm4_out:\n    .zero {4 * M * N}\n")
