"""Scaled instructions against software baselines: accuracy.

    scaled_accuracy.py

ScaledMultiply and ScaledAdd (P3109 5.8) of two binary8p4 operands with
Binary8p1uf scales, result BF16, three ways:

  scaled   the scaled-mode instruction: exact, rounded once (Saturn's result,
           checked bit for bit by benchmarks/vec-block)
  bf16     software baseline: each operand x * 2^k rounded to BF16, then the
           BF16 operation (vfwcvt, vfmul by 2^k, vfmul/vfadd)
  fp32     the same with FP32 intermediates, the result rounded to BF16

Counts results that differ from the exact one, and how: overflow to Inf, flush
to zero, other. Scale exponents drawn from a typical range (|k| <= 20) and from
the whole Binary8p1uf range (codes 1..254).
"""
import math
import os
import random
import sys
from fractions import Fraction

SAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "common-data-gen")
sys.path.insert(0, SAT)
import fma_ref as R  # noqa: E402
from gfloat import RoundMode, decode_float, encode_float, round_float  # noqa: E402
from gfloat.formats import format_info_binary32, format_info_p3109  # noqa: E402
from gfloat_ref import BF16, external_result  # noqa: E402

P4 = format_info_p3109(8, 4)
FP32 = format_info_binary32
RNE = RoundMode.TiesToEven


def rnd(fi, v):
    """v (an exact value) rounded to fi, as an exact value again."""
    return R.exact(fi, R.project(fi, v, RNE))


def scale(x, k):
    return x if x[0] != "num" else ("num", x[1], x[2] * Fraction(2) ** k)


def bf16_code(v):
    return external_result("p3109", R.project(BF16, v, RNE), 16)


def run(op, krange, n, rng):
    f = R.mul if op == "mul" else R.add
    counts = {"scaled": {}, "bf16": {}, "fp32": {}}
    for _ in range(n):
        while True:
            a, b = rng.randrange(256), rng.randrange(256)
            if a != 0x80 and b != 0x80 and (a & 0x7F) != 0x7F and (b & 0x7F) != 0x7F:
                break
        ka, kb = rng.randint(*krange), rng.randint(*krange)
        A, B = R.exact(P4, a), R.exact(P4, b)
        exact = bf16_code(f(scale(A, ka), scale(B, kb), RNE))
        res = {"scaled": exact}
        for name, fi in (("bf16", BF16), ("fp32", FP32)):
            X, Y = rnd(fi, scale(A, ka)), rnd(fi, scale(B, kb))
            Z = rnd(fi, f(X, Y, RNE)) if fi is FP32 else f(X, Y, RNE)
            res[name] = bf16_code(Z)
        e = decode_float(BF16, exact).fval
        for name, code in res.items():
            if code == exact:
                kind = "correct"
            else:
                v = decode_float(BF16, code).fval
                kind = ("Inf instead of finite" if math.isinf(v) and not math.isinf(e) else
                        "zero instead of nonzero" if v == 0 and e != 0 else
                        "NaN" if math.isnan(v) else "other")
            counts[name][kind] = counts[name].get(kind, 0) + 1
    return counts


if __name__ == "__main__":
    rng = random.Random(3)
    N = 20000
    for op in ("mul", "add"):
        for label, kr in (("typical scales, |k| <= 20", (-20, 20)), ("all scales, k in -127..126", (-127, 126))):
            c = run(op, kr, N, rng)
            print(f"\n== Scaled{'Multiply' if op == 'mul' else 'Add'}, {label} ({N} pairs)")
            for name in ("scaled", "bf16", "fp32"):
                wrong = N - c[name].get("correct", 0)
                detail = ", ".join(f"{k}: {v}" for k, v in sorted(c[name].items()) if k != "correct")
                print(f"   {name:6} wrong {wrong:6} ({100 * wrong / N:5.2f}%)  {detail}")
