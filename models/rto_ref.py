"""Round-to-odd reference for BF16 -> P3109 conversions, from the standard's text.

gfloat, the reference used for the other rounding modes, has no deterministic
round-to-odd (its RoundMode has only stochastic "odd" variants). So this
transcribes IEEE P3109 v4.0.3 directly:

  4.7.4  omegaRoundToPrecision, with the exponent range unbounded and
           RoundAway(ToOdd) = nu > 0 and CodeIsEven
         where, for P > 1, CodeIsEven = IsEven(floor(S~)).
  4.7.5  omegaSaturate. An out-of-range result is clamped by SatFinite. Under
         SatNone only TowardZero / TowardNegative / TowardPositive may return
         the largest finite value; every other mode -- ToOdd included -- gives
         +/-Inf in the extended domain and NaN in the finite domain.
  4.7.3  omegaProject = round, then saturate, then encode.

All arithmetic is exact (Fractions), so this is independent of both gfloat's
rounding and our bit-level model.
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "benchmarks", "common-data-gen"))
from gfloat import decode_float, encode_float          # noqa: E402
from gfloat_ref import BF16, canonical_nan             # noqa: E402
from fma_ref import _floor_log2                         # noqa: E402


def round_to_odd(X, P, B):
    """4.7.4 for a finite nonzero X (a Fraction): the ToOdd rounded value."""
    ax = abs(X)
    Q = max(_floor_log2(ax), 1 - B) - P + 1       # subnormals via max(., 1-B)
    St = ax / Fraction(2) ** Q                     # real-valued significand
    S = St.numerator // St.denominator             # floor
    nu = St - S
    if nu > 0 and S % 2 == 0:                      # RoundAway(ToOdd), P > 1
        S += 1
    return (1 if X > 0 else -1) * S * Fraction(2) ** Q


def convert_bf16_odd(bits, fi, sat):
    """BF16 bit pattern -> P3109 code, under (ToOdd, SatFinite or SatNone)."""
    P = fi.precision
    B = 1 << (8 - P - 1)                           # 3.1: signed, K = 8
    extended = fi.num_posinfs > 0
    Mhi = Fraction(fi.max)
    Mlo = -Mhi

    x = decode_float(BF16, bits).fval
    if math.isnan(x):
        return canonical_nan(fi)
    if math.isinf(x):
        R = x                                      # 4.7.4 leaves infinities alone
    elif x == 0:
        return 0                                   # P3109 has a single zero
    else:
        R = round_to_odd(Fraction(x), P, B)

    # 4.7.5
    if not isinstance(R, float) and Mlo <= R <= Mhi:
        return encode_float(fi, float(R))
    if sat:
        return encode_float(fi, float(Mhi if R > 0 else Mlo))
    if not extended:
        return canonical_nan(fi)
    return encode_float(fi, math.inf if R > 0 else -math.inf)
