"""Exception flags of a rounding into an 8-bit P3109 format, from exact arithmetic.

The flags follow IEEE 754 and the RISC-V fflags semantics, with tininess
detected after rounding (what hardfloat, and so Saturn, implements):

  overflow   the value rounded to the format's precision with an unbounded
             exponent is larger in magnitude than the largest finite value
  underflow  that same unbounded result is smaller in magnitude than the
             smallest normal (tiny), and the delivered result is inexact
  inexact    the delivered result differs from the exact value
  invalid    passed in by the caller (a signalling NaN operand, 0 x Inf, ...)

An infinite or NaN value raises nothing here, and neither does an exact zero.

    p3109_flags(x, fi, mode, invalid=False) -> (invalid, False, overflow, underflow, inexact)

x is an exact value in fma_ref's form: ("nan",) | ("inf", sign) |
("num", sign, magnitude as a Fraction). mode is hardfloat's rounding-mode code
(0 RNE, 1 RTZ, 2 RDN, 3 RUP, 4 RMM, 6 round-to-odd). Saturation changes which
code is delivered but not the flags, so it is not an argument.
"""
from fractions import Fraction

RNE, RTZ, RDN, RUP, RMM, RODD = 0, 1, 2, 3, 4, 6


def _floor_log2(a):
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


def _round_to_grid(mag, sign, q, mode):
    """Round mag (> 0) to a multiple of 2^q in the given mode."""
    s = mag / Fraction(2) ** q
    whole = s.numerator // s.denominator
    rest = s - whole
    if rest:
        up = {RNE: rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1),
              RMM: rest >= Fraction(1, 2),
              RTZ: False,
              RDN: sign == 1,
              RUP: sign == 0,
              RODD: whole % 2 == 0}[mode]
        whole += 1 if up else 0
    return whole * Fraction(2) ** q


def p3109_flags(x, fi, mode, invalid=False):
    if invalid:
        return (True, False, False, False, False)
    if x[0] != "num" or x[2] == 0:
        return (False, False, False, False, False)
    _, sign, mag = x
    P = fi.precision
    emin = 1 - (1 << (8 - P - 1))                 # signed K = 8: bias 2^(K-P-1)

    unbounded = _round_to_grid(mag, sign, _floor_log2(mag) - P + 1, mode)
    overflow = unbounded > Fraction(fi.max)
    tiny = unbounded < Fraction(2) ** emin
    bounded = _round_to_grid(mag, sign, max(_floor_log2(mag), emin) - P + 1, mode)
    inexact = overflow or bounded != mag
    return (False, False, overflow, tiny and inexact, inexact)
