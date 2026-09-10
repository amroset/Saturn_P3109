"""Reference model for 8-bit float conversions, backed by gfloat.

Replaces the Spike-based reference in gen_data.c/gen_data.sh.  Spike implements
the OCP FP8 formats but has no IEEE P3109 support, so it cannot be the golden
model once the conversion unit is retargeted.

gfloat has been cross-checked against Spike on all six FP8 arrays of the
checked-in vec-mx-unary/data.S (128/128 each); see validate_gfloat.py.

Requires Python >= 3.10.  gfloat declares >= 3.8.1 but uses match statements.

Spec baseline: IEEE P3109 Interim Report v4.0.3 (1 Sept 2026).
gfloat 0.5.2 (Aug 2025) predates it, so its vocabulary lags the standard:

  saturation   gfloat sat=False  == SatNone     (overflow to +/-Inf, except that
                                                 directed rounding clamps to maxFinite)
               gfloat sat=True   == SatFinite   (clamps +/-Inf too)
               SatPropagate      == not available; emulate by running sat=True and
                                    restoring +/-Inf wherever the input was infinite.

  rounding     gfloat Stochastic        == StochasticC   (verified, sec 4.7.4 formulas)
               gfloat StochasticFast    == StochasticB
               gfloat StochasticFastest == StochasticA
               gfloat StochasticOdd     == none of them (a gfloat extension)
               ToOdd                    == not in gfloat 0.5.2, though sec 4.2 defines it

Beware: "SatFinite" named a *different* mode in report v2.0 (29 Oct 2024), where the
three modes were SatMax / SatFinite / OvfInf.  v4.0.3 renamed and reassigned them.
"""

import math
import random

from gfloat import RoundMode, decode_float, encode_float, round_float
from gfloat.formats import format_info_bfloat16, format_info_ocp_e4m3, format_info_ocp_e5m2
from gfloat.formats import format_info_p3109
from gfloat.types import Domain, Signedness

BF16 = format_info_bfloat16

# RISC-V frm encodings and the gfloat mode each selects.  The two agree 1:1;
# hardfloat's constants happen to use the same numbering, so Saturn passes frm
# straight through to RoundAnyRawFNToRecFN with no translation.
FRM = {
    "rne": (0, RoundMode.TiesToEven),     # nearest, ties to even
    "rtz": (1, RoundMode.TowardZero),
    "rdn": (2, RoundMode.TowardNegative),
    "rup": (3, RoundMode.TowardPositive),
    "rmm": (4, RoundMode.TiesToAway),     # nearest, ties away from zero
}

# altfmt=0 and altfmt=1 respectively, for each supported 8-bit standard.
FP8_STANDARDS = {
    "ocp": {
        "altfmt0": format_info_ocp_e4m3,
        "altfmt1": format_info_ocp_e5m2,
    },
    "p3109": {
        "altfmt0": format_info_p3109(8, 4, Signedness.Signed, Domain.Extended),
        "altfmt1": format_info_p3109(8, 3, Signedness.Signed, Domain.Extended),
    },
}

# Canonical NaN each format is expected to produce.  P3109 has exactly one NaN
# encoding; the OCP formats and BF16 have many, and the hardware emits the
# canonical one.
# RISC-V emits the quiet NaN with zero payload: exponent all ones, significand MSB set.
CANONICAL_NAN = {
    "binary16": 0x7E00,
    "binary32": 0x7FC00000,
    "binary64": 0x7FF8000000000000,
    "bfloat16": 0x7FC0,
    "ocp_e4m3": 0x7F,
    "ocp_e5m2": 0x7F,
}


def canonical_nan(fi):
    if fi.name in CANONICAL_NAN:
        return CANONICAL_NAN[fi.name]
    return fi.code_of_nan  # P3109 has exactly one NaN, so this is canonical


def convert(src_fi, dst_fi, bits, rnd=RoundMode.TiesToEven, sat=False):
    """One conversion, as the hardware should perform it. Returns dst bit pattern."""
    v = decode_float(src_fi, bits).fval
    if math.isnan(v):
        return canonical_nan(dst_fi)
    r = round_float(dst_fi, v, rnd=rnd, sat=sat)
    if math.isnan(r):
        return canonical_nan(dst_fi)
    return encode_float(dst_fi, r)


# ---------------------------------------------------------------------------
# Test vector construction
# ---------------------------------------------------------------------------

def _bf16(v):
    """Nearest BF16 bit pattern to a real value."""
    return _enc(BF16, v)


def _enc(fi, v):
    """Nearest bit pattern in format `fi` to a real value."""
    return encode_float(fi, round_float(fi, v, rnd=RoundMode.TiesToEven))


def narrowing_inputs(dst_fi, count=128, seed=0, src_fi=BF16):
    """Bit patterns in `src_fi` chosen to exercise narrowing into `dst_fi`.

    Deliberately biased toward the encodings where the implementation is
    delicate: the top binade (where a P3109 or OCP encoder must splice a
    wider-exponent rounder in), the overflow threshold, the subnormal
    boundary, exact ties, negative values that round to zero, and the
    extremes of the source format itself.

    The patterns are encoded in `src_fi`, so they must match the source the
    caller converts from -- FP32 for the FP32 -> FP16/BF16 arrays.  Passing
    dst_fi as src_fi probes a format's own edges, which is what the 16-bit
    widening arrays want.
    """
    vals = []

    # Specials.  -0.0 matters: P3109 has no negative zero, so it must flush to
    # +0 rather than land on the NaN code point.
    vals += [_enc(src_fi, 0.0), _enc(src_fi, -0.0)]
    vals += [encode_float(src_fi, float("inf")), encode_float(src_fi, float("-inf"))]
    vals += [CANONICAL_NAN[src_fi.name]]

    # Extremes of the source format.  The groups below track the destination's
    # edges and never reach the source's own top binade.  That binade matters
    # for P3109: the x2 input scaling must not turn it into infinities, because
    # an infinite operand suppresses the rounder's overflow flag -- and the
    # result then changes under SatNone with rounding toward zero, and under
    # SatPropagate always.
    for v in (src_fi.max, 2.0 ** src_fi.emax):
        vals += [_enc(src_fi, v), _enc(src_fi, -v)]

    maxn = dst_fi.max
    minn = dst_fi.smallest_normal
    mins = dst_fi.smallest_subnormal
    emin = round(math.log2(minn))  # gfloat exposes emax but not emin

    # Straddle the overflow threshold in both directions.
    for scale in (0.9, 0.999, 1.0, 1.001, 1.5, 2.0, 1e3):
        vals += [_enc(src_fi, maxn * scale), _enc(src_fi, -maxn * scale)]

    # Inside the top binade: everything at or above 2^emax is where the
    # narrow rounder's exponent field saturates and the splice takes over.
    top = 2.0 ** dst_fi.emax
    for f in (1.0, 1.125, 1.25, 1.5, 1.75, 1.9375):
        vals += [_enc(src_fi, top * f), _enc(src_fi, -top * f)]

    # Subnormal boundary and gradual underflow.
    for v in (minn, minn * 0.5, mins, mins * 0.5, mins * 0.49, mins * 1.5):
        vals += [_enc(src_fi, v), _enc(src_fi, -v)]

    # Exact ties, to separate TiesToEven from TiesToAway.  Halfway between
    # consecutive representable values at a few magnitudes.
    for e in (dst_fi.emax - 1, 0, emin + 1):
        step = 2.0 ** (e - (dst_fi.precision - 1))
        base = 2.0 ** e
        for k in (0, 1, 2):
            vals += [_enc(src_fi, base + k * 2 * step + step), _enc(src_fi, -(base + k * 2 * step + step))]

    # Random fill across the format's full dynamic range.
    rng = random.Random(seed)
    while len(vals) < count:
        mag = math.exp(rng.uniform(math.log(mins / 4), math.log(maxn * 2)))
        vals.append(_enc(src_fi, mag if rng.random() < 0.5 else -mag))

    return vals[:count]


def widening_inputs(src_fi, count=128, seed=0):
    """8-bit patterns for the 8-bit -> BF16 widening path.

    The source space is only 256 wide, so cover all of it when count allows,
    and put the special encodings first.
    """
    specials = [0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x7E, 0xFE]
    rest = [c for c in range(256) if c not in specials]
    random.Random(seed).shuffle(rest)
    order = specials + rest                      # all 256 codes, specials first
    # An 8-bit source has only 256 distinct inputs, so cycle them to fill larger
    # counts -- returning a short array would make main.c read past the end.
    return [order[i % 256] for i in range(count)]


# ---------------------------------------------------------------------------
# data.S emission -- byte-for-byte the format print_array() in mx_data_gen.c
# produces, so main.c consumes it unchanged.
# ---------------------------------------------------------------------------

def print_header(out, note=""):
    if note:
        out.write(f"# {note}\n")
    out.write('.section .data,"aw",@progbits\n')


def print_uint32(out, name, value):
    out.write(f".global {name}\n.balign 64\n{name}:\n")
    out.write(f"    .word 0x{value:08X}\n    .word 0x00000000\n")


def print_array(out, name, suffix, array, esize):
    """Pack `array` little-endian into 32-bit words, esize bytes per element."""
    per = 4 // esize
    out.write(f".global {name}{suffix}\n.balign 64\n{name}{suffix}:\n")
    for i in range(len(array) // per):
        word = "".join(f"{array[i * per + j]:0{esize * 2}X}" for j in range(per - 1, -1, -1))
        out.write(f"    .word 0x{word}\n")
