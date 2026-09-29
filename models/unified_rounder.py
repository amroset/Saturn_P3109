"""One rounder for both P3109 formats, derived from hardfloat's RoundAnyRawFNToRecFN.

WHAT THIS REPLACES
    Today each 8-bit output needs four hardfloat rounders: a narrow one
    (E4M3 / E5M2) that is right everywhere except the top row, a wide one
    (E5M3 / E6M2) that supplies the top row, and the same pair again for the
    other format -- plus the x2 that shifts P3109's bias onto IEEE's, and two
    assemblers that splice the pairs together and fix up the code points.

    This model is one rounder that produces the 8-bit P3109 code directly, with
    the format (binary8p4 / binary8p3) as a runtime input.

WHY ONE ROUNDER IS ENOUGH -- the things that forced the split

 1. THE SPLICE exists because hardfloat's rounder reserves exponent-all-ones
    for Inf/NaN: it tests `sRoundedExp >> (outExpWidth-1) >= 3` for overflow,
    which is a statement about the *recoded tag*, not about the format's real
    range.  P3109 puts finite numbers there.  Here that test is replaced by an
    explicit comparison against the format's own maxFinite, so the top row is
    ordinary and no second rounder is needed.

 2. THE TWO FORMATS differ by one bit of precision (binary8p4 keeps 3 fraction
    bits, binary8p3 keeps 2).  hardfloat cannot vary that at run time because
    outSigWidth is a Scala parameter -- but the *mechanism* is already in there.
    `roundMask` marks the bits being discarded, and it already widens by one
    bit per step as the exponent falls below minNormExp: that is how subnormals
    are rounded without a shifter.  Losing a bit of precision is the same
    operation, so the format select is two small constants applied to that mask.

        delta       moves the exponent at which the mask starts widening, so it
                    matches this format's own smallest normal number.
        prec_shift  adds the bits of precision this format does not have.  It
                    CANNOT be folded into delta: the mask decoder saturates at
                    zero, so a constant added to the exponent vanishes for every
                    normal number.  It is applied as a shift with the low bits
                    filled, which survives the clamp.

    US 6,490,607 (Oberman, AMD, filed 1999, expired 2019) makes the same move
    for single/double/extended precision, where the format selects which
    rounding constant is added and where it sits (FIG. 6).  The patent can use
    a constant at a fixed position because its subnormal results trap to
    microcode; ours must round subnormals in hardware, so the position has to
    move with the exponent -- which is what the mask already does.

 3. THE x2 DISAPPEARS.  It existed only to make P3109's bias line up with the
    bias baked into the IEEE-shaped rounders.  Here the bias is ours, so the
    exponent constants below are P3109's own and nothing is doubled.

STRUCTURE
    Everything from `adjusted_sig` to `s_rounded_exp` is a transcription of
    RoundAnyRawFNToRecFN, line for line, so the rounding itself is hardfloat's
    tested logic.  The generalisations are marked "CHANGED".  The encoder at the
    end is new: hardfloat emits a recoded number (which has no subnormals), and
    we need a P3109 8-bit code (which does).

VALIDATION
    validate_unified.py      conversions: every BF16 input x 5 modes x 2 formats
                             x 2 domains x sat on/off  -- 5,242,880 cases, clean
    validate_unified_fma.py  FMA results: every 8-bit operand pair x 3 ops x 2
                             formats x 5 core sizes, plus modes/domains/sat and
                             both significand normalisations -- clean
"""

import sys

sys.path.insert(0, "/home/amroset/Thesis/Chipyard_P3109/generators/saturn/benchmarks/common-data-gen")

# ---------------------------------------------------------------------------
# hardfloat primitives, ported
# ---------------------------------------------------------------------------


def low_mask(val, width, top_bound, bottom_bound):
    """Port of hardfloat's lowMask (primitives.scala:45).

    Returns a (|top-bottom|)-bit thermometer mask.  Only the base case is
    transcribed; the recursive divide-and-conquer branch above 64 values is a
    synthesis optimisation that computes the same function.
    """
    num_in_vals = 1 << width
    if top_bound < bottom_bound:
        return low_mask(~val & (num_in_vals - 1), width,
                        num_in_vals - 1 - top_bound, num_in_vals - 1 - bottom_bound)
    shift = (-1 << num_in_vals) >> val          # Python >> on negatives is arithmetic
    hi, lo = num_in_vals - 1 - bottom_bound, num_in_vals - top_bound
    field = (shift >> lo) & ((1 << (hi - lo + 1)) - 1)
    out, n = 0, hi - lo + 1                     # Reverse()
    for i in range(n):
        if field & (1 << i):
            out |= 1 << (n - 1 - i)
    return out


def raw_from_fn(bits, exp_width, sig_width):
    """Port of rawFloatFromFN: an IEEE bit pattern -> hardfloat's RawFloat.

    sig is (sig_width+1) bits, '0 ## 1 ## fraction' for a normal number, with
    subnormals normalised by the shift below.  sExp is biased by 2^exp_width.
    """
    sign = (bits >> (exp_width + sig_width - 1)) & 1
    exp_in = (bits >> (sig_width - 1)) & ((1 << exp_width) - 1)
    fract_in = bits & ((1 << (sig_width - 1)) - 1)

    is_zero_exp = exp_in == 0
    is_zero_fract = fract_in == 0

    norm_dist = (sig_width - 1) - fract_in.bit_length() if fract_in else sig_width - 1
    subnorm_fract = ((fract_in << norm_dist) & ((1 << (sig_width - 2)) - 1)) << 1
    adjusted_exp = (
        (norm_dist ^ ((1 << (exp_width + 1)) - 1)) if is_zero_exp else exp_in
    ) + ((1 << (exp_width - 1)) | (2 if is_zero_exp else 1))

    is_zero = is_zero_exp and is_zero_fract
    is_special = (adjusted_exp >> (exp_width - 1)) & 3 == 3

    return dict(
        isNaN=is_special and not is_zero_fract,
        isInf=is_special and is_zero_fract,
        isZero=is_zero,
        sign=sign,
        sExp=adjusted_exp & ((1 << (exp_width + 1)) - 1),
        sig=(0 if is_zero else 1 << (sig_width - 1))
            | (subnorm_fract if is_zero_exp else fract_in),
    )


# ---------------------------------------------------------------------------
# the unified rounder
# ---------------------------------------------------------------------------

# Rounding modes, hardfloat's encoding (== RISC-V frm for 0..4).
RNE, RTZ, RDN, RUP, RMM, RODD = 0, 1, 2, 3, 4, 6

# Internal datapath, sized for the wider of the two formats.
SIG_INT = 4        # hardfloat 'outSigWidth' terms: binary8p4's precision
B_INT = 256        # internal exponent bias, 2^8 (BF16's own, so conversions re-bias by 0)
EXP_SLICE = 10     # bits of exponent fed to the mask decoder

# The mask decoder is built around binary8p4's minNormExp.  binary8p3 reaches it
# through `delta` instead of a second decoder.
MIN_NORM_REF = B_INT - 7                       # binary8p4: emin = 1 - 8 = -7
MASK_TOP = MIN_NORM_REF - SIG_INT - 1          # lowMask bounds, as in hardfloat
MASK_BOTTOM = MIN_NORM_REF

FMT = {
    # binary8pP in 8 bits: 1 sign, (8-P) exponent, (P-1) fraction, bias 2^(7-P).
    # Two knobs carry the format:
    #   delta       lines this format's minNormExp up with the one the mask
    #               decoder was built around, so the mask starts widening at the
    #               right exponent.  delta = MIN_NORM_REF - min_norm.
    #   prec_shift  how many bits of precision this format gives up relative to
    #               the internal datapath.  It CANNOT be folded into delta: the
    #               decoder's thermometer saturates at zero, so a constant added
    #               to the exponent disappears for every normal number.  It is
    #               applied as a shift-with-fill instead, which survives the
    #               clamp.  prec_shift = SIG_INT - P.
    "p4": dict(
        frac_bits=3, exp_bits=4, bias=8,
        min_norm=B_INT - 7,        # emin = -7
        min_nonzero=B_INT - 10,    # emin - (P-1) = -10, hardfloat's outMinNonzeroExp
        emax=B_INT + 7,
        delta=0, prec_shift=0,     # the datapath is built for this format
    ),
    "p3": dict(
        frac_bits=2, exp_bits=5, bias=16,
        min_norm=B_INT - 15,       # emin = -15
        min_nonzero=B_INT - 17,
        emax=B_INT + 15,
        delta=8, prec_shift=1,     # 249 - 241 = 8, and one fewer fraction bit
    ),
}


def unified_round(raw, fmt, mode, sat=False, finite=False,
                  in_exp_width=8, in_sig_width=8,
                  invalid_exc=False, sig_msb_always_zero=True,
                  detect_tininess_after=True):
    """Round a hardfloat RawFloat straight to an 8-bit P3109 code.

    fmt     "p4" | "p3"          -- a wire in hardware (altfmt)
    finite  domain               -- a build-time choice, as today
    sat     saturating variant   -- the .sat instruction bit
    Returns (code, flags) with flags = (invalid, infinite, overflow, underflow, inexact).
    """
    f = FMT[fmt]

    near_even, near_max = mode == RNE, mode == RMM
    odd = mode == RODD
    round_mag_up = (mode == RDN and raw["sign"]) or (mode == RUP and not raw["sign"])

    # --- re-bias the exponent (hardfloat: sAdjustedExp) --------------------
    s_adjusted_exp = raw["sExp"] + (B_INT - (1 << in_exp_width))

    # --- align the significand (hardfloat: adjustedSig) --------------------
    # Result is SIG_INT+3 bits: [top][hidden][3 fraction][guard][sticky].
    if in_sig_width <= SIG_INT + 2:
        adjusted_sig = raw["sig"] << (SIG_INT - in_sig_width + 2)
    else:
        keep = raw["sig"] >> (in_sig_width - SIG_INT - 1)
        rest = raw["sig"] & ((1 << (in_sig_width - SIG_INT - 1)) - 1)
        adjusted_sig = (keep << 1) | (1 if rest else 0)

    do_shift_down1 = 0 if sig_msb_always_zero else (adjusted_sig >> (SIG_INT + 2)) & 1

    # --- the round mask ----------------------------------------------------
    # CHANGED: the format select, in two parts.
    #
    #   delta       shifts the exponent the decoder sees, so the mask starts
    #               widening at this format's own minNormExp.
    #   prec_shift  adds this format's missing bits of precision, as a shift
    #               with the low bits filled.  It has to be a shift rather than
    #               an offset on the exponent, because the decoder's output
    #               saturates at zero and an offset would vanish for every
    #               normal number.
    #
    # With delta = prec_shift = 0 these lines are hardfloat's, unchanged.
    mask_exp = (s_adjusted_exp + f["delta"]) & ((1 << EXP_SLICE) - 1)
    mask_body = low_mask(mask_exp, EXP_SLICE, MASK_TOP, MASK_BOTTOM) | do_shift_down1
    mask_body = (mask_body << f["prec_shift"]) | ((1 << f["prec_shift"]) - 1)
    mask_body &= (1 << (SIG_INT + 1)) - 1       # the thermometer saturates, so this is safe
    round_mask = (mask_body << 2) | 0b11

    shifted_round_mask = round_mask >> 1
    round_pos_mask = ~shifted_round_mask & round_mask
    round_pos_bit = (adjusted_sig & round_pos_mask) != 0
    any_round_extra = (adjusted_sig & shifted_round_mask) != 0
    any_round = round_pos_bit or any_round_extra

    round_incr = ((near_even or near_max) and round_pos_bit) or (round_mag_up and any_round)
    if round_incr:
        rounded_sig = ((adjusted_sig | round_mask) >> 2) + 1
        if near_even and round_pos_bit and not any_round_extra:
            rounded_sig &= ~(round_mask >> 1)          # ties-to-even: clear the LSB
    else:
        rounded_sig = (adjusted_sig & ~round_mask) >> 2
        if odd and any_round:
            rounded_sig |= round_pos_mask >> 1

    s_rounded_exp = s_adjusted_exp + (rounded_sig >> SIG_INT)

    # hardfloat's common_fractOut: 3 bits, binary8p4's fraction field.
    frac3 = ((rounded_sig >> 1) if do_shift_down1 else rounded_sig) & 0b111
    # binary8p3 keeps 2: its lowest bit was masked away above, so this is exact.
    assert fmt == "p4" or frac3 & 1 == 0, "binary8p3 kept a bit the mask should have cleared"
    frac = frac3 >> (3 - f["frac_bits"])

    # --- range checks ------------------------------------------------------
    # CHANGED: hardfloat asks whether the recoded exponent reached the Inf tag.
    # We ask whether the value passed the format's real largest finite number,
    # which is one ULP lower in the extended domain because 0x7F is an infinity.
    max_frac = (1 << f["frac_bits"]) - (1 if finite else 2)
    common_overflow = s_rounded_exp > f["emax"] or (
        s_rounded_exp == f["emax"] and frac > max_frac)
    common_total_underflow = s_rounded_exp < f["min_nonzero"]

    # hardfloat's underflow, with tininess detected after rounding.
    round_carry = (rounded_sig >> (SIG_INT + 1 if do_shift_down1 else SIG_INT)) & 1
    ur_round_pos_bit = (adjusted_sig >> (2 if do_shift_down1 else 1)) & 1
    ur_any_round = (do_shift_down1 and (adjusted_sig >> 2) & 1) or (adjusted_sig & 0b11) != 0
    ur_round_incr = ((near_even or near_max) and ur_round_pos_bit) or (round_mag_up and ur_any_round)
    common_underflow = common_total_underflow or (
        any_round and s_adjusted_exp <= f["min_norm"]
        and ((round_mask >> (3 if do_shift_down1 else 2)) & 1)
        and not (detect_tininess_after
                 and not ((round_mask >> (4 if do_shift_down1 else 3)) & 1)
                 and round_carry and round_pos_bit and ur_round_incr))
    common_inexact = common_total_underflow or any_round

    # --- specials and flags (hardfloat's tail) -----------------------------
    is_nan_out = invalid_exc or raw["isNaN"]
    is_inf_in = raw["isInf"]
    common_case = not is_nan_out and not is_inf_in and not raw["isZero"]
    overflow = common_case and common_overflow
    underflow = common_case and common_underflow
    inexact = overflow or (common_case and common_inexact)

    # CHANGED from hardfloat: round-to-odd goes out on overflow too, as P3109
    # 4.7.5 requires (see overflowGoesOut in P3109Rounder.scala).
    overflow_round_mag_up = near_even or near_max or round_mag_up or odd
    peg_min_nonzero = common_case and common_total_underflow and (round_mag_up or odd)
    peg_max_finite = overflow and not overflow_round_mag_up

    # --- encode an 8-bit P3109 code ----------------------------------------
    # This part is new: hardfloat emits a recoded number, which has no
    # subnormals and no P3109 code points.
    sign = raw["sign"]
    sb = sign << 7
    nan_code = 0x80
    max_finite_code = sb | (0x7F if finite else 0x7E)
    inf_code = nan_code if finite else (sb | 0x7F)

    if is_nan_out:
        code = nan_code
    elif is_inf_in:
        # A true infinity clamps only when saturating; otherwise it stays an
        # infinity, or becomes NaN in the finite domain, which has none.
        code = max_finite_code if sat else inf_code
    elif raw["isZero"]:
        code = 0x00                                   # P3109 has no -0
    elif overflow:
        # sat forces the clamp; otherwise inward rounding clamps and the
        # nearest/outward modes go to the overflow code point.
        code = max_finite_code if (sat or peg_max_finite) else inf_code
    elif common_total_underflow:
        code = (sb | 1) if peg_min_nonzero else 0x00  # -0 flushes to +0
    elif s_rounded_exp < f["min_norm"]:
        # Subnormal: the widened mask already rounded to the subnormal grid,
        # so this shift only moves the bits into place and loses nothing.
        shift = f["min_norm"] - s_rounded_exp
        sig_with_hidden = (1 << f["frac_bits"]) | frac
        assert sig_with_hidden & ((1 << shift) - 1) == 0, "subnormal shift dropped a bit"
        field = sig_with_hidden >> shift
        code = 0x00 if field == 0 else (sb | field)
    else:
        exp_field = s_rounded_exp - B_INT + f["bias"]
        code = sb | (exp_field << f["frac_bits"]) | frac

    return code, (invalid_exc, False, overflow, underflow, inexact)


def convert_bf16(bits, fmt, mode, sat=False, finite=False):
    """BF16 bit pattern -> P3109 code, the whole conversion path."""
    return unified_round(raw_from_fn(bits, 8, 8), fmt, mode, sat, finite)[0]
