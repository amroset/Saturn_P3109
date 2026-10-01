package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._

// =============================================================================
// One rounder for both P3109 formats
// =============================================================================
//
// What this module is for
// -----------------------
// It takes a number that is too precise for an 8-bit P3109 format and produces
// the 8-bit code closest to it. Which of the two formats is decided by a wire
// (altfmt), not by a build option, so one copy of this module does the job that
// currently takes four rounders plus two assemblers plus a doubling.
//
// It is a prototype: nothing instantiates it yet. The existing path in
// p3109Fp8.scala / FPConv.scala is untouched, so both can run side by side and
// be compared before either is removed.
//
//
// How it relates to hardfloat
// ---------------------------
// The rounding itself is hardfloat's RoundAnyRawFNToRecFN, transcribed. Only
// three things are different, and all are marked CHANGED below:
//
//   1. the round mask takes two extra inputs, which is how the format is
//      selected at run time (see "the two knobs" below);
//   2. the overflow test compares against the format's real largest finite
//      number instead of asking whether the exponent reached IEEE's infinity
//      encoding, which P3109 does not have;
//   3. round-to-odd overflows to Inf (or NaN in the finite domain) rather than
//      stopping at the largest finite, as P3109 4.7.5 requires.
//
// The tail of the module -- turning the rounded number into an 8-bit code -- is
// new. hardfloat ends by producing a "recoded" number, which has no subnormals
// and no P3109 code points, so it could not be reused.
//
//
// The two knobs
// -------------
// hardfloat's round mask marks the bits that are being thrown away. It already
// widens by one bit for every step the exponent falls below the smallest normal
// number: that is how subnormal results are rounded without any shifter. Giving
// up a bit of precision is the same operation, so a second format does not need
// a second rounder -- it needs a second reason for the mask to widen.
//
//   delta       moves the exponent at which the mask starts widening, so that it
//               lands on this format's own smallest normal number.
//
//   precShift   adds the bits of precision this format does not have. This one
//               cannot be folded into delta: the mask decoder's output stops at
//               zero, so a constant added to the exponent quietly disappears for
//               every normal number. It is applied as a shift with the low bits
//               filled in, which survives that clamp.
//
// The same idea appears in US 6,490,607 (Oberman, AMD, filed 1999, expired
// 2019), where the precision of the operation selects which rounding constant
// is added and where it sits. That design can use a constant at a fixed place
// because its subnormal results are handled in microcode; ours must round them
// in hardware, so the place has to move with the exponent.
//
//
// Checked against
// ---------------
// models/unified_rounder.py, which is this same algorithm written in Python and
// compared with gfloat over roughly 8.4 million cases: every BF16 input against
// every rounding mode, both formats, both domains and both saturation settings;
// and every 8-bit operand pair through multiply, add and subtract on all five
// FMA core sizes.
// =============================================================================


/** Everything the rounder needs to know about one binary8pP format.
  *
  * All of it is worked out from the precision, so adding a third format later
  * means adding one line, not new hardware. The numbers are in the rounder's
  * internal exponent, which is biased by P3109Rounder.IntBias.
  */
case class P3109FormatInfo(precision: Int, finite: Boolean) {
  import P3109Rounder.{IntBias, IntSig}

  val fracBits = precision - 1                      // bits of fraction stored
  val expBits  = 8 - precision                      // bits of exponent stored
  val bias     = 1 << (7 - precision)               // P3109's own bias, 2^(K-P-1)

  /** Smallest normal number's exponent (P3109's emin = 1 - bias). */
  val minNorm     = IntBias + 1 - bias
  /** Smallest subnormal's exponent: below this everything rounds to zero. */
  val minNonzero  = minNorm - fracBits
  /** Largest finite exponent. */
  val emax        = IntBias + ((1 << expBits) - 1 - bias)

  /** Largest fraction that is still a finite number at emax. In the extended
    * domain 0x7F is an infinity, so the largest finite value is one step lower.
    */
  val maxFrac     = (1 << fracBits) - (if (finite) 1 else 2)

  /** Knob 1: how far this format's smallest normal sits from the one the mask
    * decoder was built around.
    */
  val delta       = P3109Rounder.MaskBottom - minNorm
  /** Knob 2: bits of precision this format gives up against the datapath. */
  val precShift   = IntSig - precision

  require(precision >= 2 && precision <= IntSig,
    s"precision $precision does not fit a datapath built for $IntSig bits")
}


object P3109Rounder {
  /** Internal significand width, in hardfloat's "outSigWidth" terms. Sized for
    * the widest format the rounder serves, which is binary8p4.
    */
  val IntSig = 4

  /** Internal exponent bias. Any constant works as long as every format's
    * exponents stay positive; 2^8 is BF16's own, so conversions re-bias by zero.
    */
  val IntBias = 1 << 8

  // The mask decoder is built once, around binary8p4's smallest normal. Every
  // other format reaches it through `delta` rather than through its own decoder.
  val MaskBottom = IntBias + 1 - (1 << 3)     // binary8p4: emin = 1 - 8 = -7
  val MaskTop    = MaskBottom - IntSig - 1    // the same bounds hardfloat uses

  /** Width of the value handed to the decoder. The exponent is clamped into
    * [MaskTop, MaskBottom] first, so this only has to hold that range.
    */
  val MaskExpWidth = log2Ceil(MaskBottom + 1)
}


/** Round a RawFloat straight to an 8-bit IEEE P3109 code.
  *
  * @param inExpWidth  exponent width of the incoming RawFloat
  * @param inSigWidth  significand width of the incoming RawFloat
  * @param formats     which domain each of the two formats uses (a build option)
  * @param sigMSBitAlwaysZero  true when the caller guarantees the significand is
  *                    below 2, as it is for a value read from a register. The
  *                    FMA's unrounded result can reach 4, so it must pass false.
  */
class P3109Rounder(
  inExpWidth: Int,
  inSigWidth: Int,
  formats: P3109Formats,
  sigMSBitAlwaysZero: Boolean = false
) extends RawModule {
  import P3109Rounder._

  override def desiredName = s"P3109Rounder_ie${inExpWidth}_is${inSigWidth}"

  val io = IO(new Bundle {
    val in             = Input(new hardfloat.RawFloat(inExpWidth, inSigWidth))
    val altfmt         = Input(Bool())      // 0 = binary8p4, 1 = binary8p3
    val roundingMode   = Input(UInt(3.W))
    val sat            = Input(Bool())      // the .sat instruction variant
    val invalidExc     = Input(Bool())
    val detectTininess = Input(UInt(1.W))
    val out            = Output(UInt(8.W))
    val exceptionFlags = Output(UInt(5.W))
  })

  // ---------------------------------------------------------------------------
  // The format. Everything here is a constant picked at build time; altfmt just
  // chooses between two of them.
  // ---------------------------------------------------------------------------
  val p4 = P3109FormatInfo(4, formats.p4 == P3109Domain.Finite)
  val p3 = P3109FormatInfo(3, formats.p3 == P3109Domain.Finite)

  val fracBits   = Mux(io.altfmt, p3.fracBits.U,   p4.fracBits.U)
  val precShift  = Mux(io.altfmt, p3.precShift.U,  p4.precShift.U)
  val delta      = Mux(io.altfmt, p3.delta.S,      p4.delta.S)
  val minNorm    = Mux(io.altfmt, p3.minNorm.S,    p4.minNorm.S)
  val minNonzero = Mux(io.altfmt, p3.minNonzero.S, p4.minNonzero.S)
  val emax       = Mux(io.altfmt, p3.emax.S,       p4.emax.S)
  val maxFrac    = Mux(io.altfmt, p3.maxFrac.U,    p4.maxFrac.U)
  val bias       = Mux(io.altfmt, p3.bias.S,       p4.bias.S)

  // ---------------------------------------------------------------------------
  // Rounding mode
  // ---------------------------------------------------------------------------
  val nearEven = io.roundingMode === hardfloat.consts.round_near_even
  val nearMax  = io.roundingMode === hardfloat.consts.round_near_maxMag
  val toOdd    = io.roundingMode === hardfloat.consts.round_odd
  // "round away from zero": the two directed modes, once the sign is known.
  val roundMagUp =
    (io.roundingMode === hardfloat.consts.round_min && io.in.sign) ||
    (io.roundingMode === hardfloat.consts.round_max && !io.in.sign)

  // ---------------------------------------------------------------------------
  // Line the number up. Both steps are hardfloat's, unchanged.
  // ---------------------------------------------------------------------------

  // Put the incoming exponent onto the internal bias.
  val sAdjustedExp = io.in.sExp +& (IntBias - (1 << inExpWidth)).S

  // Put the significand into a fixed slot, the same one for either format:
  //     [top] [hidden] [3 fraction] [guard] [sticky]
  // Anything that does not fit is OR-ed into the sticky bit, which is all the
  // rounding ever needs to know about it.
  val adjustedSig = if (inSigWidth <= IntSig + 2) {
    io.in.sig << (IntSig - inSigWidth + 2)
  } else {
    io.in.sig(inSigWidth, inSigWidth - IntSig - 1) ##
      io.in.sig(inSigWidth - IntSig - 2, 0).orR
  }

  // True when the significand reached 2, so everything sits one place higher.
  val doShiftSigDown1 =
    if (sigMSBitAlwaysZero) false.B else adjustedSig(IntSig + 2)

  // ---------------------------------------------------------------------------
  // The round mask: which bits are being thrown away.
  //
  // CHANGED -- this is the format select, and the only place it appears.
  // ---------------------------------------------------------------------------

  // The decoder only distinguishes exponents between MaskTop and MaskBottom:
  // above that nothing is discarded, below it everything is. Clamping first
  // keeps the decoder small and stops a very small exponent from wrapping round.
  val maskExp = sAdjustedExp + delta
  val maskExpClamped = Mux(maskExp < MaskTop.S, MaskTop.U,
                       Mux(maskExp > MaskBottom.S, MaskBottom.U,
                           maskExp.asUInt))

  // One bit per step the exponent has fallen below this format's smallest
  // normal -- hardfloat's gradual underflow, reached through `delta`.
  val subnormalBits = hardfloat.lowMask(
    maskExpClamped(MaskExpWidth - 1, 0), MaskTop, MaskBottom)

  // Add the bit the significand borrowed by reaching 2, then the bits this
  // format never had. `precShift` has to be a shift, not an OR: the two mask
  // runs both start at the bottom, so ORing them gives the longer one rather
  // than the sum of their lengths.
  val maskBody = (((subnormalBits | doShiftSigDown1) << precShift) |
                  ((1.U << precShift) - 1.U))(IntSig, 0)

  // The bottom two places are always discarded: they are the guard and sticky.
  val roundMask = maskBody ## 3.U(2.W)

  // ---------------------------------------------------------------------------
  // Guard, sticky, and the decision. All hardfloat, unchanged.
  // ---------------------------------------------------------------------------
  val shiftedRoundMask = 0.U(1.W) ## (roundMask >> 1)
  val roundPosMask     = ~shiftedRoundMask & roundMask   // just the guard's place
  val roundPosBit      = (adjustedSig & roundPosMask).orR
  val anyRoundExtra    = (adjustedSig & shiftedRoundMask).orR
  val anyRound         = roundPosBit || anyRoundExtra

  val roundIncr =
    ((nearEven || nearMax) && roundPosBit) || (roundMagUp && anyRound)

  val roundedSig = Mux(roundIncr,
    // Fill the discarded bits with ones and add one, so the carry lands in the
    // bits we keep. Then, for a tie in round-to-nearest-even, clear the last
    // bit again.
    (((adjustedSig | roundMask) >> 2) +& 1.U) &
      ~Mux(nearEven && roundPosBit && !anyRoundExtra, roundMask >> 1, 0.U((IntSig + 2).W)),
    // Otherwise just clear them.
    (adjustedSig & ~roundMask) >> 2 |
      Mux(toOdd && anyRound, roundPosMask >> 1, 0.U))

  // Rounding up can overflow the significand, which moves the exponent.
  val sRoundedExp = sAdjustedExp +& (roundedSig >> IntSig).asUInt.zext

  // The fraction, in binary8p4's three-bit field. binary8p3 keeps the top two:
  // its lowest bit was masked away above, so dropping it loses nothing.
  val frac3 = Mux(doShiftSigDown1, roundedSig(IntSig - 1, 1), roundedSig(IntSig - 2, 0))
  val frac  = frac3 >> precShift

  // ---------------------------------------------------------------------------
  // Did it still fit?
  //
  // CHANGED -- hardfloat asks whether the exponent reached the encoding that
  // means infinity. P3109 keeps finite numbers there, so we ask the question
  // that actually matters: is this past the largest number the format holds?
  // ---------------------------------------------------------------------------
  val commonOverflow = (sRoundedExp > emax) ||
                       ((sRoundedExp === emax) && (frac > maxFrac))
  val commonTotalUnderflow = sRoundedExp < minNonzero

  // hardfloat's underflow, with tininess judged after rounding. Its bit
  // positions are relative to the format's last kept bit, which sits
  // precShift places higher for a narrower format.
  val lsb = doShiftSigDown1.asUInt +& precShift
  val roundCarry = Mux(doShiftSigDown1, roundedSig(IntSig + 1), roundedSig(IntSig))
  val unboundedRoundPosBit = (adjustedSig >> (lsb +& 1.U))(0)
  val unboundedAnyRound = (adjustedSig & ((1.U << (lsb +& 2.U)) - 1.U)).orR
  val unboundedRoundIncr =
    ((nearEven || nearMax) && unboundedRoundPosBit) || (roundMagUp && unboundedAnyRound)
  val commonUnderflow = commonTotalUnderflow ||
    (anyRound && (sAdjustedExp <= minNorm) &&
      (roundMask >> (lsb +& 2.U))(0) &&
      !((io.detectTininess === hardfloat.consts.tininess_afterRounding) &&
        !(roundMask >> (lsb +& 3.U))(0) &&
        roundCarry && roundPosBit && unboundedRoundIncr))

  val commonInexact = commonTotalUnderflow || anyRound

  // ---------------------------------------------------------------------------
  // Exception flags
  // ---------------------------------------------------------------------------
  val isNaNOut  = io.invalidExc || io.in.isNaN
  val commonCase = !isNaNOut && !io.in.isInf && !io.in.isZero
  val overflow  = commonCase && commonOverflow
  val underflow = commonCase && commonUnderflow
  val inexact   = overflow || (commonCase && commonInexact)

  // Which way an out-of-range value goes: the nearest and outward modes reach
  // for the overflow code point, the inward ones stop at the largest finite.
  //
  // CHANGED -- round-to-odd goes out too. hardfloat stops it at the largest
  // finite, which is harmless in IEEE formats because their largest finite has
  // an all-ones, odd, significand. P3109 4.7.5 lets only TowardZero,
  // TowardNegative and TowardPositive stop there under SatNone; every other
  // mode gives Inf (extended domain) or NaN (finite domain). In the extended
  // domain it also matters for round-to-odd's own guarantee: the largest finite
  // there is 0x7E, an even code, so stopping at it would return an inexact even
  // result.
  val overflowGoesOut = nearEven || nearMax || roundMagUp || toOdd
  val pegMinNonzero   = commonCase && commonTotalUnderflow && (roundMagUp || toOdd)

  // ---------------------------------------------------------------------------
  // Write the 8-bit code.
  //
  // New: hardfloat stops one step earlier, at a recoded number, which has no
  // subnormals and none of P3109's code points.
  // ---------------------------------------------------------------------------
  val signBit = io.in.sign ## 0.U(7.W)

  val nanCode        = "h80".U(8.W)                                   // the only NaN
  val maxFiniteCodeP = Mux(io.altfmt, (if (p3.finite) "h7F" else "h7E").U,
                                      (if (p4.finite) "h7F" else "h7E").U)
  val maxFiniteCode  = signBit | maxFiniteCodeP
  // The finite domain has no infinity, so anything that would be one is a NaN.
  val eitherFinite   = Mux(io.altfmt, p3.finite.B, p4.finite.B)
  val infCode        = Mux(eitherFinite, nanCode, signBit | "h7F".U)

  // A subnormal result: the mask already rounded it onto the subnormal grid, so
  // this shift only moves the bits into place and cannot lose one.
  val subnormalShift = (minNorm - sRoundedExp).asUInt
  val sigWithHidden  = ((1.U(7.W) << fracBits) | frac)(6, 0)
  val subnormalField = (sigWithHidden >> subnormalShift)(6, 0)

  val normalExpField = (sRoundedExp - IntBias.S + bias).asUInt
  val normalCode     = signBit | ((normalExpField << fracBits) | frac)(6, 0)

  io.out := Mux(isNaNOut, nanCode,
            // A true infinity is clamped only when saturating.
            Mux(io.in.isInf, Mux(io.sat, maxFiniteCode, infCode),
            Mux(io.in.isZero, 0.U,                      // P3109 has no -0
            Mux(overflow, Mux(io.sat || !overflowGoesOut, maxFiniteCode, infCode),
            Mux(commonTotalUnderflow, Mux(pegMinNonzero, signBit | 1.U, 0.U),
            Mux(sRoundedExp < minNorm,
                Mux(subnormalField === 0.U, 0.U, signBit | subnormalField),
                normalCode))))))

  io.exceptionFlags := io.invalidExc ## false.B ## overflow ## underflow ## inexact

  // Two invariants hold here, and both are worth knowing when reading this:
  //
  //   * for binary8p3, frac3's low bit is always zero -- the round mask cleared
  //     it, which is what lets the format drop it without losing anything;
  //   * the subnormal shift above never drops a set bit, for the same reason.
  //
  // They are not Chisel asserts because this is a RawModule and has no clock.
  // models/unified_rounder.py checks both on every one of its 8.4 million
  // cases, and the chiseltest that drives this module should check them too.
}
