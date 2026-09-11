package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._

// =============================================================================
// IEEE P3109 8-bit formats
// =============================================================================
//
// What this file is for
// ---------------------
// The conversion unit (FPConv.scala) can turn BF16 numbers into 8-bit numbers
// and back. Normally it uses the two OCP FP8 formats, E4M3 and E5M2. When the
// build option "p3109" is set (VectorParams.p3109 = Some(P3109Formats(...))),
// it uses the two IEEE P3109 formats instead:
//
//   binary8p4 : chosen with altfmt = 0, in the place of OCP E4M3
//   binary8p3 : chosen with altfmt = 1, in the place of OCP E5M2
//
// This file holds the P3109 pieces. The OCP build never uses anything here.
//
// Three ideas you will meet again and again below
// -----------------------------------------------
// 1. "Half as much".
//    The same 8 bits mean exactly half as much in P3109 as they do in the
//    usual IEEE-style format of the same shape. So to turn a number v into
//    P3109, we hand the existing hardfloat rounder the number 2*v: the bits it
//    gives back are already the right P3109 bits for v.
//
// 2. "The top row is missing".
//    The usual IEEE-style rounders keep their whole top row of exponents for
//    Infinity and NaN. P3109 does not: at most two codes of that row are
//    infinities (0x7F = +Inf and 0xFF = -Inf, in the "extended" domain), and in
//    the "finite" domain none are. The rest of the top row holds ordinary
//    numbers, which the usual rounder cannot produce. So next to the usual
//    ("narrow") rounder we run a second ("wide") rounder that has one more
//    exponent bit and can reach that top row, and we use its answer when the
//    result lands there. Below the top row we trust the narrow rounder: for
//    the very smallest numbers the wide one keeps digits that the 8-bit format
//    does not have. OCP E4M3 already works this way.
//
// 3. "Special codes".
//    P3109 has a single NaN (0x80) and no negative zero. In the extended
//    domain it also has infinities (0x7F and 0xFF). The usual rounders use
//    other codes for these, so the last step always rewrites them.
// =============================================================================


// -----------------------------------------------------------------------------
// Build option: which "domain" each P3109 format uses
// -----------------------------------------------------------------------------
// Every P3109 format comes in two versions, called domains:
//
//   Extended : has infinities. 0x7F is +Inf and 0xFF is -Inf.
//   Finite   : has no infinities. 0x7F and 0xFF are ordinary numbers instead
//              (the largest positive and negative ones), so the format reaches
//              one step further. Where an extended format would give infinity
//              (a number too big to fit, with saturation off), a finite format
//              gives NaN.
//
// The two formats can be set separately, for example
// P3109Formats(p4 = P3109Domain.Finite, p3 = P3109Domain.Extended).
sealed trait P3109Domain
object P3109Domain {
  case object Extended extends P3109Domain
  case object Finite extends P3109Domain
}
case class P3109Formats(
  p4: P3109Domain = P3109Domain.Extended,   // binary8p4, the altfmt = 0 format
  p3: P3109Domain = P3109Domain.Extended    // binary8p3, the altfmt = 1 format
)


// -----------------------------------------------------------------------------
// Doubles a number before it is rounded (idea 1).
// -----------------------------------------------------------------------------
// The number arrives in hardfloat's "raw" form: a sign, an exponent, the
// digits, and three separate yes/no flags that say "this is zero",
// "this is infinity" and "this is NaN". Doubling a number means adding 1 to
// its exponent, so that is all this does.
//
// Zero, infinity and NaN are recognised by their flags, not by their
// exponent, so adding 1 to the exponent leaves them alone.
//
// Why the raw form and not the "recoded" form used elsewhere in FPConv? In
// the recoded form, the exponent field is also what marks a number as
// infinity, and the biggest ordinary BF16 numbers sit one step below that
// mark: adding 1 there would turn them into infinity. The raw form does not
// have this problem. Its exponent field has room to spare: for BF16 the
// largest value after adding 1 is 384, and the field goes up to 511.
object p3109TimesTwo {
  def apply(in: hardfloat.RawFloat): hardfloat.RawFloat = {
    val out = WireInit(in)
    out.sExp := in.sExp + 1.S
    out
  }
}


// -----------------------------------------------------------------------------
// Builds the final binary8p4 code (altfmt = 0).
// -----------------------------------------------------------------------------
// Both rounders were given 2*v (idea 1). Their answers come in as plain
// IEEE-style bit patterns:
//
//   ieeeE5M3 : the wide rounder (5 exponent bits). It can reach binary8p4's
//              top row, but is too precise for the very smallest numbers.
//   ieeeE4M3 : the narrow rounder (4 exponent bits). It is right everywhere
//              except the top row, which it cannot produce.
//
// The other inputs are the saturation bit, the rounding mode, the wide
// rounder's "overflow" flag (explained where it is used), and whether this
// build uses the finite domain for binary8p4 (see P3109Domain above).
//
// This follows assembleOFPE4M3 (e5M3ToFp8.scala) step by step. Every decision
// about which number comes out is the same; only the codes used for NaN,
// infinity and negative zero are different.
object assembleP3109P4 {
  def apply(ieeeE5M3: UInt, ieeeE4M3: UInt, saturate: Bool, roundingMode: Bits, e5m3Overflow: Bool,
            finite: Boolean = false): UInt = {
    val sign = ieeeE4M3(7)
    val expWide = ieeeE5M3(7, 3)   // the wide rounder's 5 exponent bits
    val sigWide = ieeeE5M3(2, 0)   // ... and its 3 fraction bits

    // What did the wide rounder give back?
    val isNaN = expWide === "b11111".U && sigWide =/= 0.U   // NaN: the input was NaN
    val isInf = expWide === "b11111".U && sigWide === 0.U   // infinity
    // The result lands in binary8p4's top row
    // (in doubled terms: from 256 up to, but not including, 512).
    val inTopRow = expWide === "b10111".U
    // The result is bigger than anything in the top row (or is infinity).
    val aboveTopRow = expWide(4, 3) === "b11".U
    // In the extended domain the last code of the top row, 0x7F, is +Inf and
    // not a number, so a result that would land there is too big as well. In
    // the finite domain 0x7F is an ordinary number (the largest one).
    val tooBig = if (finite) aboveTopRow else aboveTopRow || (inTopRow && sigWide === "b111".U)

    // When the result is too big, do we give the largest number, or the
    // "too big" code: infinity (0x7F / 0xFF) in the extended domain, NaN in
    // the finite domain, which has no infinity?
    //
    //   - Saturation on: always the largest number.
    //   - Saturation off: it depends on the rounding direction.
    //       * Rounding to nearest, or rounding away from zero for this number
    //         (up for a positive number, down for a negative one): the "too
    //         big" code.
    //       * Rounding towards zero for this number (towards zero, or down for
    //         a positive number, up for a negative one): the largest number,
    //         because jumping to infinity would be rounding the wrong way.
    //   - Saturation off, and the input was already infinity: the "too big"
    //     code.
    val roundsAwayFromZero = (roundingMode === hardfloat.consts.round_min && sign) ||
      (roundingMode === hardfloat.consts.round_max && !sign)
    val tooBigBecomesInf = roundingMode === hardfloat.consts.round_near_even ||
      roundingMode === hardfloat.consts.round_near_maxMag || roundsAwayFromZero

    // The wide rounder says "infinity" in two different cases: the input really
    // was infinity, or the input was an ordinary number that was too big. The
    // rules above treat those differently, and the rounder's answer alone cannot
    // tell them apart. Its overflow flag can: hardfloat sets it only when an
    // ordinary number was too big, never for an input that was already infinity.
    val inputWasInf = isInf && !e5m3Overflow
    val giveLargest = saturate || (!inputWasInf && !tooBigBecomesInf)

    val nan      = "h80".U(8.W)                 // P3109's only NaN
    val infinity = sign ## "b1111111".U(7.W)   // 0x7F or 0xFF
    // The largest number is 0x7E (or 0xFE) in the extended domain, and 0x7F
    // (or 0xFF) in the finite domain, where those codes are free.
    val largest    = if (finite) sign ## "b1111111".U(7.W) else sign ## "b1111110".U(7.W)
    val tooBigCode = if (finite) nan else infinity

    // P3109 has no negative zero, and the code the narrow rounder uses for it
    // (0x80) is P3109's NaN. So a negative zero from the narrow rounder becomes
    // plain zero. Only the narrow rounder's answer can be a negative zero; the
    // 0x80 codes made elsewhere below are real NaNs and must stay NaN.
    val narrowAnswer = Mux(ieeeE4M3 === "h80".U, 0.U(8.W), ieeeE4M3)

    Mux(isNaN, nan,
    Mux(tooBig, Mux(giveLargest, largest, tooBigCode),
    Mux(inTopRow, sign ## "b1111".U(4.W) ## sigWide,   // top row: take the wide rounder's answer
                  narrowAnswer)))                      // everywhere else: the narrow rounder's answer
  }
}


// -----------------------------------------------------------------------------
// Builds the final binary8p3 code (altfmt = 1).
// -----------------------------------------------------------------------------
// Exactly the same steps as assembleP3109P4 just above, for the other format:
//
//   ieeeE6M2 : the wide rounder (6 exponent bits). It can reach binary8p3's
//              top row.
//   ieeeE5M2 : the narrow rounder (5 exponent bits). It is right everywhere
//              except the top row.
//
// "finite" says whether this build uses the finite domain for binary8p3.
//
// Under OCP this slot (E5M2) needed no wide rounder at all, because OCP E5M2
// keeps its whole top row for Infinity and NaN, just like IEEE. P3109 uses that
// top row for numbers, so binary8p3 needs the same trick as the other format.
object assembleP3109P3 {
  def apply(ieeeE6M2: UInt, ieeeE5M2: UInt, saturate: Bool, roundingMode: Bits, e6m2Overflow: Bool,
            finite: Boolean = false): UInt = {
    val sign = ieeeE5M2(7)
    val expWide = ieeeE6M2(7, 2)   // the wide rounder's 6 exponent bits
    val sigWide = ieeeE6M2(1, 0)   // ... and its 2 fraction bits

    val isNaN = expWide === "b111111".U && sigWide =/= 0.U
    val isInf = expWide === "b111111".U && sigWide === 0.U
    // binary8p3's top row (in doubled terms: from 65536 up to, but not including, 131072).
    val inTopRow = expWide === "b101111".U
    val aboveTopRow = expWide(5, 4) === "b11".U
    // Extended domain: 0x7F is +Inf, so the last code of the top row is too big
    // as well. Finite domain: 0x7F is the largest number.
    val tooBig = if (finite) aboveTopRow else aboveTopRow || (inTopRow && sigWide === "b11".U)

    // Largest number or the "too big" code? Same rules as in assembleP3109P4.
    val roundsAwayFromZero = (roundingMode === hardfloat.consts.round_min && sign) ||
      (roundingMode === hardfloat.consts.round_max && !sign)
    val tooBigBecomesInf = roundingMode === hardfloat.consts.round_near_even ||
      roundingMode === hardfloat.consts.round_near_maxMag || roundsAwayFromZero
    val inputWasInf = isInf && !e6m2Overflow
    val giveLargest = saturate || (!inputWasInf && !tooBigBecomesInf)

    val nan        = "h80".U(8.W)
    val infinity   = sign ## "b1111111".U(7.W)   // 0x7F or 0xFF
    val largest    = if (finite) sign ## "b1111111".U(7.W) else sign ## "b1111110".U(7.W)
    val tooBigCode = if (finite) nan else infinity

    // No negative zero in P3109 (see assembleP3109P4).
    val narrowAnswer = Mux(ieeeE5M2 === "h80".U, 0.U(8.W), ieeeE5M2)

    Mux(isNaN, nan,
    Mux(tooBig, Mux(giveLargest, largest, tooBigCode),
    Mux(inTopRow, sign ## "b11111".U(5.W) ## sigWide,  // top row: take the wide rounder's answer
                  narrowAnswer)))                      // everywhere else: the narrow rounder's answer
  }
}


// -----------------------------------------------------------------------------
// Reads a P3109 8-bit code and rewrites it as an E5M3 number (widening).
// -----------------------------------------------------------------------------
// On the way from 8 bits to BF16, the conversion unit first rewrites the 8-bit
// number as E5M3 (1 sign bit, 5 exponent bits, 3 fraction bits). Every
// binary8p4 and binary8p3 number fits exactly in E5M3, so nothing is rounded
// here: bits are only moved around.
//
// There is no "times two" on this side: we write the number's true value.
// (Doubled, the biggest binary8p3 number would not fit in E5M3.)
//
// This takes the place of fp8ToE5M3 (fp8ToE5M3.scala), which reads the OCP
// formats. p4Finite / p3Finite say which formats use the finite domain.
object p3109ToE5M3 {
  def apply(in: Bits, altfmt: Bool, p4Finite: Boolean = false, p3Finite: Boolean = false): UInt = {
    val sign = in(7)

    // The three special codes, the same in both P3109 formats.
    val isZero = in === "h00".U        // zero (P3109 has only one zero)
    val isNaN  = in === "h80".U        // NaN
    // 0x7F / 0xFF are +Inf / -Inf only in the extended domain. In the finite
    // domain they are ordinary numbers, and the paths below already read them
    // correctly (as the largest positive / negative number).
    val infCodesAreInf = Mux(altfmt, (!p3Finite).B, (!p4Finite).B)
    val isInf  = in(6, 0) === "h7F".U && infCodesAreInf

    // --- binary8p4: 1 sign bit, 4 exponent bits, 3 fraction bits ------------
    val exp4 = in(6, 3)
    val sig4 = in(2, 0)
    // The exponent is stored with an offset of 8 in binary8p4, and of 15 in
    // E5M3. So for an ordinary number the exponent goes up by 15 - 8 = 7.
    val normal4 = sign ## ((0.U(1.W) ## exp4) + 7.U) ## sig4
    // Very small numbers (exponent field 0, called "subnormals") have no hidden
    // leading 1. E5M3 can hold them as ordinary numbers: shift the fraction left
    // until its first 1 becomes the hidden leading 1, and lower the exponent by
    // the same amount. (Same method as the OCP reader, with 7 in place of 8.)
    val shift4 = PriorityEncoder(Reverse(sig4))   // how far to shift
    val tinySig4 = ((sig4 << 1.U) << shift4)(2, 0)
    val tinyExp4 = 7.U(5.W) - shift4
    val tiny4 = sign ## tinyExp4 ## tinySig4
    val p4 = Mux(exp4 === 0.U, tiny4, normal4)

    // --- binary8p3: 1 sign bit, 5 exponent bits, 2 fraction bits ------------
    val exp3 = in(6, 2)
    val sig3 = in(1, 0)
    // The offset is 16 in binary8p3 and 15 in E5M3, so an ordinary number's
    // exponent goes down by 1. E5M3 has one more fraction bit, which is 0.
    val normal3 = sign ## (exp3 - 1.U) ## sig3 ## 0.U(1.W)
    // Careful: binary8p3's smallest ordinary numbers (exponent field 1) are too
    // small to be ordinary numbers in E5M3: "exponent - 1" would give 0 and turn
    // them into zero. So they, and binary8p3's subnormals (exponent field 0),
    // are written as E5M3 subnormals. The two line up exactly: the E5M3 fraction
    // is simply the exponent bit followed by the two fraction bits.
    val tiny3 = sign ## 0.U(5.W) ## exp3(0) ## sig3
    val p3 = Mux(exp3(4, 1) === 0.U, tiny3, normal3)   // exponent field 0 or 1

    val number   = Mux(altfmt, p3, p4)
    val nan      = 0.U(1.W) ## "b11111".U(5.W) ## "b100".U(3.W)   // E5M3 NaN
    val infinity = sign ## "b11111".U(5.W) ## 0.U(3.W)            // E5M3 +/-Inf
    Mux(isNaN, nan, Mux(isInf, infinity, Mux(isZero, 0.U(9.W), number)))
  }
}
