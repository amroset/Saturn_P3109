package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._

// IEEE P3109 8-bit formats, in place of OCP FP8 when VectorParams.p3109 is set:
// binary8p4 (altfmt = 0, in place of E4M3) and binary8p3 (altfmt = 1, E5M2).

// Extended: 0x7F/0xFF are +-Inf. Finite: they are the largest finite values,
// and an overflow that would give Inf gives NaN.
sealed trait P3109Domain
object P3109Domain {
  case object Extended extends P3109Domain
  case object Finite extends P3109Domain
}

case class P3109Formats(
  p4: P3109Domain = P3109Domain.Extended,
  p3: P3109Domain = P3109Domain.Extended,
  block: Boolean = false   // block scale factors in FPConv (p3109Block.scala)
) {
  def p4Finite = p4 == P3109Domain.Finite
  def p3Finite = p3 == P3109Domain.Finite
}

// P3109 code -> E5M3, which holds every binary8p4 and binary8p3 value exactly.
object p3109ToE5M3 {
  def apply(in: Bits, altfmt: Bool, p4Finite: Boolean, p3Finite: Boolean): UInt = {
    val sign = in(7)
    val isZero = in === "h00".U
    val isNaN = in === "h80".U
    val isInf = in(6, 0) === "h7F".U && Mux(altfmt, (!p3Finite).B, (!p4Finite).B)

    // binary8p4: bias 8 -> 15
    val exp4 = in(6, 3)
    val sig4 = in(2, 0)
    val shift4 = PriorityEncoder(Reverse(sig4))
    val p4 = Mux(exp4 === 0.U,
      sign ## (7.U(5.W) - shift4) ## ((sig4 << 1.U) << shift4)(2, 0), // Subnormal
      sign ## ((0.U(1.W) ## exp4) + 7.U) ## sig4)                       // Normal

    // binary8p3: bias 16 -> 15. Exponent fields 0 and 1 become E5M3 subnormals.
    val exp3 = in(6, 2)
    val sig3 = in(1, 0)
    val p3 = Mux(exp3(4, 1) === 0.U,
      sign ## 0.U(5.W) ## exp3(0) ## sig3,
      sign ## (exp3 - 1.U) ## sig3 ## 0.U(1.W))

    Mux(isNaN, "b0_11111_100".U(9.W),
      Mux(isInf, sign ## "b11111_000".U(8.W),
        Mux(isZero, 0.U(9.W), Mux(altfmt, p3, p4))))
  }
}

// The FMA's unrounded result -> P3109 code and flags (cf. rawUnroundedToFp8).
// The significand can reach [2,4), and there is no saturating form.
object rawUnroundedToP3109 {
  def apply(unroundedType: FType, unroundedIn: hardfloat.RawFloat, invalidExc: Bool,
            altfmt: Bool, roundingMode: Bits, formats: P3109Formats): (UInt, UInt) = {
    val r = Module(new P3109Rounder(unroundedType.exp, unroundedType.sig + 2, formats,
                                    sigMSBitAlwaysZero = false))
    r.io.in := unroundedIn
    r.io.altfmt := altfmt
    r.io.roundingMode := roundingMode
    r.io.sat := false.B
    r.io.invalidExc := invalidExc
    r.io.detectTininess := hardfloat.consts.tininess_afterRounding
    (r.io.out, r.io.exceptionFlags)
  }
}
