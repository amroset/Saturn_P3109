package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import saturn.common._

// Precision sweep: P3109 builds that carry more formats than binary8p4/p3, so
// synthesis can measure what each further precision costs.
//
// The formats are selected by a format code, fmt = p3109_fmt_hi ## altfmt:
// 0 is binary8p4, 1 binary8p3, 2, 3, ... the build's extra precisions. No
// instruction sets p3109_fmt_hi yet (ExecuteSequencer ties it to zero), so
// software reaches only the pair; the units are synthesized on their own, where
// the code is a free input.
//
// With more formats:
//  - P3109Rounder's datapath is as wide as the widest format;
//  - the 8-bit FMA core, and every other core an 8-bit lane runs on, must hold
//    every format exactly (coreFor);
//  - operands are read by p3109ToCore instead of p3109ToE5M3.
// Widening to BF16 needs nothing new: BF16 holds every binary8pP exactly.

object P3109Sweep {
  // Smallest IEEE-style type that holds every listed binary8pP exactly,
  // subnormals included. E5M3 for the pair.
  def coreFor(precisions: Seq[Int]): FType = {
    def maxExp(p: Int) = (1 << (7 - p)) - 1           // largest unbiased exponent
    def minSub(p: Int) = 1 - (1 << (7 - p)) - (p - 1)  // smallest subnormal's exponent
    val sig = precisions.max
    val exp = (2 to 11).find { e =>
      val bias = (1 << (e - 1)) - 1
      bias >= precisions.map(maxExp).max && 1 - bias - (sig - 1) <= precisions.map(minSub).min
    }.get
    FType(exp, sig)
  }

  // The type 8-bit operands are read into, in FPConv and the FMA. E5M3 holds
  // Binary4p2sf too.
  def core8(p3109: Option[P3109Formats]): FType =
    p3109.filter(_.genericReader).map(f => coreFor(f.formats.filter(_.k == 8).map(_.p))).getOrElse(MXFType.E5M3)

  // Choose a per-format value by format code (fmtWidth bits), xs in the order
  // of P3109Formats.byCode. Two formats are one mux on bit 0, as in the pair's
  // datapath; unused codes read as binary8p4.
  def pick[T <: Data](fmt: UInt, formats: P3109Formats, xs: Seq[T]): T = {
    val codes = formats.byCode.map(_._1)
    require(codes.size == xs.size)
    if (codes == Seq(0, 1)) Mux(fmt(0), xs(1), xs(0))
    else if (codes == Seq(0, 1, P3109Formats.fp4Code)) Mux(P3109Formats.isFP4(fmt, formats), xs(2), Mux(fmt(0), xs(1), xs(0)))
    else VecInit(Seq.tabulate(1 << fmt.getWidth)(c => xs(codes.indexOf(c) max 0)))(fmt)
  }
}

// Any of the build's P3109 formats to its core8 type, which holds them exactly.
// A 4-bit format is read from the low nibble.
// The generic version of p3109ToE5M3: each format is decoded to an unbiased
// exponent and a significand with its leading 1, and one encoder writes that as
// a core normal or subnormal.
object p3109ToCore {

  def apply(in: Bits, fmt: UInt, formats: P3109Formats): UInt = {
    val core8 = P3109Sweep.core8(Some(formats))
    val expW = core8.exp
    val sigW = core8.sig
    val coreBias = (1 << (expW - 1)) - 1

    // Per format, from the low K bits: (exponent, sigW-bit significand, sign,
    // zero, NaN, Inf). Code 1 0..0 is NaN; 0 1..1 is Inf in the extended domain.
    val decoded = formats.formats.map { case P3109Fmt(k, p, finite) =>
      val f = p - 1
      val bias = 1 << (k - 1 - p)
      val expField = in(k - 2, f)
      val fracField = in(f - 1, 0)
      val normExp = expField.zext -& bias.S // -& so the subtraction cannot wrap
      // Subnormal 0.frac x 2^(1 - bias): shift the first 1 up to the hidden bit,
      // which lowers the exponent by lz + 1
      val lz = PriorityEncoder(Reverse(fracField))
      val subFrac = ((fracField << lz) << 1)(f - 1, 0)
      val subExp = (-bias).S -& lz.zext
      val isSub = expField === 0.U
      val exp = Mux(isSub, subExp, normExp)
      val frac = Mux(isSub, subFrac, fracField)
      (exp, (1.U(1.W) ## frac ## 0.U((sigW - p).W))(sigW - 1, 0), in(k - 1),
       in(k - 1, 0) === 0.U, in(k - 1, 0) === (1 << (k - 1)).U,
       (!finite).B && in(k - 2, 0).andR)
    }
    val decodedExpW = decoded.map(_._1.getWidth).max
    def pick[T <: Data](xs: Seq[T]) = P3109Sweep.pick(fmt, formats, xs)
    val exp = pick(decoded.map(_._1.pad(decodedExpW)))
    val sig = pick(decoded.map(_._2))
    val sign = pick(decoded.map(_._3))
    val isZero = pick(decoded.map(_._4))
    val isNaN = pick(decoded.map(_._5))
    val isInf = pick(decoded.map(_._6))

    // Below the core's smallest normal, shift into a core subnormal
    val isCoreNormal = exp >= (1 - coreBias).S
    val biasedExp = (exp + coreBias.S).asUInt
    val normExpField = biasedExp(expW - 1, 0)
    val subShift = ((1 - coreBias).S - exp).asUInt
    val subFracField = (sig >> subShift)(sigW - 2, 0)
    val number = Mux(isCoreNormal, sign ## normExpField ## sig(sigW - 2, 0), sign ## 0.U(expW.W) ## subFracField)

    Mux(isNaN, 0.U(1.W) ## Fill(expW, 1.U(1.W)) ## 1.U(1.W) ## 0.U((sigW - 2).W),
      Mux(isInf, sign ## Fill(expW, 1.U(1.W)) ## 0.U((sigW - 1).W),
        Mux(isZero, 0.U((expW + sigW).W), number)))
  }
}
