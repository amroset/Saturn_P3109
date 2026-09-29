package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._

// =============================================================================
// The precision sweep: more than two P3109 formats in one build
// =============================================================================
//
// What this file is for
// ---------------------
// The conformance pair (binary8p4, binary8p3) is selected by altfmt, one bit.
// To measure what each further precision costs, a build can list more formats
// (P3109Formats.extra). They are selected by a wider format code, `fmt`:
//
//     fmt = [ p3109_fmt_hi ][ altfmt ]      code 0 = binary8p4, 1 = binary8p3,
//                                           2, 3, ... = the extra formats
//
// No instruction sets the upper bits yet -- ExecuteSequencer ties them to zero
// -- so software still reaches only the pair. The units are synthesized on
// their own, where the upper bits are free inputs, so the hardware for every
// format is kept and measured. Giving them an encoding later (the two spare
// vsetvli immediate bits, say) changes only what drives p3109_fmt_hi.
//
// What changes with the format list
// ---------------------------------
//   * the unified rounder's datapath is as wide as the widest format
//     (P3109Rounder, intSig);
//   * the FMA's 8-bit core must hold every format exactly (coreFor, below),
//     and so must every other core an 8-bit lane runs on (FPFMAPipe);
//   * the operand reader decodes every format (p3109ToCore, below).
// Conversion into BF16 needs nothing new: BF16 holds every signed binary8pP
// with P <= 7 exactly.
// =============================================================================

object P3109Sweep {
  /** Smallest IEEE-style FType that holds every binary8pP in `precisions`
    * exactly, subnormals included: enough exponent for the largest and the
    * smallest value, and as many significand bits as the widest format.
    * For the pair this is E5M3, the core Saturn already uses.
    */
  def coreFor(precisions: Seq[Int]): FType = {
    def maxExp(p: Int) = (1 << (7 - p)) - 1                 // largest unbiased exponent
    def minSub(p: Int) = 1 - (1 << (7 - p)) - (p - 1)        // smallest subnormal's exponent
    val sig = precisions.max
    val exp = (2 to 11).find { e =>
      val bias = (1 << (e - 1)) - 1
      bias >= precisions.map(maxExp).max && 1 - bias - (sig - 1) <= precisions.map(minSub).min
    }.get
    FType(exp, sig)
  }

  /** Choose among per-format values by format code. Two formats keep the
    * pair's exact structure (one mux on bit 0); more are padded to a power of
    * two with the first one, so unused codes read as binary8p4.
    */
  def pick[T <: Data](fmt: UInt, xs: Seq[T]): T =
    if (xs.size == 1) xs.head
    else if (xs.size == 2) Mux(fmt(0), xs(1), xs(0))
    else {
      val n = 1 << log2Ceil(xs.size)
      VecInit(xs ++ Seq.fill(n - xs.size)(xs.head))(fmt(log2Ceil(xs.size) - 1, 0))
    }
}


// -----------------------------------------------------------------------------
// Reads an 8-bit P3109 code, in any of the build's formats, as a core number
// -----------------------------------------------------------------------------
// The generic version of p3109ToE5M3. Every format is first decoded to the
// same shape -- an unbiased exponent and a significand with its leading 1 --
// and one encoder then writes that into the core's IEEE layout, as a normal
// number or, when it is too small for that, as a core subnormal. coreFor makes
// sure the core is wide enough that this never loses a bit.
object p3109ToCore {
  def apply(in: UInt, fmt: UInt, formats: P3109Formats): UInt = {
    val core   = formats.coreType
    val E      = core.exp
    val S      = core.sig
    val biasC  = (1 << (E - 1)) - 1
    val sign   = in(7)

    // The three special codes, the same in every P3109 format.
    val isZero = in === "h00".U
    val isNaN  = in === "h80".U

    // One decoder per format: (unbiased exponent, S-bit significand, is the
    // 0x7F/0xFF code an infinity in this format).
    val decoded = formats.list.map { case (p, finite) =>
      val f    = p - 1                        // fraction bits
      val bias = 1 << (7 - p)
      val ef   = in(6, f)                     // exponent field
      val fr   = in(f - 1, 0)                 // fraction field
      // Normal: 1.fr x 2^(ef - bias).
      // (-& widens the result: Chisel's plain - keeps the operands' width and
      // would wrap, e.g. -8 - 2 in four bits is +6.)
      val normExp = ef.zext -& bias.S
      // Subnormal: 0.fr x 2^(1 - bias). Shift the fraction up until its first
      // 1 becomes the leading 1; the exponent drops by one per place.
      val lz      = PriorityEncoder(Reverse(fr))
      val subFrac = ((fr << lz) << 1)(f - 1, 0)
      val subExp  = (0 - bias).S -& lz.zext
      val isSub   = ef === 0.U
      val exp  = Mux(isSub, subExp, normExp)
      val frac = Mux(isSub, subFrac, fr)
      val sig  = (1.U(1.W) ## frac ## 0.U((S - p).W))(S - 1, 0)
      (exp.pad(10), sig, (!finite).B)
    }
    val exp          = P3109Sweep.pick(fmt, decoded.map(_._1))
    val sig          = P3109Sweep.pick(fmt, decoded.map(_._2))
    val infCodesAreInf = P3109Sweep.pick(fmt, decoded.map(_._3))
    val isInf        = in(6, 0) === "h7F".U && infCodesAreInf

    // Encode into the core. A value below the core's smallest normal becomes a
    // core subnormal: the significand shifts right by how far short it falls.
    val isCoreNormal = exp >= (1 - biasC).S
    val normField    = (exp + biasC.S).asUInt.apply(E - 1, 0)
    val subShift     = ((1 - biasC).S - exp).asUInt
    val subField     = (sig >> subShift)(S - 2, 0)
    val number = Mux(isCoreNormal, sign ## normField ## sig(S - 2, 0),
                                   sign ## 0.U(E.W) ## subField)

    val nan      = 0.U(1.W) ## Fill(E, 1.U(1.W)) ## 1.U(1.W) ## 0.U((S - 2).W)
    val infinity = sign ## Fill(E, 1.U(1.W)) ## 0.U((S - 1).W)
    Mux(isNaN, nan, Mux(isInf, infinity, Mux(isZero, 0.U((E + S).W), number)))
  }
}
