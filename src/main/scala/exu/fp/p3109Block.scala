package saturn.exu

import chisel3._
import chisel3.util._
import hardfloat.RawFloat

// =============================================================================
// IEEE P3109 block operations: applying and removing a block scale
// =============================================================================
//
// What this file is for
// ---------------------
// A "block" in P3109 (section 5.1) is a scale factor s together with a sequence
// of elements x_1 .. x_B. Everything the standard defines on blocks is built
// out of just two steps:
//
//   omegaBlockDecode  (5.4.1)   Z_i = Multiply(decode(s), decode(x_i))
//                               -- multiply by the scale on the way IN
//
//   omegaBlockProject (5.4.2)   Z_i = Divide(X_i, decode(s)), then round
//                               -- divide by the scale on the way OUT
//
// and then:
//
//   ConvertFromBlock  (5.5.1) = decode, then round             (dequantise)
//   ConvertToBlock    (5.5.2) = divide by the scale, then round  (quantise)
//   ScaledOp          (5.8)   = decode, operate, round
//   BlockOp   (B > 1) (5.7)   = decode, operate, divide, round
//
// So this file has exactly two functions, and everything above is one of them,
// the other, or both with an operation in between.
//
// Why this is cheap
// -----------------
// The scale format is Binary8p1uf (section 4.5, Fs). Its precision is 1, which
// means it has no fraction bits at all: every scale is a power of two. The NOTE
// at the end of section 5.8 says so outright.
//
// Multiplying by a power of two does not touch the significand. It only moves
// the exponent. So "multiply by the scale" is an integer ADD on the exponent
// and "divide by the scale" is an integer SUBTRACT -- no multiplier, no
// rounding, nothing else.
//
// What a Binary8p1uf code means
// -----------------------------
// It is 8 bits, unsigned, finite domain, precision 1, so (section 3.1) its bias
// is B = 2^(K-P) = 2^7 = 128, and there is no sign bit and no fraction field.
// The code point IS the biased exponent:
//
//     code 0         -> zero
//     code 1 .. 254  -> 2^(code - 128)
//     code 255       -> NaN            (unsigned formats put NaN at 2^K - 1)
//
// There are no infinities (the domain is finite) and no negative values (it is
// unsigned). That removes several cases the general definitions in 5.4 have to
// allow for, and the comments below say where.
//
// Note in passing: E8M0, the scale used by OCP Microscaling, is the same 8 bits
// with a bias of 127 instead of 128 -- so a P3109 scale means exactly half what
// the same byte means in MX. This is the same "half as much" relation that
// binary8p4 has with E4M3 (see p3109Fp8.scala).

object P3109Scale {
  val Bits = 8
  val Bias = 128      // 2^(K-P) for K=8, P=1
  val NaNCode = 255
  val MinExp = 1   - Bias   // -127, from code 1
  val MaxExp = 254 - Bias   // +126, from code 254
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------
object P3109ScaleHelper {
  import P3109Scale._

  def isNaN(scale: UInt)  = scale === NaNCode.U
  def isZero(scale: UInt) = scale === 0.U

  // The exponent the scale stands for, as a signed number: code - 128.
  def exponent(scale: UInt): SInt = (0.U(1.W) ## scale).asSInt - Bias.S

  // Move a raw number's exponent by `delta`, and saturate rather than wrap.
  //
  // Saturating is safe here because the only consumer of the result is a
  // rounder, and a rounder only ever compares the exponent against the target
  // format's limits. A number that has been pushed far above the largest
  // exponent the field can hold is an overflow either way; one pushed far below
  // is an underflow either way. Wrapping, by contrast, would silently turn a
  // huge number into a small one.
  def shiftExp(in: RawFloat, delta: SInt): RawFloat = {
    val w = in.sExp.getWidth
    val wide = in.sExp +& delta                     // one bit wider, cannot wrap
    val hi = ((BigInt(1) << (w - 1)) - 1).S(w.W)    // largest value the field holds
    val lo = (-(BigInt(1) << (w - 1))).S(w.W)
    val out = WireInit(in)
    out.sExp := Mux(wide > hi.pad(wide.getWidth), hi,
                Mux(wide < lo.pad(wide.getWidth), lo, wide(w - 1, 0).asSInt))
    out
  }

  def makeNaN(in: RawFloat): RawFloat = {
    val out = WireInit(in)
    out.isNaN := true.B
    out.isInf := false.B
    out.isZero := false.B
    out
  }

  def makeZero(in: RawFloat, sign: Bool): RawFloat = {
    val out = WireInit(in)
    out.isNaN := false.B
    out.isInf := false.B
    out.isZero := true.B
    out.sign := sign
    out
  }
}

// -----------------------------------------------------------------------------
// omegaBlockDecode (5.4.1): multiply a block element by its scale
// -----------------------------------------------------------------------------
// Z = Multiply(S, X), so the special cases are Multiply's, from section 4.10.4:
//
//   Multiply(NaN, *)         -> NaN        either operand NaN
//   Multiply(*, NaN)         -> NaN
//   Multiply(+Inf, 0)        -> NaN        infinity times zero
//   Multiply(X, 0)           -> 0          otherwise, zero absorbs
//   Multiply(+Inf, Y)        -> +/-Inf     sign from Y
//   Multiply(X, Y)           -> X * Y
//
// The scale is always zero, a positive power of two, or NaN -- never negative
// and never infinite -- so the sign of the result is always the element's sign,
// and the "scale is infinite" rows of 4.10.4 cannot occur.
//
// The caller must pass an `in` whose exponent field has room for the shift: an
// E5M3 raw number has only a 7-bit signed exponent, which a scale of 2^126
// would overflow. Widen it first (resizeRawFloat to a BF16 shape is enough).
object p3109ApplyScale {
  import P3109ScaleHelper._

  def apply(in: RawFloat, scale: UInt): RawFloat = {
    val scaleNaN  = isNaN(scale)
    val scaleZero = isZero(scale)

    // Infinity times zero is the one combination that makes a NaN out of two
    // non-NaN operands.
    val nanOut = in.isNaN || scaleNaN || (in.isInf && scaleZero)
    // Zero times anything finite is zero. (Zero times infinity was caught
    // above; zero times zero is zero.)
    val zeroOut = !nanOut && (in.isZero || scaleZero)

    val shifted = shiftExp(in, exponent(scale))
    Mux(nanOut, makeNaN(in),
    Mux(zeroOut, makeZero(in, in.sign),
                 shifted))
  }
}

// -----------------------------------------------------------------------------
// omegaBlockProject (5.4.2): remove the scale before rounding into a block
// -----------------------------------------------------------------------------
// Section 5.4.2 gives this case list directly, in order:
//
//   Z = NaN                      if S is NaN or X is NaN
//     = 0                        if S = 0
//     = sgn(X) * sgn(S)          if S = +/-Inf
//     = Divide(X, S)             otherwise
//
// Note the second row: if the scale is zero, every element of the block becomes
// zero, including the infinities (NOTE 1 of 5.4.2 says so). That is different
// from the decode direction, where infinity times zero is NaN.
//
// The third row cannot happen with Binary8p1uf, whose domain is finite, so it
// is not built.
//
// In the last row, Divide's own special cases (4.10.5) still apply, but with a
// scale that is a nonzero finite power of two only two of them are reachable:
// Divide(+/-Inf, S) -> +/-Inf and Divide(0, S) -> 0. Both fall out of shifting
// the exponent of a number that is already flagged infinite or zero, since the
// flags travel with it.
object p3109RemoveScale {
  import P3109ScaleHelper._

  def apply(in: RawFloat, scale: UInt): RawFloat = {
    val scaleNaN  = isNaN(scale)
    val scaleZero = isZero(scale)

    val nanOut  = in.isNaN || scaleNaN
    val zeroOut = !nanOut && scaleZero     // a zero scale flattens the whole block

    val shifted = shiftExp(in, -exponent(scale))
    Mux(nanOut, makeNaN(in),
    Mux(zeroOut, makeZero(in, in.sign),
                 shifted))
  }
}
