package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import saturn.common._

// Scaled mode (docs/scaled-mode.md): one element of a scaled FMA instruction.
//
// A scaled 8-bit operand lies within 2^-144 and 2^142, outside every lane of
// SegmentedFMAPipe (exponents of 5 or 8 bits), so scaled instructions run on
// these lanes instead. Each computes in a recoded format with a 10-bit
// exponent and an 8-bit significand (W): every input has at most 8
// significant bits (8-bit formats 4, BF16 8), so decoding, widening and
// scaling are exact, and the only rounding is the final one, to FP8 or BF16.
//
// The scales are folded into the operands' exponents. With e(x) the scale
// exponent of operand x (0 if x is not a block) and er that of the result
// block (0 for a BF16 result, or a zero or NaN result scale):
//   Z / 2^er = (a 2^(e(a) - er)) (b 2^e(b)) + c 2^(e(c) - er)
// |e(a) - er| <= 253, so a scaled operand stays within 2^-270 and 2^269,
// inside W's normal range. A zero or NaN result scale (P3109 5.4.2) is applied
// to the rounded result: NaN, or zero for anything but NaN.
class ScaledFMALane(depth: Int, formats: P3109Formats) extends Module {
  require(depth == 4, "the FMA pipe's own timing: decode, 2-stage FMA, round")
  val W = new FType(FType.S.exp + 2, MXFType.BF16.sig)

  val io = IO(new Bundle {
    val valid = Input(Bool())
    val op = Input(UInt(2.W))    // FMA command: negate product, negate addend
    val frm = Input(UInt(3.W))
    val altfmt = Input(Bool())   // format of the 8-bit operands and result
    val addsub = Input(Bool())   // a * 1 + c
    val mul = Input(Bool())      // a * b + 0
    // Elements: an 8-bit code in bits 7:0, or a BF16 value (*_wide)
    val a = Input(UInt(16.W))
    val b = Input(UInt(16.W))
    val c = Input(UInt(16.W))
    val a_wide = Input(Bool())
    val b_wide = Input(Bool())
    val c_wide = Input(Bool())
    // Binary8p1uf scale codes; 128 (2^0) for an operand that is not a block
    val sa = Input(UInt(8.W))
    val sb = Input(UInt(8.W))
    val sc = Input(UInt(8.W))
    val sr = Input(UInt(8.W))    // the result block's scale, if out_wide is false
    val out_wide = Input(Bool()) // BF16 result
    val out = Output(UInt(16.W))
    val exc = Output(UInt(5.W))
  })

  def widen(in: UInt, from: FType): UInt = {
    val w = Module(new hardfloat.RecFNToRecFN(from.exp, from.sig, W.exp, W.sig))
    w.io.in := in
    w.io.roundingMode := hardfloat.consts.round_near_even // exact: W is wider
    w.io.detectTininess := hardfloat.consts.tininess_afterRounding
    w.io.out
  }
  def decode(x: UInt, wide: Bool): UInt = Mux(wide,
    widen(MXFType.BF16.recode(x(15, 0)), MXFType.BF16),
    widen(MXFType.E5M3.recode(p3109ToE5M3(x(7, 0), io.altfmt, formats)), MXFType.E5M3))

  val qnan = Cat(0.U(1.W), 7.U(3.W), 0.U((W.exp - 2).W), 1.U(1.W), 0.U((W.sig - 2).W))
  def exponent(code: UInt): SInt = (0.U(1.W) ## code).asSInt - 128.S
  // x times scale code `code`, its exponent also moved by `off` (P3109
  // omegaBlockDecode, 5.4.1): NaN scale, or zero scale times Inf or NaN, gives
  // NaN; zero scale otherwise gives zero
  def scale(x: UInt, code: UInt, off: SInt): UInt = {
    val sign = x(W.exp + W.sig)
    val exp = x(W.exp + W.sig - 1, W.sig - 1)
    val isZero = exp(W.exp, W.exp - 2) === 0.U
    val isSpecial = exp(W.exp, W.exp - 1).andR
    val moved = Cat(sign, (exp.zext + exponent(code) + off)(W.exp, 0), x(W.sig - 2, 0))
    Mux(code === 255.U || (code === 0.U && isSpecial), qnan,
      Mux(code === 0.U, Cat(sign, 0.U((W.exp + W.sig).W)),
        Mux(isZero || isSpecial, x, moved)))
  }

  val sr_special = io.sr === 0.U || io.sr === 255.U
  val er = Mux(io.out_wide || sr_special, 0.S, exponent(io.sr))
  val a = scale(decode(io.a, io.a_wide), io.sa, -er)
  val b = scale(decode(io.b, io.b_wide), io.sb, 0.S)
  val c = scale(decode(io.c, io.c_wide), io.sc, -er)
  val one = W.recode((((BigInt(1) << (W.exp - 1)) - 1) << (W.sig - 1)).U(W.ieeeWidth.W))
  val b_in = Mux(io.addsub, one, b)
  val c_in = Mux(io.mul, (a ^ b) & (1.U << W.ieeeWidth), c) // signed zero of a * b

  val fma = Module(new MulAddRecFNPipeUnrounded(depth - 2, W.exp, W.sig))
  fma.io.validin := RegNext(io.valid, false.B)
  fma.io.op := RegEnable(io.op, io.valid)
  fma.io.roundingMode := RegEnable(io.frm, io.valid)
  fma.io.detectTininess := hardfloat.consts.tininess_afterRounding
  fma.io.a := RegEnable(a, io.valid)
  fma.io.b := RegEnable(b_in, io.valid)
  fma.io.c := RegEnable(c_in, io.valid)

  // Control for the rounding, at the FMA's output
  val frm_r = Pipe(io.valid, io.frm, depth - 1).bits
  val altfmt_r = Pipe(io.valid, io.altfmt, depth - 1).bits
  val out_wide_r = Pipe(io.valid, io.out_wide, depth - 1).bits
  val sr_r = Pipe(io.valid, io.sr, depth - 1).bits

  // FP8 result, then the result scale's special cases (P3109 5.4.2)
  val (out8, exc8) = rawUnroundedToP3109(W, fma.io.out, fma.io.invalidExc, altfmt_r.asUInt, frm_r, formats)
  val nan8 = 0x80.U(8.W)
  val res8 = Mux(sr_r === 255.U, nan8, Mux(sr_r === 0.U && out8 =/= nan8, 0.U(8.W), out8))

  // BF16 result. P3109 encodes its one zero in BF16 as +0 (4.8.2).
  val round16 = Module(new hardfloat.RoundAnyRawFNToRecFN(W.exp, W.sig + 2, MXFType.BF16.exp, MXFType.BF16.sig, 0))
  round16.io.in := fma.io.out
  round16.io.invalidExc := fma.io.invalidExc
  round16.io.infiniteExc := false.B
  round16.io.roundingMode := frm_r
  round16.io.detectTininess := hardfloat.consts.tininess_afterRounding
  val bf16 = MXFType.BF16.ieee(round16.io.out)
  val res16 = Mux(bf16(14, 0) === 0.U, 0.U(16.W), bf16)

  io.out := Pipe(fma.io.validout, Mux(out_wide_r, res16, 0.U(8.W) ## res8), depth - 4).bits
  io.exc := Pipe(fma.io.validout, Mux(out_wide_r, round16.io.exceptionFlags, exc8), depth - 4).bits
}
