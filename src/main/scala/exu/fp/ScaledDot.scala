package saturn.exu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import saturn.common._

// vfdot4 (docs/vfdot4.md): one FP32 lane. a and b each hold four 8-bit
// elements (element j in bits 8j+7:8j); sa and sb are their Binary8p1uf scales.
//   out = c + sa sb (a0 b0 + a1 b1 + a2 b2 + a3 b3), rounded once to FP32 (frm)
// An element is k 2^(q-18) with k < 16 and 1 <= q <= 30 (E5M3), so a product
// is an integer multiple of 2^-34 below 2^32, and the sum S is exact as a
// 69-bit integer. S converts exactly to D (10-bit exponent, 70-bit
// significand), the scales add to its exponent, c joins it exactly, and the
// result is rounded once. A zero or NaN scale applies to every element of its
// operand, as P3109's block decoding (5.4.1) does.
class ScaledDotLane(depth: Int, formats: P3109Formats) extends Module {
  require(depth == 4, "the FMA pipe's own timing: products and sum, scaling, addition, rounding")
  val D = new FType(10, 70)

  val io = IO(new Bundle {
    val valid = Input(Bool())
    val frm = Input(UInt(3.W))
    val altfmt = Input(Bool())
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val c = Input(UInt(32.W))   // FP32 accumulator
    val sa = Input(UInt(8.W))
    val sb = Input(UInt(8.W))
    val out = Output(UInt(32.W))
    val exc = Output(UInt(5.W))
  })

  // ---- stage 0: decode, products, exact sum
  class Elem(code: UInt, s: UInt) {
    val e = p3109ToE5M3(code, io.altfmt, formats)
    val sign = e(8)
    val exp = e(7, 3)
    val man = e(2, 0)
    val special = exp.andR
    val zero = exp === 0.U && man === 0.U
    // after the scale (block decoding): NaN scale, or zero scale times Inf or
    // NaN, is NaN; a zero scale makes every other element zero
    val nan = (special && man =/= 0.U) || s === 255.U || (s === 0.U && special)
    val inf = special && man === 0.U && !nan
    val isZero = !nan && !inf && (zero || s === 0.U)
    val k = Mux(exp === 0.U, 0.U(1.W), 1.U(1.W)) ## man
    val q = Mux(exp === 0.U, 1.U(5.W), exp)
  }
  val as = Seq.tabulate(4)(j => new Elem(io.a(8 * j + 7, 8 * j), io.sa))
  val bs = Seq.tabulate(4)(j => new Elem(io.b(8 * j + 7, 8 * j), io.sb))
  val prods = as.zip(bs).map { case (x, y) =>
    val nan = x.nan || y.nan || (x.inf && y.isZero) || (y.inf && x.isZero)
    val inf = (x.inf || y.inf) && !nan
    val sign = x.sign ^ y.sign
    val mag = ((x.k * y.k) << (x.q +& y.q - 2.U))(65, 0)
    val v = Mux(sign, -(0.U(1.W) ## mag).zext, (0.U(1.W) ## mag).zext)
    (nan, inf, sign, Mux(x.isZero || y.isZero || nan || inf, 0.S, v))
  }
  val sum = prods.map(_._4.pad(69)).reduce(_ +& _)(68, 0).asSInt
  val posInf = prods.map(p => p._2 && !p._3).reduce(_ || _)
  val negInf = prods.map(p => p._2 && p._3).reduce(_ || _)
  val genNaN = prods.map(p => p._1).reduce(_ || _) || (posInf && negInf)
  // 0 * Inf and Inf - Inf are invalid; a NaN element or scale is not
  val inv0 = as.zip(bs).map { case (x, y) => (x.inf && y.isZero) || (y.inf && x.isZero) }.reduce(_ || _) ||
    (posInf && negInf && !prods.map(_._1).reduce(_ || _))
  val escale = (0.U(2.W) ## io.sa).asSInt +& (0.U(2.W) ## io.sb).asSInt - (256 + 34).S

  val v1 = RegNext(io.valid, false.B)
  val s1 = RegEnable(sum, io.valid)
  val nan1 = RegEnable(genNaN, io.valid)
  val inf1 = RegEnable(posInf || negInf, io.valid)
  val infSign1 = RegEnable(negInf, io.valid)
  val inv1 = RegEnable(inv0, io.valid)
  val esc1 = RegEnable(escale, io.valid)
  val c1 = RegEnable(io.c, io.valid)
  val frm1 = RegEnable(io.frm, io.valid)

  // ---- stage 1: S to D (exact), times sa sb 2^-34, c to D (exact)
  val i2f = Module(new hardfloat.INToRecFN(69, D.exp, D.sig))
  i2f.io.signedIn := true.B
  i2f.io.in := s1.asUInt
  i2f.io.roundingMode := hardfloat.consts.round_near_even // exact: 69 bits fit
  i2f.io.detectTininess := hardfloat.consts.tininess_afterRounding
  val rs = i2f.io.out
  val rsExp = rs(D.exp + D.sig - 1, D.sig - 1)
  val rsZero = rsExp(D.exp, D.exp - 2) === 0.U
  val scaled = Cat(rs(D.exp + D.sig), (rsExp.zext + esc1)(D.exp, 0), rs(D.sig - 2, 0))
  val qnan = Cat(0.U(1.W), 7.U(3.W), 0.U((D.exp - 2).W), 1.U(1.W), 0.U((D.sig - 2).W))
  val inf = Cat(infSign1, 6.U(3.W), 0.U((D.exp + D.sig - 3).W))
  val p = Mux(nan1, qnan, Mux(inf1, inf, Mux(rsZero, rs, scaled)))
  val widen = Module(new hardfloat.RecFNToRecFN(FType.S.exp, FType.S.sig, D.exp, D.sig))
  widen.io.in := FType.S.recode(c1)
  widen.io.roundingMode := hardfloat.consts.round_near_even // exact: D is wider
  widen.io.detectTininess := hardfloat.consts.tininess_afterRounding

  val v2 = RegNext(v1, false.B)
  val p2 = RegEnable(p, v1)
  val c2 = RegEnable(widen.io.out, v1)
  val inv2 = RegEnable(inv1, v1)
  val frm2 = RegEnable(frm1, v1)

  // ---- stage 2: c + p, exact
  val add = Module(new hardfloat.AddRawFN(D.exp, D.sig))
  add.io.subOp := false.B
  add.io.a := hardfloat.rawFloatFromRecFN(D.exp, D.sig, c2)
  add.io.b := hardfloat.rawFloatFromRecFN(D.exp, D.sig, p2)
  add.io.roundingMode := frm2

  val raw3 = RegEnable(add.io.rawOut, v2)
  val inv3 = RegEnable(add.io.invalidExc || inv2, v2)
  val frm3 = RegEnable(frm2, v2)

  // ---- stage 3: one rounding, to FP32. P3109 encodes its one zero in
  // binary32 as +0 (Interim Report v4.0.3, 4.8.2).
  val round = Module(new hardfloat.RoundAnyRawFNToRecFN(D.exp, D.sig + 2, FType.S.exp, FType.S.sig, 0))
  round.io.in := raw3
  round.io.invalidExc := inv3
  round.io.infiniteExc := false.B
  round.io.roundingMode := frm3
  round.io.detectTininess := hardfloat.consts.tininess_afterRounding
  val out32 = FType.S.ieee(round.io.out)
  io.out := Mux(out32(30, 0) === 0.U, 0.U, out32)
  io.exc := round.io.exceptionFlags
}
