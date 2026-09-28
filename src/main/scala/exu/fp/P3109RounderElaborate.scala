package saturn.exu

import chisel3._
import freechips.rocketchip.tile.FType

/** The conversion unit's use of the rounder: a BF16 pattern in, an 8-bit
  * P3109 code out. Exists so a standalone testbench can drive raw bit patterns
  * without reproducing rawFloatFromFN in C++.
  */
class P3109ConvWrapper(formats: P3109Formats, name: String) extends RawModule {
  override def desiredName = name
  val io = IO(new Bundle {
    val in             = Input(UInt(16.W))
    val altfmt         = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val sat            = Input(Bool())
    val out            = Output(UInt(8.W))
    val exceptionFlags = Output(UInt(5.W))
  })
  val rounder = Module(new P3109Rounder(8, 8, formats, sigMSBitAlwaysZero = true))
  rounder.io.in             := hardfloat.rawFloatFromFN(8, 8, io.in)
  rounder.io.altfmt         := io.altfmt
  rounder.io.roundingMode   := io.roundingMode
  rounder.io.sat            := io.sat
  rounder.io.invalidExc     := false.B
  rounder.io.detectTininess := hardfloat.consts.tininess_afterRounding
  io.out            := rounder.io.out
  io.exceptionFlags := rounder.io.exceptionFlags
}

/** Same fix-up the conversion unit applies after resizeRawFloat: the hardfloat
  * routine leaves the exponent of a NaN or an infinity wrong when it widens. */
object resizeRawFixed {
  def apply(t: FType, in: hardfloat.RawFloat): hardfloat.RawFloat = {
    val out = WireInit(hardfloat.resizeRawFloat(t.exp, t.sig, in))
    when (in.isNaN || in.isInf) {
      out.sExp := ((1 << t.exp) + (1 << (t.exp - 1))).U(t.exp, 0).zext
    }
    out
  }
}

/** ConvertToBlock (P3109 5.5.2): a BF16 pattern and a Binary8p1uf scale in, an
  * 8-bit P3109 code out. The scale is divided out before rounding. */
class P3109ToBlockWrapper(formats: P3109Formats, name: String) extends RawModule {
  override def desiredName = name
  val io = IO(new Bundle {
    val in             = Input(UInt(16.W))
    val scale          = Input(UInt(8.W))
    val altfmt         = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val sat            = Input(Bool())
    val out            = Output(UInt(8.W))
    val exceptionFlags = Output(UInt(5.W))
  })
  val raw = hardfloat.rawFloatFromFN(8, 8, io.in)
  val rounder = Module(new P3109Rounder(8, 8, formats, sigMSBitAlwaysZero = true))
  rounder.io.in             := p3109RemoveScale(raw, io.scale)
  rounder.io.altfmt         := io.altfmt
  rounder.io.roundingMode   := io.roundingMode
  rounder.io.sat            := io.sat
  rounder.io.invalidExc     := hardfloat.isSigNaNRawFloat(raw)
  rounder.io.detectTininess := hardfloat.consts.tininess_afterRounding
  io.out            := rounder.io.out
  io.exceptionFlags := rounder.io.exceptionFlags
}

/** ConvertFromBlock (P3109 5.5.1): an 8-bit P3109 code and a Binary8p1uf scale
  * in, a BF16 pattern out. The scale is multiplied in after the exponent field
  * has been widened to make room for it. */
class P3109FromBlockWrapper(formats: P3109Formats, name: String) extends RawModule {
  override def desiredName = name
  val io = IO(new Bundle {
    val in             = Input(UInt(8.W))
    val scale          = Input(UInt(8.W))
    val altfmt         = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val out            = Output(UInt(16.W))
    val exceptionFlags = Output(UInt(5.W))
  })
  val e5m3 = p3109ToE5M3(io.in, io.altfmt,
    formats.p4 == P3109Domain.Finite, formats.p3 == P3109Domain.Finite)
  val raw8 = hardfloat.rawFloatFromFN(MXFType.E5M3.exp, MXFType.E5M3.sig, e5m3)
  val wide = resizeRawFixed(MXFType.BF16, raw8)

  val round = Module(new hardfloat.RoundAnyRawFNToRecFN(
    MXFType.BF16.exp, MXFType.BF16.sig, MXFType.BF16.exp, MXFType.BF16.sig,
    hardfloat.consts.flRoundOpt_sigMSBitAlwaysZero))
  round.io.in             := p3109ApplyScale(wide, io.scale)
  round.io.invalidExc     := hardfloat.isSigNaNRawFloat(raw8)
  round.io.infiniteExc    := false.B
  round.io.roundingMode   := io.roundingMode
  round.io.detectTininess := hardfloat.consts.tininess_afterRounding
  io.out            := MXFType.BF16.ieee(round.io.out)
  io.exceptionFlags := round.io.exceptionFlags
}

/** The multiply-add unit's use of the rounder, for one core shape.
  *
  * Takes the RawFloat fields directly, because the FMA core hands its rounder a
  * raw number that is not an IEEE bit pattern -- the significand can be in
  * [2,4), and everything below the core's precision is already OR-ed into a
  * sticky bit. The testbench drives exactly what the Python model builds with
  * raw_from_exact.
  *
  * Calls rawUnroundedToP3109Unified, the same function FPFMAPipe calls, so the
  * logic under test is the logic in the design -- nothing is re-created here.
  */
class P3109FmaRoundWrapper(core: FType, formats: P3109Formats, name: String) extends RawModule {
  override def desiredName = name
  val inSigWidth = core.sig + 2   // what MulAddRecFNPipeUnrounded produces
  val io = IO(new Bundle {
    val isNaN          = Input(Bool())
    val isInf          = Input(Bool())
    val isZero         = Input(Bool())
    val sign           = Input(Bool())
    val sExp           = Input(SInt((core.exp + 2).W))
    val sig            = Input(UInt((inSigWidth + 1).W))
    val altfmt         = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val out            = Output(UInt(8.W))
    val exceptionFlags = Output(UInt(5.W))
  })
  val raw = Wire(new hardfloat.RawFloat(core.exp, inSigWidth))
  raw.isNaN  := io.isNaN
  raw.isInf  := io.isInf
  raw.isZero := io.isZero
  raw.sign   := io.sign
  raw.sExp   := io.sExp
  raw.sig    := io.sig
  val (out, flags) = rawUnroundedToP3109Unified(core, raw, false.B, io.altfmt, io.roundingMode, formats)
  io.out            := out
  io.exceptionFlags := flags
}

// =============================================================================
// Development harness, not part of any design.
// =============================================================================
//
// Nothing instantiates P3109Rounder yet, so a plain compile only type-checks it:
// widths and bit ranges are worked out while the module is being built, not
// while it is being compiled. This emits it so those errors show up, and so the
// Verilog can be fed to a standalone Verilator testbench.
//
//   java -cp .classpath_cache/chipyard.jar saturn.exu.P3109RounderElaborate [outdir]
//
// Delete this file once the rounder is instantiated by the conversion unit.
// =============================================================================
object P3109RounderElaborate extends App {
  val dir = if (args.nonEmpty) args(0) else "."
  val opts = Array("-disable-all-randomization", "-strip-debug-info")

  def emit(gen: => RawModule, file: String): Unit = {
    val sv = circt.stage.ChiselStage.emitSystemVerilog(gen, firtoolOpts = opts)
    java.nio.file.Files.write(java.nio.file.Paths.get(s"$dir/$file"), sv.getBytes)
    println(s"wrote $dir/$file  (${sv.linesIterator.size} lines)")
  }

  val extended = P3109Formats(P3109Domain.Extended, P3109Domain.Extended)
  val finite   = P3109Formats(P3109Domain.Finite,   P3109Domain.Finite)

  emit(new P3109ConvWrapper(extended, "P3109ConvExt"), "P3109ConvExt.sv")
  emit(new P3109ConvWrapper(finite,   "P3109ConvFin"), "P3109ConvFin.sv")

  // The shape the FMA hands it: two extra significand bits, and a significand
  // that can reach 4, so doShiftSigDown1 is live. Elaboration check only.
  emit(new P3109Rounder(8, 10, extended, sigMSBitAlwaysZero = false), "P3109Rounder_fma.sv")

  // The two halves of the block machinery (p3109Block.scala).
  emit(new P3109ToBlockWrapper(extended,   "P3109ToBlockExt"),   "P3109ToBlockExt.sv")
  emit(new P3109ToBlockWrapper(finite,     "P3109ToBlockFin"),   "P3109ToBlockFin.sv")
  emit(new P3109FromBlockWrapper(extended, "P3109FromBlockExt"), "P3109FromBlockExt.sv")
  emit(new P3109FromBlockWrapper(finite,   "P3109FromBlockFin"), "P3109FromBlockFin.sv")

  // The FMA's 8-bit rounder, once per core shape an 8-bit lane can run on
  // (see ftype_used_for in FPFMAPipe.scala), in both domains.
  val fmaCores = Seq("FP64" -> FType.D, "FP32" -> FType.S, "FP16" -> FType.H,
                     "BF16" -> MXFType.BF16, "E5M3" -> MXFType.E5M3)
  for ((coreName, core) <- fmaCores; (dom, fmts) <- Seq("Ext" -> extended, "Fin" -> finite)) {
    val n = s"P3109FmaRound${coreName}${dom}"
    emit(new P3109FmaRoundWrapper(core, fmts, n), s"$n.sv")
  }
}
