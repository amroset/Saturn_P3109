package saturn.exu

import chisel3._

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
}
