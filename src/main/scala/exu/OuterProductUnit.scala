package saturn.exu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket._
import freechips.rocketchip.util._
import freechips.rocketchip.tile._
import chisel3.util.experimental.decode._
import saturn.common._
import saturn.backend._
import hardfloat._
import scala.math._

// Declare supported types (dictates multipler instantiated)
object OPUTypes extends Enumeration { val INT32, INT16, INT8, E4M3, E5M2 = Value }

// Parameters for configured OPU
case class OPUParameters (
  val aWidth : Int = 8,
  val bWidth : Int = 8,
  val cWidth : Int = 32, // Accumulator size

  val nMrfRegs : Int = 4,
  val nScaleEntries : Int = 4 // OPFMACC.S scale cache: scale registers held
)

trait HasOPUParams extends HasVectorParams { this: HasCoreParameters =>
  def maxLMUL = 2 // TODO: make this dynamic
  def regsPerTileReg = (vLen/dLen) * (vLen/dLen)
  def regsPerCell = regsPerTileReg * opuParams.nMrfRegs
  def cellRegIdxBits = log2Ceil(regsPerCell) 
  def prodWidth = opuParams.aWidth + opuParams.bWidth

  def clusterXdim = opuParams.cWidth / opuParams.bWidth
  def clusterYdim = opuParams.cWidth / opuParams.aWidth

  def yDim = (dLen / opuParams.aWidth) / clusterYdim
  def xDim = (dLen / opuParams.bWidth) / clusterXdim

}

class OuterProductCell(implicit p: Parameters) extends CoreModule()(p) with HasOPUParams {

  val io = IO(new Bundle{
    // Data signals
    val in_l = Input(SInt(opuParams.aWidth.W)) // left input
    val in_t = Input(SInt(opuParams.bWidth.W)) // top input
    // Block scales for OPFMACC.S, P3109 Binary8p1uf: 0 is zero, 255 is NaN,
    // otherwise 2^(code - 128); 128 (2^0) for every other operation
    val scale_l = Input(UInt(8.W))
    val scale_t = Input(UInt(8.W))

    // Contol Signals
    val altfmt = if (useMxOPU) Some(Input(Bool())) else None // alternate format for outer product
    val fp8 = if (useMxOPU) Some(Input(Bool())) else None // FP8 format for outer product
    val fp4 = if (vParams.opuFP4) Some(Input(Bool())) else None // Binary4p2sf pairs (element-format code 6)
    val mrf_idx = Input(UInt(cellRegIdxBits.W)) // Index for µarch register to write
    val macc = Input(Bool())
    val mvin = Input(Bool())
    val mvin_bcast = Input(Bool())
    val mvin_bcast_col = Input(Bool())
    val mvin_data = Input(UInt(opuParams.cWidth.W))
    val out = Output(UInt(opuParams.cWidth.W))
  })
  // Matrix Register + Logic
  val regs = Reg(Vec(regsPerCell, UInt(opuParams.cWidth.W)))
  val prod = Mux(io.macc, io.in_l * io.in_t, 0.S)
  val sum_int8 = prod + regs(io.mrf_idx).asSInt

  val int_macc = if (useMxOPU) (io.macc && !io.fp8.get) else io.macc

  val mrf_idx_pipe = Wire(UInt(cellRegIdxBits.W))
  val fp8_macc_complete = Wire(Bool())
  val sum_fp8 = Wire(UInt(opuParams.cWidth.W))

  if (useMxOPU) {
    def widen(in: UInt, inT: FType, outT: FType, active: Bool): UInt = {
      val widen = Module(new hardfloat.RecFNToRecFN(inT.exp, inT.sig, outT.exp, outT.sig))
      widen.io.in := Mux(active, in, 0.U)
      widen.io.roundingMode := hardfloat.consts.round_near_even
      widen.io.detectTininess := hardfloat.consts.tininess_afterRounding
      widen.io.out
    }
    val f8macc = io.macc && io.fp8.get
    def read8(u: UInt) = vParams.p3109 match {
      case Some(f) => p3109ToE5M3(u, io.altfmt.get, f)
      case None    => fp8ToE5M3(u, io.altfmt.get)
    }
    // c + sA*sB*a*b, rounded once to FP32. A scaled operand can leave FP32's
    // range even when the result does not, so the cell computes in a recoded
    // format with a 10-bit exponent. A product of two 8-bit elements has at
    // most 8 significant bits, so the scaled product is exact in that format,
    // and one addition and one rounding finish the multiply-accumulate. With
    // both scales 2^0 this is the unscaled OPFMACC.
    val wide = FType(FType.S.exp + 2, FType.S.sig)
    val latency = 2
    val scaleExp = io.scale_l.zext +& io.scale_t.zext - 256.S // log2(sA*sB)

    // An element times its scale, as P3109's omegaBlockDecode: a NaN scale,
    // or a zero scale times Inf or NaN, gives NaN; a zero scale otherwise zero
    class ScaledElem(code: UInt, s: UInt) {
      val raw = hardfloat.rawFloatFromFN(MXFType.E5M3.exp, MXFType.E5M3.sig, read8(code))
      val nan = raw.isNaN || s === 255.U || (s === 0.U && raw.isInf)
      val inf = raw.isInf && !nan
      val zero = !nan && !inf && (raw.isZero || s === 0.U)
    }
    val a = new ScaledElem(io.in_l.asUInt, io.scale_l)
    val b = new ScaledElem(io.in_t.asUInt, io.scale_t)
    val sig8 = a.raw.sig(3, 0) * b.raw.sig(3, 0) // 1.xxx * 1.yyy, in [1, 4)
    val carry = sig8(7)
    val prod8 = Wire(new hardfloat.RawFloat(wide.exp, wide.sig))
    prod8.isNaN := a.nan || b.nan || (a.inf && b.zero) || (a.zero && b.inf)
    prod8.isInf := (a.inf || b.inf) && !prod8.isNaN
    prod8.isZero := (a.zero || b.zero) && !prod8.isNaN
    prod8.sign := a.raw.sign ^ b.raw.sign
    // raw exponents are biased by 2^expWidth: E5M3's by 32, wide's by 1024
    prod8.sExp := (a.raw.sExp +& b.raw.sExp +& scaleExp +& carry.zext +& (1024 - 64).S)(wide.exp + 1, 0).asSInt
    prod8.sig := Cat(0.U(1.W), Mux(carry, sig8, sig8(6, 0) ## 0.U(1.W)), 0.U((wide.sig - 8).W))

    // FP4 (opuFP4, element-format code 6): each byte holds two P3109
    // Binary4p2sf values, step k in bits 3:0 and step k+1 in bits 7:4. Times 4,
    // every value is an integer of magnitude at most 12 (code 8 is NaN), so
    // a0*b0 + a1*b1 is exactly an integer of at most 288, 16 times the dot
    // product, and takes the 8-bit product's place.
    val prod = if (vParams.opuFP4) {
      def int4(c: UInt): SInt = {
        val mag = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
        Mux(c(3), -mag.zext, mag.zext)
      }
      val l = io.in_l.asUInt
      val t = io.in_t.asUInt
      val nan = Seq(l(3, 0), l(7, 4), t(3, 0), t(7, 4)).map(_ === 8.U).reduce(_ || _)
      val dot16 = int4(l(3, 0)) * int4(t(3, 0)) +& int4(l(7, 4)) * int4(t(7, 4))
      val i2f = Module(new hardfloat.INToRecFN(dot16.getWidth, wide.exp, wide.sig))
      i2f.io.signedIn := true.B
      i2f.io.in := dot16.asUInt
      i2f.io.roundingMode := hardfloat.consts.round_near_even // exact: at most 9 bits
      i2f.io.detectTininess := hardfloat.consts.tininess_afterRounding
      val dot = hardfloat.rawFloatFromRecFN(wide.exp, wide.sig, i2f.io.out)
      val prod4 = WireInit(dot)
      prod4.isNaN := nan || io.scale_l === 255.U || io.scale_t === 255.U
      prod4.isZero := (dot.isZero || io.scale_l === 0.U || io.scale_t === 0.U) && !prod4.isNaN
      prod4.sExp := (dot.sExp +& scaleExp - 4.S)(wide.exp + 1, 0).asSInt
      Mux(io.fp4.get, prod4, prod8)
    } else prod8

    // AddRawFN also reads a zero's exponent and significand, so give a zero
    // product those of a recoded zero
    val prodIn = WireInit(prod)
    when (prod.isZero) {
      prodIn.sExp := 0.S
      prodIn.sig := 0.U
    }
    val c = widen(FType.S.recode(regs(io.mrf_idx)), FType.S, wide, f8macc)
    val valid1 = RegNext(f8macc, false.B)
    val prod1 = RegEnable(prodIn, f8macc)
    val c1 = RegEnable(c, f8macc)

    val add = Module(new hardfloat.AddRawFN(wide.exp, wide.sig))
    add.io.subOp := false.B
    add.io.a := hardfloat.rawFloatFromRecFN(wide.exp, wide.sig, c1)
    add.io.b := prod1
    add.io.roundingMode := hardfloat.consts.round_near_even
    val valid2 = RegNext(valid1, false.B)
    val sum2 = RegEnable(add.io.rawOut, valid1)
    val invalid2 = RegEnable(add.io.invalidExc, valid1)

    val round = Module(new hardfloat.RoundAnyRawFNToRecFN(wide.exp, wide.sig + 2, FType.S.exp, FType.S.sig, 0))
    round.io.invalidExc := invalid2
    round.io.infiniteExc := false.B
    round.io.in := sum2
    round.io.roundingMode := hardfloat.consts.round_near_even
    round.io.detectTininess := hardfloat.consts.tininess_afterRounding
    mrf_idx_pipe := Pipe(f8macc, io.mrf_idx, latency).bits

    // P3109 has one zero, and encodes it in binary32 as +0 (Interim Report
    // v4.0.3, 4.8.2), where IEEE 754 can give -0; an OCP build keeps IEEE's.
    val sum32 = FType.S.ieee(round.io.out)
    val sum32_out = if (vParams.p3109.isDefined) Mux(sum32(30, 0) === 0.U, 0.U, sum32) else sum32
    sum_fp8 := Mux(valid2, sum32_out, 0.U)
    fp8_macc_complete := valid2
  } else {
    mrf_idx_pipe := 0.U
    fp8_macc_complete := false.B
    sum_fp8 := 0.U
  }

  for (i <- 0 until regsPerCell) {
    // Integer MACC: use current control signals
    val tile_match_int      = (io.mrf_idx >> log2Ceil(regsPerTileReg)) === (i >> log2Ceil(regsPerTileReg)).U
    val subtile_match_int   = io.mrf_idx(log2Ceil(regsPerTileReg)-1,0) === (i % regsPerTileReg).U
    
    // MVIN paths: use current control signals
    val bcast_col_match = (io.mrf_idx >> log2Ceil(vLen/dLen)) === (i >> log2Ceil(vLen/dLen)).U
    val bcast_match     = io.mrf_idx(log2Ceil(vLen/dLen)-1,0) === (i % (vLen/dLen)).U
    
    if (useMxOPU) {
      // FP8 MACC: use pipelined control signals
      val tile_match_fp8      = (mrf_idx_pipe >> log2Ceil(regsPerTileReg)) === (i >> log2Ceil(regsPerTileReg)).U
      val subtile_match_fp8   = mrf_idx_pipe(log2Ceil(regsPerTileReg)-1,0) === (i % regsPerTileReg).U
      when ((tile_match_int && int_macc && subtile_match_int) ||
            (tile_match_fp8 && fp8_macc_complete && subtile_match_fp8) ||
            (tile_match_int && io.mvin && subtile_match_int) ||
            (tile_match_int && io.mvin_bcast && bcast_match) ||
            (tile_match_int && io.mvin_bcast_col && bcast_col_match)) {
        regs(i) := Mux(int_macc, sum_int8.asUInt,
                  Mux(fp8_macc_complete, sum_fp8,
                      io.mvin_data))
        }
    } else {
      when ((tile_match_int && int_macc && subtile_match_int) ||
            (tile_match_int && io.mvin && subtile_match_int) ||
            (tile_match_int && io.mvin_bcast && bcast_match) ||
            (tile_match_int && io.mvin_bcast_col && bcast_col_match)) {
      regs(i) := Mux(int_macc, sum_int8.asUInt,
                     io.mvin_data)
      }
    }
  }
  io.out := regs(io.mrf_idx)
}

class OuterProductCluster(implicit p : Parameters) extends CoreModule()(p) with HasOPUParams {
  val io = IO(new Bundle{
    val in_l      = Input(Vec(clusterYdim, UInt(opuParams.aWidth.W)))
    val in_t      = Input(Vec(clusterXdim, UInt(opuParams.bWidth.W)))
    val scale_l   = Input(Vec(clusterYdim, UInt(8.W)))
    val scale_t   = Input(Vec(clusterXdim, UInt(8.W)))

    val in_pipe   = Input(UInt(opuParams.cWidth.W))
    val out_pipe  = Output(UInt(opuParams.cWidth.W))

    val mrf_idx = Input(UInt(cellRegIdxBits.W))
    val row_idx = Input(UInt(log2Ceil(clusterYdim).W))
    val col_idx = Input(UInt(log2Ceil(clusterXdim).W))

    val macc  = Input(Bool())
    val shift = Input(Bool())
    val mvin  = Input(Bool())
    val mvin_bcast = Input(Bool())
    val altfmt = if (useMxOPU) Some(Input(Bool())) else None // alternate format for outer product
    val fp8 = if (useMxOPU) Some(Input(Bool())) else None // FP8 format for outer product
    val fp4 = if (vParams.opuFP4) Some(Input(Bool())) else None // Binary4p2sf pairs
    val mvin_col = Input(Bool())
    val mvin_bcast_col = Input(Bool())
  })

  val cells = Seq.fill(clusterXdim, clusterYdim)(Module(new OuterProductCell))
  val cell_outs = Wire(Vec(clusterYdim, Vec(clusterXdim, UInt(opuParams.cWidth.W))))
  val pipe = Reg(UInt(opuParams.cWidth.W))

  for (i <- 0 until clusterYdim) {
    for (j <- 0 until clusterXdim) {
      val cell = cells(i)(j)

      cell.io.in_l  := io.in_l(i).asSInt
      cell.io.in_t  := io.in_t(j).asSInt
      cell.io.scale_l := io.scale_l(i)
      cell.io.scale_t := io.scale_t(j)
      cell.io.mrf_idx := io.mrf_idx
      if (useMxOPU) {
        cell.io.altfmt.get := io.altfmt.get
        cell.io.fp8.get := io.fp8.get
      }
      cell.io.fp4.foreach(_ := io.fp4.get)
      cell.io.macc := io.macc
      cell.io.mvin_bcast_col := io.mvin_bcast_col
      cell_outs(i)(j) := cell.io.out.asUInt

      cell.io.mvin_bcast := Mux(io.mvin_bcast, 
        j.U === io.col_idx,
        false.B
      )
      cell.io.mvin_bcast_col := Mux(io.mvin_bcast_col, 
        i.U === io.col_idx, 
        false.B
      )
      
      cell.io.mvin := Mux(io.mvin_col, 
        j.U === io.row_idx && i.U === io.col_idx,
        Mux(io.mvin, 
          i.U === io.row_idx && j.U === io.col_idx,
          false.B
        )
      )
      cell.io.mvin_data := 
        Mux(io.mvin_col || io.mvin_bcast_col, io.in_l.asUInt,
          Mux(io.mvin || io.mvin_bcast, io.in_t.asUInt,
            DontCare
          )
        )
    }
  }

  pipe := Mux(io.shift,
    io.in_pipe,
    cell_outs(io.row_idx)(io.col_idx)
  )

  io.out_pipe := pipe
}

class OuterProductControl(implicit p: Parameters) extends CoreBundle()(p) with HasOPUParams {
  val clock_enable = Bool()

  val in_l      = Vec(yDim, Vec(clusterYdim, UInt(opuParams.aWidth.W)))
  val in_t      = Vec(xDim, Vec(clusterXdim, UInt(opuParams.bWidth.W)))
  // OPFMACC.S scales. The sequencer keeps the scale cache's tags and decides
  // fills; the cached scale registers themselves are in OuterProductUnit.
  // sa_* are A's row scales (vs1+1), sb_* B's column scales (vs2+1).
  val scaled   = Bool() // apply the scales at sa_*/sb_*; else 2^0
  val sa_entry = UInt(log2Ceil(opuParams.nScaleEntries).W) // cache entry
  val sb_entry = UInt(log2Ceil(opuParams.nScaleEntries).W)
  val sa_piece = UInt(log2Ceil(egsPerVReg).W) // element group in the entry
  val sb_piece = UInt(log2Ceil(egsPerVReg).W)
  val sa_fill  = Bool() // store in_l at sa_entry, sa_piece
  val sb_fill  = Bool() // store in_t at sb_entry, sb_piece
  // Scaled mode: E rows (columns) share a scale byte, row r taking byte
  // r >> scale_lg; sa_quad/sb_quad are the issue's row/column quadrant
  val scale_lg = if (vParams.scaledMode) Some(UInt(3.W)) else None
  val sa_quad  = if (vParams.scaledMode) Some(UInt(log2Ceil(egsPerVReg).W)) else None
  val sb_quad  = if (vParams.scaledMode) Some(UInt(log2Ceil(egsPerVReg).W)) else None

  // same values broadcast horizontally
  val mrf_idx    = Vec(yDim, UInt(cellRegIdxBits.W)) 
  val row_idx    = Vec(yDim, UInt(log2Ceil(clusterYdim).W))
  val col_idx    = Vec(yDim, UInt(log2Ceil(clusterXdim).W))
  val macc       = Vec(yDim, Bool())
  val shift      = Vec(yDim, Bool())
  val mvin       = Vec(yDim, Bool())
  val mvin_bcast = Vec(yDim, Bool())
  val altfmt     = if (useMxOPU) Some(Vec(yDim, Bool())) else None // alternate format for outer product
  val fp8        = if (useMxOPU) Some(Vec(yDim, Bool())) else None // FP8 format for outer product
  val fp4        = if (vParams.opuFP4) Some(Bool()) else None // Binary4p2sf pairs (element-format code 6)
  //mvin_col broadcasts vertically
  val mvin_bcast_col = Vec(xDim, Bool())
  val mvin_col   = Vec(xDim, Bool()) // column write
}


class OuterProductUnit(implicit p: Parameters) extends CoreModule()(p) with HasOPUParams {
  val io = IO(new Bundle {
    val op = Input(new OuterProductControl)
    val out = Output(Vec(xDim, UInt(opuParams.cWidth.W)))
    val YOU_SHALL_PASS = Output(Bool())
  })

  // clock gating
  val gated_clock = ClockGate(clock, io.op.clock_enable, "opu_clock_gate")

  // Force OuterProductUnit to have logic to be syn-mappable
  io.YOU_SHALL_PASS := io.op.macc(0) & io.op.macc(0) | io.op.shift(0)
  dontTouch(io.YOU_SHALL_PASS)

  val clusters = Seq.fill(yDim, xDim)(withClock(gated_clock) { Module(new OuterProductCluster) })

  // OPFMACC.S scale cache data, one scale register per entry, on the ungated
  // clock so that fills land whatever the array's clock does. Fills arrive
  // on in_l / in_t as raw register-file data; scales are read with the
  // multiply-accumulate that uses them, a cycle or more after their fill.
  val scale_buf = Reg(Vec(opuParams.nScaleEntries, Vec(egsPerVReg, UInt(dLen.W))))
  when (io.op.sa_fill) { scale_buf(io.op.sa_entry)(io.op.sa_piece) := io.op.in_l.asUInt }
  when (io.op.sb_fill) { scale_buf(io.op.sb_entry)(io.op.sb_piece) := io.op.in_t.asUInt }
  // laid out like the Backend lays out in_l and in_t for a macc
  val sA_bytes = scale_buf(io.op.sa_entry)(io.op.sa_piece).asTypeOf(Vec(yDim * clusterYdim, UInt(8.W)))
  val sB_bytes = scale_buf(io.op.sb_entry)(io.op.sb_piece).asTypeOf(Vec(xDim * clusterXdim, UInt(8.W)))
  val scale_l = Wire(Vec(yDim, Vec(clusterYdim, UInt(8.W))))
  val scale_t = Wire(Vec(xDim, Vec(clusterXdim, UInt(8.W))))
  // The cell at position `local` of quadrant `quad` is row (column)
  // dLenB * quad + local of the tile; in scaled mode it takes scale byte
  // (dLenB * quad + local) >> scale_lg, inside the piece the sequencer chose
  // (with scale_lg = 0, byte `local` of piece `quad`, as OPFMACC.S)
  def pick(bytes: Vec[UInt], quad: Option[UInt], local: Int): UInt = quad match {
    case Some(q) => bytes((((q << dLenOffBits) +& local.U) >> io.op.scale_lg.get)(dLenOffBits - 1, 0))
    case None    => bytes(local)
  }
  for (i <- 0 until yDim; j <- 0 until clusterYdim) {
    scale_l(i)(j) := Mux(io.op.scaled, pick(sA_bytes, io.op.sa_quad, i + j * yDim), 128.U)
  }
  for (i <- 0 until xDim; j <- 0 until clusterXdim) {
    scale_t(i)(j) := Mux(io.op.scaled, pick(sB_bytes, io.op.sb_quad, i + j * xDim), 128.U)
  }

  for (j <- 0 until xDim) {
    for (i <- 0 until yDim) {
      val cluster = clusters(i)(j)
      // column broadcast signals
      cluster.io.in_t      := io.op.in_t(j)
      cluster.io.scale_t   := scale_t(j)
      cluster.io.mvin_bcast_col := io.op.mvin_bcast_col(j)
      cluster.io.mvin_col   := io.op.mvin_col(j)
      // row broadcast signals
      cluster.io.in_l      := io.op.in_l(i)
      cluster.io.scale_l   := scale_l(i)
      cluster.io.mrf_idx    := io.op.mrf_idx(i)
      cluster.io.row_idx    := io.op.row_idx(i)
      cluster.io.col_idx    := io.op.col_idx(i)
      cluster.io.macc       := io.op.macc(i)
      cluster.io.shift      := io.op.shift(i)
      cluster.io.mvin_bcast := io.op.mvin_bcast(i)
      cluster.io.mvin       := io.op.mvin(i)
      if (useMxOPU) {
        cluster.io.altfmt.get := io.op.altfmt.get(i)
        cluster.io.fp8.get := io.op.fp8.get(i)
      }
      cluster.io.fp4.foreach(_ := io.op.fp4.get)
    }

    clusters(0)(j).io.in_pipe := 0.U
    for (i <- 1 until yDim) {
      clusters(i)(j).io.in_pipe := clusters(i-1)(j).io.out_pipe
    }
    io.out(j) := clusters(yDim-1)(j).io.out_pipe
  }
}
