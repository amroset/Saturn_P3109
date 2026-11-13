package saturn.frontend

import chisel3._
import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket._
import freechips.rocketchip.util._
import saturn.common._
import saturn.insns._

class EarlyVectorDecode(supported_ex_insns: Seq[VectorInstruction])(implicit p: Parameters) extends RocketVectorDecoder()(p) with HasVectorParams {

  io.vector := false.B
  io.legal := false.B
  io.fp := false.B
  io.read_rs1 := false.B
  io.read_rs2 := false.B
  io.read_frs1 := false.B
  io.write_rd := false.B
  io.write_frd := false.B

  val opcode = io.inst(6,0)

  val width = io.inst(14,12)
  val lumop = io.inst(24,20)
  val sumop = lumop
  val vm = io.inst(25)
  val mop = io.inst(27,26)
  val mew = io.inst(28)
  val nf = io.inst(31,29)
  val funct3 = io.inst(14,12)
  val funct6 = io.inst(31,26)
  val rs1 = io.inst(19,15)
  val rs2 = io.inst(24,20)

  val v_load = opcode === opcLoad && !width.isOneOf(1.U, 2.U, 3.U, 4.U)
  val v_store = opcode === opcStore && !width.isOneOf(1.U, 2.U, 3.U, 4.U)
  val v_arith_maybe = opcode === opcVector && funct3 =/= 7.U
  val v_arith = v_arith_maybe && new VectorDecoder(rs1, rs2, funct3, funct6, io.vconfig.vtype.vsew, supported_ex_insns, Nil).matched

  io.vector := v_load || v_store || v_arith_maybe

  // Scaled mode (docs/scaled-mode.md): register uses the pairing rule leaves
  // undefined are illegal. A block operand's group is aligned and its scale
  // group ends by v31; the destination overlaps no scale group; OPFMACC.S and
  // scaled OPFMACC take even vs1 and vs2.
  val scaled_illegal = WireInit(false.B)
  if (vParams.scaledMode) {
    val vscaled = io.vconfig.vtype.fmt_hi(2)
    val rd = io.inst(11,7)
    val opu_scaled = if (vParams.useOpu) funct3 === OPFVV &&
      (funct6 === OPFFunct6.opfmaccs.litValue.U || (funct6 === OPFFunct6.opfmacc.litValue.U && vscaled)) else false.B
    val opu_bad = opu_scaled && (rs1(0) || rs2(0))
    val vec_bad = if (vParams.sharedScaleRegs) false.B else {
      val insns = ScaledMode.insns ++ (if (vParams.scaledDot) ScaledMode.dotInsns else Nil)
      val ctrl = new VectorDecoder(rs1, rs2, funct3, funct6, io.vconfig.vtype.vsew, insns,
        Seq(ReadsVS1, ReadsVS2, Wide2VS2, Wide2VD, FPDot4))
      val l = 1.U(5.W) << scaleLmul(io.vconfig.vtype)
      val vd_len = Mux(ctrl.bool(Wide2VD), l << 1, l)
      def overlaps(a: UInt, alen: UInt, b: UInt, blen: UInt) = a < b +& blen && b < a +& alen
      val blocks = Seq(
        (ctrl.bool(ReadsVS1), rs1),
        (ctrl.bool(ReadsVS2) && !ctrl.bool(Wide2VS2), rs2),
        (!ctrl.bool(Wide2VD) && !ctrl.bool(FPDot4), rd))
      vscaled && ctrl.matched && blocks.map { case (en, r) =>
        en && ((r & (l - 1.U)) =/= 0.U || r +& (l << 1) > 32.U || overlaps(rd, vd_len, r +& l, l))
      }.reduce(_ || _)
    }
    scaled_illegal := v_arith_maybe && (opu_bad || vec_bad)
  }

  when (v_load || v_store) {
    val unit = mop === 0.U
    val whole = unit && ((v_load && lumop === lumopWhole) || (v_store && sumop === sumopWhole))
    io.legal := mew === 0.U && width.isOneOf(0.U, 5.U, 6.U, 7.U) && (!io.vconfig.vtype.vill || whole)
    when (unit) {
      when (v_load && !lumop.isOneOf(lumopUnit, lumopWhole, lumopMask, lumopFF)) { io.legal := false.B }
      when (v_store && !sumop.isOneOf(sumopUnit, sumopWhole, sumopMask)) { io.legal := false.B }
    }
    when (mew === 1.U) { io.legal := false.B }
    io.read_rs1 := true.B
    io.read_rs2 := mop === mopStrided
  } .elsewhen (v_arith) {
    // vfdot4.vx: the packed scalar (four 8-bit codes and a scale) is integer
    // data, read from x[rs1] at full width
    val dot_vx = if (supported_ex_insns.exists(_.lookup(FPDot4).value == 1))
      funct3 === OPFVF && funct6 === OPFFunct6.fdot4.litValue.U else false.B
    io.legal := !io.vconfig.vtype.vill && !scaled_illegal
    io.read_rs1 := funct3.isOneOf(OPIVX, OPMVX) || dot_vx
    io.read_frs1 := funct3 === OPFVF && !dot_vx
    io.write_rd := funct3 === OPMVV && OPMFunct6(funct6) === OPMFunct6.wrxunary0
    io.write_frd := funct3 === OPFVV && OPFFunct6(funct6) === OPFFunct6.wrfunary0
    io.fp := funct3 === OPFVF && !dot_vx
  }
}
