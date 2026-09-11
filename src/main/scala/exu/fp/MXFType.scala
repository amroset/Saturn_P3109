package saturn.exu

import chisel3._
import freechips.rocketchip.tile._

object MXFType {
  val BF16 = new FType(8, 8)
  val E5M3 = new FType(5, 4)
  val E4M3 = new FType(4, 4)
  val E5M2 = new FType(5, 3)
  // 6 exponent bits, 2 fraction bits. Only used by the P3109 build: it is the
  // "wide" rounder that can reach binary8p3's top row (see p3109Fp8.scala).
  val E6M2 = new FType(6, 3)
}
